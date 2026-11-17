"""Train matched cohorts and measure transfer with frozen checkpoints."""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .student import Dynamics, Replay, observe, outcome, planned_actions
from .world import ACTION_COUNT, DISPLACEMENTS, MAP_SETTINGS, make_world

ARMS = ("random", "autonomous", "guided")
SPLITS = ("heldout", "shifted")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def explore_action(mode, position, patch, local_counts, patch_counts, entropy, rng):
    if mode == "random":
        return int(rng.integers(ACTION_COUNT))
    if mode == "coverage":
        scores = [-local_counts[position, action] for action in range(ACTION_COUNT)]
    elif mode == "novelty":
        scores = [-patch_counts[patch, action] for action in range(ACTION_COUNT)]
    else:
        scores = entropy[position]
    return int(rng.choice(np.flatnonzero(scores == np.max(scores))))


def train(arm, seed, args, curriculum, run_dir, test_maps):
    torch.manual_seed(seed)
    action_rng, update_rng, teacher_rng = [
        np.random.default_rng([seed, stream]) for stream in range(3)
    ]
    model = Dynamics()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    replay = Replay()
    patch_counts = Counter()
    history = []
    metrics = []
    checkpoints = sorted({1, max(1, args.train_maps // 3),
                          max(1, args.train_maps // 2), args.train_maps})
    guidance_maps = args.train_maps // 2
    for map_id in range(args.train_maps):
        world = make_world("train", map_id, seed)
        patches = observe(world.tiles)
        position = world.start
        local_counts = Counter()
        teacher_steps = 0
        mode = "random" if arm == "random" or map_id == 0 else "coverage"
        teacher_mode, probability = None, 0.0
        if arm == "guided" and 0 < map_id < guidance_maps:
            probability = 1.0 - (map_id - 1) / (guidance_maps - 1)
            teacher_mode = curriculum["first" if map_id <= guidance_maps // 2 else "second"]
        entropy = None
        if teacher_mode == "uncertainty":
            probabilities = model.predict_map(patches)
            entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=-1)
        for _ in range(args.steps):
            use_teacher = probability > 0 and teacher_rng.random() < probability
            patch = patches[position]
            patch_key = tuple(patch)
            action = explore_action(teacher_mode if use_teacher else mode, position,
                                    patch_key, local_counts, patch_counts, entropy, action_rng)
            next_position = world.step(position, action)
            replay.add(patch, action, position, next_position, map_id)
            local_counts[position, action] += 1
            patch_counts[patch_key, action] += 1
            position = next_position
            teacher_steps += int(use_teacher)
        model.learn(replay, optimizer, update_rng, args.updates)
        history.append({
            "map_id": map_id, "interactions": len(replay.actions), "teacher_mode": teacher_mode,
            "teacher_probability": probability, "teacher_steps": teacher_steps,
            "unique_position_actions": len(local_counts), "unique_patch_actions": len(patch_counts),
        })
        if map_id + 1 in checkpoints:
            metrics.extend(assess(model, arm, seed, len(replay.actions), test_maps))
    arm_path = run_dir / f"seed-{seed}" / arm
    arm_path.mkdir(parents=True)
    torch.save(model.state_dict(), arm_path / "weights.pt")
    replay.save(arm_path / "experience.npz")
    write_csv(arm_path / "training.csv", history)
    return metrics


def rollout(world, predictions, rng, budget):
    policy = {} if predictions is None else planned_actions(predictions, world.goal)
    position = world.start
    trace = []
    for step in range(1, budget + 1):
        action = policy.get(position)
        if action is None:
            action = int(rng.integers(ACTION_COUNT))
        predicted = None
        if predictions is not None:
            dr, dc = DISPLACEMENTS[predictions[position][action]]
            predicted = (position[0] + dr, position[1] + dc)
        next_position = world.step(position, action)
        trace.append({
            "step": step, "position": position, "action": action,
            "predicted": predicted, "observed": next_position,
        })
        position = next_position
        if position == world.goal:
            break
    return position == world.goal, trace


def assess(model, arm, seed, interactions, test_maps):
    rows = []
    for split, map_id, world, probes, optimal in test_maps:
        predictions = None
        accuracy = None
        blocked_recall = None
        move_accuracy = None
        if model is not None:
            predictions = model.predict_map(observe(world.tiles)).argmax(-1)
            correct = [predictions[position][action] == label for position, action, label in probes]
            accuracy = float(np.mean(correct))
            blocked_recall = float(np.mean([hit for hit, probe in zip(correct, probes) if probe[2] == 0]))
            move_accuracy = float(np.mean([hit for hit, probe in zip(correct, probes) if probe[2] != 0]))
        rng = np.random.default_rng([seed, map_id, SPLITS.index(split)])
        success, trace = rollout(world, predictions, rng, 4 * world.tiles.shape[0])
        rows.append({
            "arm": arm, "seed": seed, "interactions": interactions, "split": split, "map_id": map_id,
            "success": int(success), "steps": len(trace), "optimal_steps": optimal,
            "spl": optimal / max(optimal, len(trace)) if success else 0.0,
            "next_state_accuracy": accuracy, "blocked_recall": blocked_recall,
            "move_accuracy": move_accuracy,
        })
    return rows


def build_test_maps(count):
    test_maps = []
    for split in SPLITS:
        for map_id in range(count):
            world = make_world(split, map_id)
            reachable = world.distances(world.start)
            probes = [
                (position, action, outcome(position, world.step(position, action)))
                for position in reachable for action in range(ACTION_COUNT)
            ]
            test_maps.append((split, map_id, world, probes, reachable[world.goal]))
    return test_maps


def summarize(rows):
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in ("arm", "interactions", "split"))
        groups.setdefault(key, []).append(row)
    summary = []
    for (arm, interactions, split), group in groups.items():
        seeds = sorted({row["seed"] for row in group})
        result = {
            "arm": arm, "interactions": interactions, "split": split, "episodes": len(group),
            "seeds": len(seeds),
        }
        for metric in ("success", "spl", "next_state_accuracy", "blocked_recall", "move_accuracy"):
            values = [row[metric] for row in group if row[metric] is not None]
            result[metric] = float(np.mean(values)) if values else None
        by_seed = [np.mean([row["success"] for row in group if row["seed"] == seed]) for seed in seeds]
        result["success_sd_across_seeds"] = float(np.std(by_seed, ddof=1)) if len(seeds) > 1 else None
        summary.append(result)
    return summary


def run(args, curriculum, teacher_source):
    torch.set_num_threads(1)
    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", {
        "seeds": args.seeds, "train_maps": args.train_maps, "test_maps_per_split": args.test_maps,
        "steps_per_training_map": args.steps, "updates_per_training_map": args.updates,
        "map_settings": MAP_SETTINGS, "teacher_source": teacher_source, "curriculum": curriculum,
        "torch": str(torch.__version__), "numpy": np.__version__,
    })
    test_maps = build_test_maps(args.test_maps)
    rows = []
    for seed in args.seeds:
        rows.extend(assess(None, "random_policy", seed, 0, test_maps))
        for arm in ARMS:
            print(f"Training seed={seed}, arm={arm}", flush=True)
            rows.extend(train(arm, seed, args, curriculum, run_dir, test_maps))
    write_csv(run_dir / "episodes.csv", rows)
    summary = summarize(rows)
    write_csv(run_dir / "summary.csv", summary)
    print("\nFinal checkpoint (success / SPL / next-state accuracy)")
    budget = args.train_maps * args.steps
    for row in summary:
        if row["interactions"] in (0, budget):
            accuracy = row["next_state_accuracy"]
            formatted = "--" if accuracy is None else f"{accuracy:.1%}"
            print(f'{row["arm"]:14} {row["split"]:8} {row["success"]:6.1%} / '
                  f'{row["spl"]:6.1%} / {formatted}')
    print(f"Saved results to {run_dir.resolve()}")
