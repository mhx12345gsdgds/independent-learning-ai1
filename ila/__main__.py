"""Command-line entry points for training and teacher-free replay."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .experiment import SPLITS, rollout, run, write_csv
from .student import Dynamics, observe
from .teacher import SCRIPTED, ask_ollama, validate
from .world import EDGE, FLOOR, ROCK, make_world


def demo(args):
    torch.set_num_threads(1)
    model = Dynamics()
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model.eval()
    world = make_world(args.split, args.map_id)
    predictions = model.predict_map(observe(world.tiles)).argmax(-1)
    success, trace = rollout(world, predictions, np.random.default_rng(args.map_id), 4 * len(world.tiles))
    grid = np.empty(world.tiles.shape, dtype="U1")
    for tile, symbol in ((FLOOR, "."), (EDGE, "#"), (ROCK, "X")):
        grid[world.tiles == tile] = symbol
    for entry in trace:
        grid[entry["observed"]] = "*"
    grid[world.start], grid[world.goal] = "S", "G"
    print("\n".join(" ".join(row) for row in grid))
    optimal = world.distances(world.start)[world.goal]
    errors = sum(row["predicted"] != row["observed"] for row in trace)
    print(f"Success={success} | steps={len(trace)} | shortest={optimal} | prediction errors={errors}")
    for row in trace:
        print(f'{row["step"]:2} {row["position"]} action={row["action"]} '
              f'predicted={row["predicted"]} observed={row["observed"]}')
    if args.trace:
        write_csv(args.trace, trace)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train_cli = commands.add_parser("run", help="train and compare all cohorts")
    train_cli.add_argument("--out", type=Path, default=Path("runs/first"))
    train_cli.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    train_cli.add_argument("--train-maps", type=int, default=18)
    train_cli.add_argument("--test-maps", type=int, default=24)
    train_cli.add_argument("--steps", type=int, default=80)
    train_cli.add_argument("--updates", type=int, default=20)
    teachers = train_cli.add_mutually_exclusive_group()
    teachers.add_argument("--ollama-model", help="local model name as listed by ollama list")
    teachers.add_argument("--curriculum", type=Path, help="replay a saved teacher curriculum JSON")
    demo_cli = commands.add_parser("demo", help="evaluate one saved student without a teacher")
    demo_cli.add_argument("--weights", required=True, type=Path)
    demo_cli.add_argument("--split", choices=SPLITS, default="shifted")
    demo_cli.add_argument("--map-id", type=int, default=0)
    demo_cli.add_argument("--trace", type=Path)
    args = parser.parse_args()
    if args.command == "demo":
        if args.map_id < 0 or not args.weights.is_file():
            parser.error("demo requires a nonnegative map ID and an existing weights file")
        demo(args)
        return
    if args.train_maps < 6 or min(args.test_maps, args.steps, args.updates) < 1:
        parser.error("use at least 6 training maps and positive test counts, steps, and updates")
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct nonnegative integers")
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    curriculum, source = SCRIPTED, "scripted"
    try:
        if args.ollama_model:
            curriculum, source = ask_ollama(args.ollama_model), f"ollama:{args.ollama_model}"
        elif args.curriculum:
            data = json.loads(args.curriculum.read_text(encoding="utf-8-sig"))
            # A config.json from a previous run can replay its curriculum directly.
            curriculum = validate(data.get("curriculum", data) if isinstance(data, dict) else data)
            source = "replayed:" + (
                str(data.get("teacher_source", "file")) if isinstance(data, dict) else "file"
            )
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(f"could not load teacher curriculum: {error}")
    run(args, curriculum, source)


if __name__ == "__main__":
    main()
