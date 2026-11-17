"""Experience replay, action-conditioned dynamics, and model-based planning."""

from collections import deque

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .world import ACTION_COUNT, DISPLACEMENTS, EDGE, TILE_COUNT

OUTCOME_IDS = {offset: index for index, offset in enumerate(DISPLACEMENTS)}


def observe(tiles):
    padded = np.pad(tiles, 1, constant_values=EDGE)
    patches = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    return patches.reshape(*tiles.shape, 9)


def outcome(before, after):
    return OUTCOME_IDS[(after[0] - before[0], after[1] - before[1])]


class Replay:
    def __init__(self):
        self.patches = []
        self.actions = []
        self.outcomes = []
        self.map_ids = []
        self.positions = []
        self.next_positions = []

    def add(self, patch, action, before, after, map_id):
        self.patches.append(patch)
        self.actions.append(action)
        self.outcomes.append(outcome(before, after))
        self.map_ids.append(map_id)
        self.positions.append(before)
        self.next_positions.append(after)

    def save(self, path):
        np.savez_compressed(
            path, patches=self.patches, actions=self.actions, outcomes=self.outcomes,
            map_ids=self.map_ids, positions=self.positions, next_positions=self.next_positions,
        )


class Dynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9 * TILE_COUNT + ACTION_COUNT, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, len(DISPLACEMENTS)),
        )

    def forward(self, patches, actions):
        tile_features = F.one_hot(patches, TILE_COUNT).flatten(1)
        action_features = F.one_hot(actions, ACTION_COUNT)
        return self.net(torch.cat((tile_features, action_features), dim=1).float())

    def learn(self, replay, optimizer, rng, updates):
        patches = torch.as_tensor(np.asarray(replay.patches), dtype=torch.long)
        actions = torch.tensor(replay.actions, dtype=torch.long)
        labels = torch.tensor(replay.outcomes, dtype=torch.long)
        for _ in range(updates):
            indices = rng.integers(len(labels), size=64)
            optimizer.zero_grad()
            loss = F.cross_entropy(self(patches[indices], actions[indices]), labels[indices])
            loss.backward()
            optimizer.step()

    @torch.inference_mode()
    def predict_map(self, patches):
        features = torch.tensor(patches.reshape(-1, 9), dtype=torch.long)
        actions = torch.arange(ACTION_COUNT).repeat(len(features))
        logits = self(features.repeat_interleave(ACTION_COUNT, dim=0), actions)
        return logits.softmax(-1).numpy().reshape(*patches.shape[:2], ACTION_COUNT, -1)


def planned_actions(predictions, goal):
    """Build a shortest-path policy from predicted edges via reverse BFS."""
    rows, cols, _ = predictions.shape
    incoming = {}
    for r, c in np.ndindex(rows, cols):
        for action, result in enumerate(predictions[r, c]):
            dr, dc = DISPLACEMENTS[result]
            target = (r + dr, c + dc)
            if target != (r, c) and 0 <= target[0] < rows and 0 <= target[1] < cols:
                incoming.setdefault(target, []).append(((r, c), action))
    queue = deque([goal])
    visited = {goal}
    policy = {}
    while queue:
        target = queue.popleft()
        for position, action in incoming.get(target, ()):
            if position not in visited:
                policy[position] = action
                visited.add(position)
                queue.append(position)
    return policy
