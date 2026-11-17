"""Ground-truth simulator. The student receives observations and outcomes."""

from collections import deque
from dataclasses import dataclass

import numpy as np

# Action identifiers are opaque to the student.
DISPLACEMENTS = ((0, 0), (0, 1), (-1, 0), (0, -1), (1, 0))
ACTION_OFFSETS = ((-1, 0), (1, 0), (0, 1), (0, -1))
ACTION_COUNT = 4
TILE_COUNT = 3
FLOOR, EDGE, ROCK = 0, 1, 2


@dataclass
class World:
    tiles: np.ndarray
    start: tuple[int, int]
    goal: tuple[int, int]

    def step(self, position, action):
        dr, dc = ACTION_OFFSETS[action]
        r, c = position[0] + dr, position[1] + dc
        if 0 <= r < self.tiles.shape[0] and 0 <= c < self.tiles.shape[1]:
            if self.tiles[r, c] == FLOOR:
                return (r, c)
        return position

    def distances(self, start):
        distances = {start: 0}
        queue = deque([start])
        while queue:
            position = queue.popleft()
            for action in range(ACTION_COUNT):
                neighbor = self.step(position, action)
                if neighbor not in distances:
                    distances[neighbor] = distances[position] + 1
                    queue.append(neighbor)
        return distances


MAP_SETTINGS = {"train": (8, 0.18), "heldout": (8, 0.18), "shifted": (11, 0.28)}
MAP_SCOPES = {"train": 0, "heldout": 1, "shifted": 2}


def make_world(split, map_id, training_seed=0):
    size, density = MAP_SETTINGS[split]
    seed = training_seed if split == "train" else 0
    rng = np.random.default_rng([seed, map_id, MAP_SCOPES[split]])
    for _ in range(1000):
        tiles = np.where(rng.random((size, size)) < density, ROCK, FLOOR).astype(np.uint8)
        tiles[[0, -1], :] = EDGE
        tiles[:, [0, -1]] = EDGE
        free = np.argwhere(tiles == FLOOR)
        if len(free) < 2:
            continue
        start, goal = [tuple(map(int, point)) for point in free[rng.choice(len(free), 2, replace=False)]]
        world = World(tiles, start, goal)
        if world.distances(start).get(goal, 0) >= size - 2:
            return world
    raise RuntimeError(f"Could not generate {split} map {map_id}")
