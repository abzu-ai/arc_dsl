from dataclasses import dataclass
import math
import random

import numpy as np

from arc_dsl.bitmap import Bitmap
from arc_dsl.grid import DeclarativeGrid


@dataclass
class TrainingSample:
    grid: DeclarativeGrid
    dsl_sentence: str


def normgrid(grid):
    correct_shape = np.zeros((30, 30), dtype=int)
    grid = grid + 1  # keep 0 as pad code

    h, w = grid.shape
    correct_shape[:h, :w] = grid
    return correct_shape.flatten()


def occlude_or_extends_other_objects(bitmap: Bitmap, grid: DeclarativeGrid):
    other_bitmaps: np.ndarray = grid.draw()

    # Coordinates of the bitmap region
    x0 = bitmap.x
    y0 = bitmap.y
    x1 = x0 + bitmap.h
    y1 = y0 + bitmap.w

    # Check if there is anything where the bitmap would be located
    if np.any(other_bitmaps[x0:x1, y0:y1] > 0):
        return True  # Occludes another object on the grid

    # Define the expanded region to check for adjacency
    x0_exp = max(0, x0 - 1)
    y0_exp = max(0, y0 - 1)
    x1_exp = min(grid.h, x1 + 1)
    y1_exp = min(grid.w, y1 + 1)

    # Extract the expanded region
    neighborhood = other_bitmaps[x0_exp:x1_exp, y0_exp:y1_exp]

    # Create a mask to exclude the bitmap region from the expanded region
    mask = np.ones(neighborhood.shape, dtype=bool)
    x0_rel = x0 - x0_exp
    x1_rel = x1 - x0_exp
    y0_rel = y0 - y0_exp
    y1_rel = y1 - y0_exp
    mask[x0_rel:x1_rel, y0_rel:y1_rel] = False  # Exclude bitmap region

    # Check adjacent pixels for the same color
    adjacent_pixels = neighborhood[mask]
    if np.any(adjacent_pixels == bitmap.color):
        return True  # Extends another bitmap

    return False


def sample_multi_bitmap_grid(rng=None) -> TrainingSample:
    rng = rng or random.Random()

    H, W = rng.randint(5, 30), random.randint(5, 30)
    max_bitmaps = math.sqrt(H * W)  # Arbitrary choice
    n_bitmaps = rng.randint(1, int(max_bitmaps))

    current_grid = DeclarativeGrid(H, W, [])

    for _ in range(n_bitmaps):
        bitmap = sample_contiguous(rng, H, W)

        if not occlude_or_extends_other_objects(bitmap, current_grid):
            current_grid.bitmaps.append(bitmap)

    bitmaps_dsl = "".join([b.to_dsl() for b in current_grid.bitmaps])
    return TrainingSample(current_grid, bitmaps_dsl)


def sample_contiguous(rng: random.Random, grid_h=30, grid_w=30):
    # NOTE: LLM produced code. Could be done with way less code, but the llm thing just works as well.
    # Sample an x, y, h, w that fits inside the grid.
    x = rng.randint(0, grid_h - 1)
    y = rng.randint(0, grid_w - 1)
    h = rng.randint(1, grid_h - x)
    w = rng.randint(1, grid_w - y)
    color = rng.randint(1, 9)

    # Initialize empty bitmap
    bitarr = np.zeros((h, w), dtype=bool)

    # Start with a random pixel
    start_i = rng.randint(0, h - 1)
    start_j = rng.randint(0, w - 1)
    bitarr[start_i, start_j] = True

    # Keep track of frontier pixels that can be filled
    frontier = [(start_i, start_j)]

    # While there are frontier pixels, randomly select one and fill a neighbor
    while frontier and rng.random() < 0.8:  # 80% chance to continue growing
        i, j = rng.choice(frontier)

        # Get valid neighbor positions
        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w and not bitarr[ni, nj]:
                neighbors.append((ni, nj))

        # If no valid neighbors, remove from frontier
        if not neighbors:
            frontier.remove((i, j))
            continue

        # Randomly fill one neighbor
        ni, nj = neighbors[rng.randint(0, len(neighbors) - 1)]
        bitarr[ni, nj] = True
        frontier.append((ni, nj))

    # Convert boolean array to bitstring
    bitstring = "".join(map(lambda b: str(int(b)), bitarr.flatten()))

    return Bitmap(
        x=x, y=y, h=h, w=w, color=color, encoding=Bitmap.encode_bitstring(bitstring)
    ).compress()
