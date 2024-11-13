import numpy as np
import matplotlib.pyplot as plt

from .interpreter import parse_sentence, eval_program
from .errors import *

COLORS = {
    0: (0, 0, 0),
    1: (0, 0.454, 0.851),  # blue
    2: (1, 0.255, 0.212),  # red
    3: (0.18, 0.8, 0.251),  # green
    4: (1, 0.863, 0),  # yellow
    5: (0.667, 0.667, 0.667),  # grey
    6: (0.941, 0.071, 0.745),  # fuchsia
    7: (1, 0.522, 0.106),  # orange
    8: (0.498, 0.988, 1),  # teal
    9: (0.529, 0.047, 0.145),  # brown
}


def gridstr_coloured(grid) -> str:
    res = ""
    for row in grid:
        for element in row:
            r, g, b = COLORS[element]
            r, g, b = int(r * 255), int(g * 255), int(b * 255)
            res += f"\033[38;2;{r};{g};{b}m█\033[0m"
        res += "\n"
    return res


def image(grid):
    # Constants
    CELLSIZE = 20
    BORDERCOLOR = (0.333, 0.333, 0.333)  # RGB equivalent of 0.333, 0.333, 0.333
    BORDERWIDTH = 2

    rows, cols = grid.shape
    img_size = (
        rows * (CELLSIZE + BORDERWIDTH) + BORDERWIDTH,
        cols * (CELLSIZE + BORDERWIDTH) + BORDERWIDTH,
    )

    # Create the image with border color
    img = np.full((*img_size, 3), BORDERCOLOR, dtype=float)

    for i in range(rows):
        for j in range(cols):
            row_start = i * (CELLSIZE + BORDERWIDTH) + BORDERWIDTH
            row_end = row_start + CELLSIZE
            col_start = j * (CELLSIZE + BORDERWIDTH) + BORDERWIDTH
            col_end = col_start + CELLSIZE

            img[row_start:row_end, col_start:col_end, :] = COLORS[grid[i, j]]

    return img


def show_grids(**grids):
    title = grids.pop("title", None)

    num_images = len(grids)
    _, axes = plt.subplots(ncols=num_images, figsize=(12, 5))

    if num_images == 1:
        axes = [axes]

    for i, (grid_title, g) in enumerate(grids.items()):
        axes[i].imshow(image(g))
        axes[i].axis("off")
        axes[i].set_title(grid_title)

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()


def print_lists_tree(form, indent=""):
    if isinstance(form, list):
        for _form in form:
            print_lists_tree(_form, indent + "  ")
    else:
        print(f"{indent}{form}")
