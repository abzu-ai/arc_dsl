import numpy as np

from dataclasses import dataclass

from arc_dsl.errors import DrawError
from .bitmap import Bitmap


@dataclass
class DeclarativeGrid:
    """An ARC grid defined declaratively as an ordered set of bitmaps."""

    h: int
    w: int
    bitmaps: list[Bitmap]

    def __post_init__(self):
        self.h = self.h % 31
        self.w = self.w % 31

    def __eq__(self, other):
        if isinstance(other, DeclarativeGrid):
            x, y = self.draw(), other.draw()
            return (x == y).all()
        if isinstance(other, np.ndarray):
            return (self.draw() == other).all()
        return False

    def draw(self) -> np.ndarray:
        grid = np.zeros((self.h, self.w), dtype=int)
        for bm in self.bitmaps:
            if not isinstance(bm, Bitmap):
                raise DrawError("Expected a Bitmap")

            grid[bm.get_ixs(self.h - 1, self.w - 1)] = bm.color

        return grid

    def to_dsl(self):
        res = f"(MAKE-GRID {self.h} {self.w}"
        for b in self.bitmaps:
            res += "\n    " + b.to_dsl()
        res += ")"

        return res
