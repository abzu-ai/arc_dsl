import pytest

from arc_dsl.bitmap import Bitmap
from arc_dsl.grid import DeclarativeGrid
from arc_dsl.interpreter import eval_sentence


@pytest.mark.parametrize(
    "wasteful_bm,compressed_bm",
    [
        (
            Bitmap.from_bitstring(
                "0000 0000 0110 0110 0000", h=5, w=4, x=0, y=1, color=1
            ),
            Bitmap.from_bitstring("1111", x=2, y=2, h=2, w=2, color=1),
        ),
        (
            Bitmap.from_bitstring("11 10 00", h=3, w=2, x=1, y=1, color=1),
            Bitmap.from_bitstring("1110", x=1, y=1, h=2, w=2, color=1),
        ),
        (
            Bitmap.from_bitstring("011 001", h=2, w=3, x=1, y=1, color=1),
            Bitmap.from_bitstring("11 01", x=1, y=2, h=2, w=2, color=1),
        ),
        (
            Bitmap.from_bitstring("00 10 01", h=3, w=2, x=1, y=1, color=1),
            Bitmap.from_bitstring("10 01", x=2, y=1, h=2, w=2, color=1),
        ),
        (
            Bitmap.from_bitstring("100 010", h=2, w=3, x=1, y=1, color=1),
            Bitmap.from_bitstring("10 01", x=1, y=1, h=2, w=2, color=1),
        ),
    ],
)
def test_bitmap_compression(wasteful_bm: Bitmap, compressed_bm: Bitmap):
    assert wasteful_bm.compress() == compressed_bm

    grid_before = DeclarativeGrid(30, 30, [wasteful_bm]).draw()
    grid_after = DeclarativeGrid(30, 30, [compressed_bm]).draw()
    assert (grid_after == grid_before).all()


@pytest.mark.parametrize(
    "bitmap_redundant",
    [
        Bitmap(1, 2, 3, 4, 2, [11, 10, 13, 8]),
        Bitmap(1, 2, 3, 4, 2, [11, 10, 13, 8, 0, 0, 0, 0]),
        Bitmap(1, 2, 3, 4, 2, [27, 26, 29, 24]),
        Bitmap(4, 3, 3, 4, 2, [11, 10, 13, 8]),
        Bitmap(1, 2, 3, 4, 12, [11, 10, 13, 8]),
        Bitmap(1, 2, 3, 4, 2, [255, 255, 255, 255]),
        Bitmap(1, 2, 3, 4, 2, [0, 0, 0, 0]),
        Bitmap(1, 2, 3, 4, 2, [5, 10, 5, 10]),
        Bitmap(1, 2, 3, 4, 2, [1, 2, 1, 2, 1, 2]),
        Bitmap(1, 2, 3, 4, 2, [123, 456, 789, 101112]),
    ],
)
def test_bitmap_superfluous_encoding_removal(bitmap_redundant: Bitmap):
    bitmap_clean = bitmap_redundant.compress()

    assert (
        bitmap_redundant.encoding != bitmap_clean.encoding
    ), "Expected encoding to change"

    grid_size = (15, 15)
    grid_redundant = DeclarativeGrid(*grid_size, [bitmap_redundant]).draw()
    grid_clean = DeclarativeGrid(*grid_size, [bitmap_clean]).draw()

    assert (
        grid_redundant == grid_clean
    ).all(), "Redundancy removal does not change resulting grid"


def test_to_dsl():
    reference_sentence = "(MAKE-GRID 15 15 (MAKE-BITMAP 2 3 5 4 FUCHSIA 15 7 8))"
    bitmap = Bitmap(x=2, y=3, h=5, w=4, color=6, encoding=[15, 7, 8])
    sentence = f"(MAKE-GRID 15 15 {bitmap.to_dsl()})"

    assert eval_sentence(sentence) == eval_sentence(reference_sentence)
    assert sentence == reference_sentence
