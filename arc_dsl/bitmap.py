from dataclasses import dataclass

import numpy as np

COLOR_NAMES = {
    0: "BLACK",
    1: "BLUE",
    2: "RED",
    3: "GREEN",
    4: "YELLOW",
    5: "GREY",
    6: "FUCHSIA",
    7: "ORANGE",
    8: "TEAL",
    9: "BROWN",
}


@dataclass
class Bitmap:
    x: int
    y: int
    h: int
    w: int
    color: int
    encoding: list[int]

    def __post_init__(self):
        self.color = self.color % 10
        self.x = self.x % 30
        self.y = self.y % 30
        self.w = self.w % 31
        self.h = self.h % 31
        self.encoding = [c % 16 for c in self.encoding]

    def get_ixs(self, max_i=29, max_j=29):
        i_cords = []
        j_cords = []
        total_bits = self.w * self.h
        bits_per_chunk = 4
        n = 0  # Overall bit index

        for num in self.encoding:
            for i in range(bits_per_chunk):
                if n >= total_bits:
                    break  # Stop if we've processed all bits
                # Extract the bit at the current position
                bit = (num >> (bits_per_chunk - 1 - i)) & 1
                if bit == 1:
                    i_cord = self.x + n // self.w
                    j_cord = self.y + n % self.w
                    if i_cord <= max_i and j_cord <= max_j:
                        i_cords.append(i_cord)
                        j_cords.append(j_cord)
                n += 1
        return (i_cords, j_cords)

    def decode(self):
        total_bits = self.w * self.h

        bits_per_chunk = 4
        bitstring = ""
        for num in self.encoding:
            bitstring += format(num, f"0{bits_per_chunk}b")
            bitstring = bitstring[:total_bits]

        return bitstring

    def compress(self):
        bitstr = self.decode()

        if "1" not in bitstr:
            return Bitmap(0, 0, 0, 0, 0, [])

        bitarr = np.fromiter(map(int, bitstr), dtype=bool).reshape(self.h, self.w)

        # Find columns and rows with any '1's
        col_mask = np.any(bitarr, axis=0)
        row_mask = np.any(bitarr, axis=1)

        col_indices = np.where(col_mask)[0]
        row_indices = np.where(row_mask)[0]

        first_col_left = col_indices[0]
        first_col_right = col_indices[-1]
        first_row_top = row_indices[0]
        first_row_btm = row_indices[-1]

        # Slice the array
        bitarr = bitarr[
            first_row_top : first_row_btm + 1, first_col_left : first_col_right + 1
        ]

        new_h, new_w = bitarr.shape

        # Flatten and get bitstring
        bitstr = "".join(bitarr.flatten().astype(int).astype(str))

        new_encoding = Bitmap.encode_bitstring(bitstr)

        return Bitmap(
            x=self.x + first_row_top,
            y=self.y + first_col_left,
            h=new_h,
            w=new_w,
            color=self.color,
            encoding=new_encoding,
        )

    @staticmethod
    def from_bitstring(bitstring, x, y, h, w, color):
        good = ("0", "1")
        bitstring = "".join(b for b in bitstring if b in good)
        encoding = Bitmap.encode_bitstring(bitstring)
        return Bitmap(x, y, h, w, color, encoding)

    @staticmethod
    def encode_bitstring(bitstring) -> list[int]:
        bits_per_chunk = 4  # Using 4 bits per integer (values 0-15)
        # Pad the bitstring with zeros to make its length a multiple of bits_per_chunk
        padding_length = (-len(bitstring)) % bits_per_chunk
        bitstring += "0" * padding_length
        encoded = []
        for i in range(0, len(bitstring), bits_per_chunk):
            chunk = bitstring[i : i + bits_per_chunk]
            num = int(chunk, 2)
            encoded.append(num)
        return encoded

    def to_dsl(self):
        """
        Outputs: `(MAKE-BITMAP {x} {y} {h} {w} {color} {enc[0]} {enc[1]} ... {enc[k]})`
        """
        return f"(MAKE-BITMAP {self.x} {self.y} {self.h} {self.w} {COLOR_NAMES[self.color]} {' '.join(map(str, self.encoding))})"
