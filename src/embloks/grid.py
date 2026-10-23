from dataclasses import dataclass


@dataclass(frozen=True)
class Grid:
    """A grid of square cells laid out on the screen.

    Cells are addressed as (col, row), with (0, 0) top-left. The grid is
    centered in the window; leftover pixels form a margin.
    """

    cols: int
    rows: int
    cell_px: int
    origin_x: int
    origin_y: int

    @classmethod
    def fit(cls, density: int, width: int, height: int) -> "Grid":
        """Build a grid of roughly `density` columns filling a width x height area.

        Cells are a whole number of pixels, so the actual column count can be
        higher than requested (e.g. 150 -> 6 px cells -> 166 columns at 1000 px).
        """
        cell_px = max(1, width // density)
        cols = width // cell_px
        rows = max(1, height // cell_px)
        origin_x = (width - cols * cell_px) // 2
        origin_y = (height - rows * cell_px) // 2
        return cls(cols, rows, cell_px, origin_x, origin_y)

    @property
    def pixel_width(self) -> int:
        return self.cols * self.cell_px

    @property
    def pixel_height(self) -> int:
        return self.rows * self.cell_px

    def cell_at(self, x: float, y: float) -> tuple[int, int] | None:
        """The cell under screen point (x, y), or None outside the grid."""
        col = int((x - self.origin_x) // self.cell_px)
        row = int((y - self.origin_y) // self.cell_px)
        if 0 <= col < self.cols and 0 <= row < self.rows:
            return col, row
        return None

    def cell_rect(self, col: int, row: int) -> tuple[int, int, int, int]:
        """Screen rectangle (x, y, w, h) of a cell."""
        return (
            self.origin_x + col * self.cell_px,
            self.origin_y + row * self.cell_px,
            self.cell_px,
            self.cell_px,
        )
