from dataclasses import dataclass, field

import numpy as np

from embloks.config import Settings
from embloks.grid import Grid
from embloks.materials import Material, Orientation


@dataclass
class Brush:
    settings: Settings
    tool: Material = Material.CONDUCTOR
    # 0 paints a single cell; r paints a disc of radius r cells.
    radius: int = 0
    orientation: Orientation = Orientation.RIGHT
    # Indices into the settings' value ladders.
    resistance_index: int = field(init=False)
    emf_index: int = field(init=False)

    def __post_init__(self):
        s = self.settings
        self.resistance_index = s.resistance_values.index(s.default_resistance)
        self.emf_index = s.emf_values.index(s.default_emf)

    @property
    def resistance(self) -> float:
        return self.settings.resistance_values[self.resistance_index]

    @property
    def emf(self) -> float:
        return self.settings.emf_values[self.emf_index]

    def resize(self, delta: int) -> None:
        self.radius = max(0, min(self.settings.max_brush_radius, self.radius + delta))

    def rotate(self) -> None:
        self.orientation = self.orientation.rotated()

    def adjust_value(self, delta: int) -> None:
        """Step the current tool's value (resistance or EMF) up or down its ladder."""
        s = self.settings
        if self.tool == Material.CONDUCTOR:
            self.resistance_index = _clamp_index(self.resistance_index + delta, s.resistance_values)
        elif self.tool == Material.BATTERY:
            self.emf_index = _clamp_index(self.emf_index + delta, s.emf_values)

    def footprint(self, center: tuple[int, int], grid: Grid) -> tuple[np.ndarray, np.ndarray]:
        """Columns and rows of the cells covered by the brush at `center`, clipped to the grid."""
        r = self.radius
        dc, dr = np.mgrid[-r : r + 1, -r : r + 1]
        # (r + 0.5)^2 rounds the disc nicely: r=1 gives a full 3x3 block.
        inside = dc**2 + dr**2 <= (r + 0.5) ** 2
        cols = center[0] + dc[inside]
        rows = center[1] + dr[inside]
        on_grid = (cols >= 0) & (cols < grid.cols) & (rows >= 0) & (rows < grid.rows)
        return cols[on_grid], rows[on_grid]


def _clamp_index(index: int, values: tuple) -> int:
    return max(0, min(len(values) - 1, index))


def line_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """Cells from start to end, inclusive, where consecutive cells share an edge.

    Unlike Bresenham, this never steps diagonally, so a dragged one-cell-wide
    wire is always 4-connected (diagonal contact doesn't conduct).
    """
    (x, y), (x1, y1) = start, end
    dx, dy = abs(x1 - x), abs(y1 - y)
    sx = 1 if x1 > x else -1
    sy = 1 if y1 > y else -1
    cells = [(x, y)]
    ix = iy = 0
    while ix < dx or iy < dy:
        # Step along whichever axis the ideal line crosses a cell border on first.
        if (1 + 2 * ix) * dy < (1 + 2 * iy) * dx:
            x += sx
            ix += 1
        else:
            y += sy
            iy += 1
        cells.append((x, y))
    return cells
