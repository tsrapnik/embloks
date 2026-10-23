from enum import IntEnum

import numpy as np


class Material(IntEnum):
    VACUUM = 0
    CONDUCTOR = 1
    BATTERY = 2


class Orientation(IntEnum):
    """Direction of a battery's + terminal, in screen coordinates (y points down)."""

    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

    @property
    def vector(self) -> tuple[int, int]:
        return ((1, 0), (0, 1), (-1, 0), (0, -1))[self]

    def rotated(self) -> "Orientation":
        """The next orientation, clockwise on screen."""
        return Orientation((self + 1) % 4)


class Materials:
    """Per-cell material properties, as arrays of shape (cols, rows) indexed [col, row]."""

    def __init__(self, cols: int, rows: int):
        self.cols = cols
        self.rows = rows
        self.kind = np.zeros((cols, rows), dtype=np.int8)
        # Only meaningful for conductors.
        self.resistance = np.zeros((cols, rows))
        # Only meaningful for batteries.
        self.emf = np.zeros((cols, rows))
        self.orientation = np.zeros((cols, rows), dtype=np.int8)
        # Bumped on every change, so caches (rendering, physics) know when to rebuild.
        self.version = 0

    @property
    def conductive(self) -> np.ndarray:
        """Cells that hold a fixed ion and mobile electrons (conductors and batteries)."""
        return self.kind != Material.VACUUM

    def paint(
        self,
        cols: np.ndarray,
        rows: np.ndarray,
        material: Material,
        resistance: float = 0.0,
        emf: float = 0.0,
        orientation: Orientation = Orientation.RIGHT,
    ) -> None:
        """Set the given cells to `material`, clearing properties it doesn't use."""
        self.kind[cols, rows] = material
        self.resistance[cols, rows] = resistance if material == Material.CONDUCTOR else 0.0
        is_battery = material == Material.BATTERY
        self.emf[cols, rows] = emf if is_battery else 0.0
        self.orientation[cols, rows] = orientation if is_battery else 0
        self.version += 1

    def clear(self) -> None:
        all_cols, all_rows = np.indices((self.cols, self.rows))
        self.paint(all_cols.ravel(), all_rows.ravel(), Material.VACUUM)
