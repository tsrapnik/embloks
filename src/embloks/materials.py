from enum import IntEnum

import numpy as np


class Material(IntEnum):
    VACUUM = 0
    CONDUCTOR = 1
    BATTERY = 2


# Unit vector towards the + terminal for each Orientation value, for vectorised lookups.
ORIENTATION_VECTORS = np.array([(1, 0), (0, 1), (-1, 0), (0, -1)], dtype=float)


class Orientation(IntEnum):
    """Direction of a battery's + terminal, in screen coordinates (y points down)."""

    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

    @property
    def vector(self) -> tuple[int, int]:
        x, y = ORIENTATION_VECTORS[self]
        return int(x), int(y)

    def rotated(self) -> "Orientation":
        """The next orientation, clockwise on screen."""
        return Orientation((self + 1) % 4)


class Materials:
    """Per-cell material properties, as arrays of shape (cols, rows) indexed [col, row]."""

    def __init__(self, cols: int, rows: int):
        self.cols = cols
        self.rows = rows
        self.kind = np.zeros((cols, rows), dtype=np.int8)
        # Conductors and batteries (a battery's internal resistance).
        self.resistance = np.zeros((cols, rows))
        # Charge density of conductive cells, in units of a wire's: ion charge and
        # electrons per cell are this many times cell_charge and carriers_per_cell.
        self.density = np.zeros((cols, rows))
        # Only meaningful for batteries.
        self.emf = np.zeros((cols, rows))
        self.orientation = np.zeros((cols, rows), dtype=np.int8)
        # AC frequency (cycles per time unit); 0 is a DC battery.
        self.frequency = np.zeros((cols, rows))
        # Bumped on every change, so caches (rendering, physics) know when to rebuild.
        self.version = 0

    @property
    def conductive(self) -> np.ndarray:
        """Cells that hold a fixed ion and mobile electrons: conductors and batteries.

        A battery conducts like a wire, but also pushes its electrons towards its
        - side (see battery.py)."""
        return (self.kind == Material.CONDUCTOR) | (self.kind == Material.BATTERY)

    def paint(
        self,
        cols: np.ndarray,
        rows: np.ndarray,
        material: Material,
        resistance: float = 0.0,
        emf: float = 0.0,
        orientation: Orientation = Orientation.RIGHT,
        frequency: float = 0.0,
        density: float = 1.0,
    ) -> None:
        """Set the given cells to `material`, clearing properties it doesn't use."""
        self.kind[cols, rows] = material
        is_battery = material == Material.BATTERY
        is_conductive = material == Material.CONDUCTOR or is_battery
        self.resistance[cols, rows] = resistance if is_conductive else 0.0
        self.density[cols, rows] = density if is_conductive else 0.0
        self.emf[cols, rows] = emf if is_battery else 0.0
        self.orientation[cols, rows] = orientation if is_battery else 0
        self.frequency[cols, rows] = frequency if is_battery else 0.0
        self.version += 1

    def clear(self) -> None:
        all_cols, all_rows = np.indices((self.cols, self.rows))
        self.paint(all_cols.ravel(), all_rows.ravel(), Material.VACUUM)
