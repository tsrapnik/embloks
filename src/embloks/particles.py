import numpy as np


class Particles:
    """Mobile charges (electrons), stored as parallel arrays.

    Positions are in cell units: cell (col, row) spans [col, col + 1) x [row, row + 1),
    with y pointing down like the screen. Fixed ions are not stored here; every
    conductive cell implicitly has one spread over it (see World).
    """

    def __init__(self):
        self.pos = np.empty((0, 2))
        self.vel = np.empty((0, 2))
        self.charge = np.empty(0)
        self.mass = np.empty(0)

    def __len__(self) -> int:
        return len(self.charge)

    def add(self, pos: np.ndarray, charge: float, mass: float) -> None:
        n = len(pos)
        self.pos = np.concatenate([self.pos, pos])
        self.vel = np.concatenate([self.vel, np.zeros((n, 2))])
        self.charge = np.concatenate([self.charge, np.full(n, charge)])
        self.mass = np.concatenate([self.mass, np.full(n, mass)])

    def keep(self, mask: np.ndarray) -> None:
        """Drop every particle where `mask` is False."""
        self.pos = self.pos[mask]
        self.vel = self.vel[mask]
        self.charge = self.charge[mask]
        self.mass = self.mass[mask]

    def cells(self) -> tuple[np.ndarray, np.ndarray]:
        """Column and row of the cell each particle is in."""
        cell = np.floor(self.pos).astype(np.int64)
        return cell[:, 0], cell[:, 1]
