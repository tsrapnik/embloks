"""Putting charges on the grid, and the Poynting vector.

The fields themselves are computed in maxwell.py; close-range corrections between
electrons in close_range.py.
"""

import numpy as np

from embloks.config import Settings


class CloudInCell:
    """Bilinear (cloud-in-cell) weights of particles onto the 4 nearest cell centres.

    Computed once per particle configuration, then reused to deposit charge and
    current. Corners outside the grid are clamped onto the edge cells, so weights
    per particle still sum to 1.
    """

    def __init__(self, positions: np.ndarray, cols: int, rows: int):
        self.shape = (cols, rows)
        u = positions[:, 0] - 0.5
        v = positions[:, 1] - 0.5
        c0, r0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
        fu, fv = u - c0, v - r0
        index, weight = [], []
        for dc, wc in ((0, 1 - fu), (1, fu)):
            c = np.clip(c0 + dc, 0, cols - 1)
            for dr, wr in ((0, 1 - fv), (1, fv)):
                index.append(c * rows + np.clip(r0 + dr, 0, rows - 1))
                weight.append(wc * wr)
        # Shape (4, particles): one row per corner.
        self.index = np.array(index)
        self.weight = np.array(weight)

    def deposit(self, values: np.ndarray) -> np.ndarray:
        """Spread values per particle, (N,) or (N, d), over the grid: (cols, rows) or
        (cols, rows, d). Conserves the sum."""
        size = self.shape[0] * self.shape[1]
        columns = values.reshape(len(values), int(np.prod(values.shape[1:])))
        flat = np.stack([
            np.bincount(self.index.ravel(), (self.weight * column).ravel(), minlength=size)
            for column in columns.T
        ], axis=-1)
        return flat.reshape(self.shape + values.shape[1:])

    def gather(self, grid: np.ndarray) -> np.ndarray:
        """Read a (cols, rows, d) grid at the particles, shape (particles, d)."""
        flat = grid.reshape(self.shape[0] * self.shape[1], -1)
        return (flat[self.index] * self.weight[..., None]).sum(axis=0)


def spread_over_cells(values: np.ndarray) -> np.ndarray:
    """Deposit each cell's value (cols, rows) as if spread evenly over its cell.

    This is the cloud-in-cell deposit of a uniform square: per axis 3/4 stays on
    the cell's centre and 1/8 goes to each neighbour. Ions are deposited this way,
    so electrons evenly filling their cells cancel them, rather than only electrons
    stacked on the centre. Parts beyond the grid edge stay on the edge cell, as in
    CloudInCell, so the sum is conserved.
    """
    for axis in (0, 1):
        a = np.moveaxis(values, axis, 0)
        out = 0.75 * a
        out[1:] += a[:-1] / 8
        out[:-1] += a[1:] / 8
        out[0] += a[0] / 8
        out[-1] += a[-1] / 8
        values = np.moveaxis(out, 0, axis)
    return values


def ion_charge(materials, cell_charge: float) -> np.ndarray:
    """Charge of the fixed ions on the cell centres (cols, rows): spread over their cells."""
    return spread_over_cells(materials.conductive * cell_charge)


def poynting(e_grid: np.ndarray, bz: np.ndarray, settings: Settings) -> np.ndarray:
    """Poynting vector S = E x B / mu0 per cell, shape (cols, rows, 2): energy flow.

    With E in the plane and B = (0, 0, B_z), E x B = (E_y B_z, -E_x B_z) is back
    in the plane. 1/mu0 = c^2 / (4 pi k) in our units.
    """
    inv_mu0 = settings.speed_of_light**2 / (4 * np.pi * settings.coulomb_k)
    return inv_mu0 * np.stack([e_grid[..., 1] * bz, -e_grid[..., 0] * bz], axis=-1)
