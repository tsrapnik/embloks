from functools import lru_cache

import numpy as np
from scipy import fft

from embloks.config import Settings

# Bound on (points x sources) per vectorised block, to keep memory use flat.
_BLOCK_SIZE = 1_000_000


def coulomb_field(
    points: np.ndarray,
    sources: np.ndarray,
    charges: np.ndarray,
    softening: float,
    k: float = 1.0,
) -> np.ndarray:
    """Electric field at `points` (M, 2) from point charges at `sources` (N, 2).

    Uses Plummer softening: E = k q r / (r^2 + eps^2)^1.5. This smears each
    charge over a radius ~eps, avoids the 1/r^2 blow-up, and makes a charge's
    field at its own position exactly zero (so particles never push themselves).
    """
    field = np.zeros((len(points), 2))
    if len(points) == 0 or len(sources) == 0:
        return field
    rows_per_block = max(1, _BLOCK_SIZE // len(sources))
    eps2 = softening**2
    for start in range(0, len(points), rows_per_block):
        block = points[start : start + rows_per_block]
        dx = block[:, 0, None] - sources[None, :, 0]
        dy = block[:, 1, None] - sources[None, :, 1]
        r2 = dx * dx + dy * dy + eps2
        weight = charges / (r2 * np.sqrt(r2))
        # sum_n w (p - s_n) = p * sum_n w - w @ s, letting BLAS do the heavy part.
        total = weight.sum(axis=1)
        field[start : start + rows_per_block] = k * (block * total[:, None] - weight @ sources)
    return field


class GridKernel:
    """Field at every cell centre from charges on cell centres, via FFT convolution.

    Equivalent to coulomb_field over all cells, but O(n log n). The kernel's
    transform is computed once per grid size and softening.
    """

    def __init__(self, cols: int, rows: int, softening: float, k: float = 1.0):
        self.cols, self.rows = cols, rows
        # Circular convolution is exact for the outputs we keep when the transform
        # is at least as long as the kernel (2n - 1): nothing wraps into them.
        self.shape = (
            fft.next_fast_len(2 * cols - 1, real=True),
            fft.next_fast_len(2 * rows - 1, real=True),
        )
        dx, dy = np.meshgrid(
            np.arange(-(cols - 1), cols), np.arange(-(rows - 1), rows), indexing="ij"
        )
        inv_r3 = k / (dx**2 + dy**2 + softening**2) ** 1.5
        self._kx = fft.rfft2(dx * inv_r3, s=self.shape)
        self._ky = fft.rfft2(dy * inv_r3, s=self.shape)

    def field(self, charge_grid: np.ndarray) -> np.ndarray:
        """`charge_grid` (cols, rows) -> field (cols, rows, 2)."""
        q = fft.rfft2(charge_grid, s=self.shape)
        # Offset 0 sits at kernel index n - 1, so the centred outputs start there.
        keep = (slice(self.cols - 1, 2 * self.cols - 1), slice(self.rows - 1, 2 * self.rows - 1))
        ex = fft.irfft2(q * self._kx, s=self.shape)[keep]
        ey = fft.irfft2(q * self._ky, s=self.shape)[keep]
        return np.stack([ex, ey], axis=-1)


@lru_cache(maxsize=8)
def grid_kernel(cols: int, rows: int, softening: float, k: float = 1.0) -> GridKernel:
    return GridKernel(cols, rows, softening, k)


def deposit(positions: np.ndarray, charges: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """Spread point charges over the 4 nearest cell centres (cloud-in-cell).

    Charge is conserved: weight that would land outside the grid stays on the edge cells.
    """
    grid = np.zeros((cols, rows))
    u = positions[:, 0] - 0.5
    v = positions[:, 1] - 0.5
    c0, r0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    fu, fv = u - c0, v - r0
    for dc, wc in ((0, 1 - fu), (1, fu)):
        for dr, wr in ((0, 1 - fv), (1, fv)):
            c = np.clip(c0 + dc, 0, cols - 1)
            r = np.clip(r0 + dr, 0, rows - 1)
            np.add.at(grid, (c, r), charges * wc * wr)
    return grid


class ElectricField:
    """Electric field of a World: ions, electrons and fixed test charges.

    `at` is exact (direct sums). `grid` is the fast version for display: all
    charges go through FFT convolution on the grid (the ion part is cached until
    the next paint). Electrons are deposited onto cell centres for this, which
    blurs their field within about a cell.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._ion_key = None
        self._ion_grid: np.ndarray | None = None

    def _kernel(self, world, softening: float) -> GridKernel:
        m = world.materials
        return grid_kernel(m.cols, m.rows, softening, self.settings.coulomb_k)

    def ion_grid(self, world) -> np.ndarray:
        """Ion field at every cell centre, shape (cols, rows, 2)."""
        key = (id(world.materials), world.materials.version)
        if key != self._ion_key:
            s = self.settings
            charge = world.materials.conductive * s.elementary_charge
            self._ion_grid = self._kernel(world, s.conductor_softening).field(charge)
            self._ion_key = key
        return self._ion_grid

    def electron_grid(self, world) -> np.ndarray:
        """Approximate electron field at every cell centre, shape (cols, rows, 2)."""
        m, p = world.materials, world.particles
        charge = deposit(p.pos, p.charge, m.cols, m.rows)
        return self._kernel(world, self.settings.conductor_softening).field(charge)

    def fixed_grid(self, world) -> np.ndarray:
        """Test charge field at every cell centre (exact: test charges sit on centres)."""
        m = world.materials
        charge = deposit(world.fixed_pos, world.fixed_charge, m.cols, m.rows)
        return self._kernel(world, self.settings.test_charge_softening).field(charge)

    def grid(self, world) -> np.ndarray:
        """Total field at every cell centre (fast, display quality), shape (cols, rows, 2)."""
        return self.ion_grid(world) + self.electron_grid(world) + self.fixed_grid(world)

    def at(self, world, points: np.ndarray) -> np.ndarray:
        """Exact total field at arbitrary points (M, 2), all sources summed directly."""
        s = self.settings
        ions = np.argwhere(world.materials.conductive) + 0.5
        ion_charges = np.full(len(ions), s.elementary_charge)
        p = world.particles
        return (
            coulomb_field(points, ions, ion_charges, s.conductor_softening, s.coulomb_k)
            + coulomb_field(points, p.pos, p.charge, s.conductor_softening, s.coulomb_k)
            + self._fixed(world, points)
        )

    def _fixed(self, world, points: np.ndarray) -> np.ndarray:
        s = self.settings
        return coulomb_field(
            points, world.fixed_pos, world.fixed_charge, s.test_charge_softening, s.coulomb_k
        )
