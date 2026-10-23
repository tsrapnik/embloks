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


def coulomb_potential(
    points: np.ndarray,
    sources: np.ndarray,
    charges: np.ndarray,
    softening: float,
    k: float = 1.0,
) -> np.ndarray:
    """Electric potential at `points` (M,) from softened point charges; zero far away.

    V = k q / sqrt(r^2 + eps^2), the potential whose gradient is coulomb_field.
    """
    potential = np.zeros(len(points))
    if len(points) == 0 or len(sources) == 0:
        return potential
    rows_per_block = max(1, _BLOCK_SIZE // len(sources))
    eps2 = softening**2
    for start in range(0, len(points), rows_per_block):
        block = points[start : start + rows_per_block]
        dx = block[:, 0, None] - sources[None, :, 0]
        dy = block[:, 1, None] - sources[None, :, 1]
        potential[start : start + rows_per_block] = k * (charges / np.sqrt(dx * dx + dy * dy + eps2)).sum(axis=1)
    return potential


def biot_savart(
    points: np.ndarray,
    sources: np.ndarray,
    currents: np.ndarray,
    softening: float,
    k: float = 1.0,
    c: float = 1.0,
) -> np.ndarray:
    """B_z at `points` (M,) from moving charges at `sources` with q*v = `currents` (N, 2).

    Low-speed Biot-Savart with the same softening as Coulomb:
    B_z = (k / c^2) * sum q (v x r)_z / (r^2 + eps^2)^1.5, with r = point - source.
    Screen coordinates have y pointing down, so positive B_z points INTO the screen.
    """
    bz = np.zeros(len(points))
    if len(points) == 0 or len(sources) == 0:
        return bz
    rows_per_block = max(1, _BLOCK_SIZE // len(sources))
    eps2 = softening**2
    for start in range(0, len(points), rows_per_block):
        block = points[start : start + rows_per_block]
        dx = block[:, 0, None] - sources[None, :, 0]
        dy = block[:, 1, None] - sources[None, :, 1]
        r2 = dx * dx + dy * dy + eps2
        cross = currents[None, :, 0] * dy - currents[None, :, 1] * dx
        bz[start : start + rows_per_block] = (cross / (r2 * np.sqrt(r2))).sum(axis=1)
    return k / c**2 * bz


class GridKernel:
    """Field and potential at every cell centre from charges on cell centres, via FFT.

    Equivalent to coulomb_field / coulomb_potential over all cells, but O(n log n).
    The kernel transforms are computed once per grid size and softening.
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
        r2 = dx**2 + dy**2 + softening**2
        inv_r3 = k / r2**1.5
        self._kx = fft.rfft2(dx * inv_r3, s=self.shape)
        self._ky = fft.rfft2(dy * inv_r3, s=self.shape)
        self._kv = fft.rfft2(k / np.sqrt(r2), s=self.shape)
        # Offset 0 sits at kernel index n - 1, so the centred outputs start there.
        self._keep = (slice(cols - 1, 2 * cols - 1), slice(rows - 1, 2 * rows - 1))

    def _apply(self, q: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        return fft.irfft2(q * kernel, s=self.shape)[self._keep]

    def field(self, charge_grid: np.ndarray) -> np.ndarray:
        """`charge_grid` (cols, rows) -> field (cols, rows, 2)."""
        q = fft.rfft2(charge_grid, s=self.shape)
        return np.stack([self._apply(q, self._kx), self._apply(q, self._ky)], axis=-1)

    def potential(self, charge_grid: np.ndarray) -> np.ndarray:
        """`charge_grid` (cols, rows) -> potential (cols, rows)."""
        return self._apply(fft.rfft2(charge_grid, s=self.shape), self._kv)

    def curl(self, current_grid: np.ndarray) -> np.ndarray:
        """Current density q*v (cols, rows, 2) -> sum J x r / r^3, z-component (cols, rows).

        Divide by c^2 for B_z (biot_savart without the 1/c^2).
        """
        jx = fft.rfft2(current_grid[..., 0], s=self.shape)
        jy = fft.rfft2(current_grid[..., 1], s=self.shape)
        return self._apply(jx, self._ky) - self._apply(jy, self._kx)


@lru_cache(maxsize=8)
def grid_kernel(cols: int, rows: int, softening: float, k: float = 1.0) -> GridKernel:
    return GridKernel(cols, rows, softening, k)


def _cic_weights(positions: np.ndarray, cols: int, rows: int):
    """The 4 cell centres around each position and their bilinear weights.

    Yields (flat_index, weight) per corner. Corners outside the grid are clamped
    onto the edge cells, so weights per position still sum to 1.
    """
    u = positions[:, 0] - 0.5
    v = positions[:, 1] - 0.5
    c0, r0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    fu, fv = u - c0, v - r0
    for dc, wc in ((0, 1 - fu), (1, fu)):
        c = np.clip(c0 + dc, 0, cols - 1)
        for dr, wr in ((0, 1 - fv), (1, fv)):
            r = np.clip(r0 + dr, 0, rows - 1)
            yield c * rows + r, wc * wr


def deposit(positions: np.ndarray, charges: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """Spread point charges over the 4 nearest cell centres (cloud-in-cell).

    Charge is conserved: weight that would land outside the grid stays on the edge cells.
    """
    grid = np.zeros(cols * rows)
    for index, weight in _cic_weights(positions, cols, rows):
        grid += np.bincount(index, charges * weight, minlength=cols * rows)
    return grid.reshape(cols, rows)


def interpolate(grid: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Read a (cols, rows, d) grid at positions with the same weights `deposit` uses.

    Using identical weights both ways makes the grid force on a particle from
    its own charge cancel exactly (the Coulomb kernel is antisymmetric).
    """
    cols, rows = grid.shape[:2]
    flat = grid.reshape(cols * rows, -1)
    out = np.zeros((len(positions), flat.shape[1]))
    for index, weight in _cic_weights(positions, cols, rows):
        out += flat[index] * weight[:, None]
    return out


class ElectricField:
    """Electric field of a World: ions, electrons and fixed test charges.

    `at` is exact (direct sums). `grid` is the fast version, used for display and
    for forces: all charges go through FFT convolution on the grid. Ions and test
    charges sit on cell centres, so their part is exact and cached until they
    change. Electrons are deposited onto cell centres, which blurs their field
    within about a cell (on the scale of the softening anyway).
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._static_key = None
        self._static_grid: np.ndarray | None = None
        self._static_potential: np.ndarray | None = None

    def _kernel(self, world, softening: float) -> GridKernel:
        m = world.materials
        return grid_kernel(m.cols, m.rows, softening, self.settings.coulomb_k)

    def _update_static(self, world) -> None:
        """(Re)compute the field and potential of ions and test charges if they changed."""
        key = (id(world), world.materials.version, world.fixed_version)
        if key == self._static_key:
            return
        s, m = self.settings, world.materials
        ion_kernel = self._kernel(world, s.conductor_softening)
        ions = m.conductive * s.cell_charge
        fixed_kernel = self._kernel(world, s.test_charge_softening)
        fixed = deposit(world.fixed_pos, world.fixed_charge, m.cols, m.rows)
        self._static_grid = ion_kernel.field(ions) + fixed_kernel.field(fixed)
        self._static_potential = ion_kernel.potential(ions) + fixed_kernel.potential(fixed)
        self._static_key = key

    def static_grid(self, world) -> np.ndarray:
        """Field of ions and test charges at every cell centre, shape (cols, rows, 2)."""
        self._update_static(world)
        return self._static_grid

    def _electron_charge(self, world) -> np.ndarray:
        m, p = world.materials, world.particles
        return deposit(p.pos, p.charge, m.cols, m.rows)

    def electron_grid(self, world) -> np.ndarray:
        """Approximate electron field at every cell centre, shape (cols, rows, 2)."""
        kernel = self._kernel(world, self.settings.conductor_softening)
        return kernel.field(self._electron_charge(world))

    def grid(self, world) -> np.ndarray:
        """Total field at every cell centre, shape (cols, rows, 2)."""
        return self.static_grid(world) + self.electron_grid(world)

    def potential_grid(self, world) -> np.ndarray:
        """Total potential at every cell centre, shape (cols, rows); zero far away."""
        self._update_static(world)
        kernel = self._kernel(world, self.settings.conductor_softening)
        return self._static_potential + kernel.potential(self._electron_charge(world))

    def at(self, world, points: np.ndarray) -> np.ndarray:
        """Exact total field at arbitrary points (M, 2), all sources summed directly."""
        return self._sum_sources(world, points, coulomb_field)

    def potential_at(self, world, points: np.ndarray) -> np.ndarray:
        """Exact electric potential at arbitrary points (M,), zero far from all charges."""
        return self._sum_sources(world, points, coulomb_potential)

    def _sum_sources(self, world, points: np.ndarray, kernel) -> np.ndarray:
        s = self.settings
        ions = np.argwhere(world.materials.conductive) + 0.5
        ion_charges = np.full(len(ions), s.cell_charge)
        p = world.particles
        return (
            kernel(points, ions, ion_charges, s.conductor_softening, s.coulomb_k)
            + kernel(points, p.pos, p.charge, s.conductor_softening, s.coulomb_k)
            + kernel(points, world.fixed_pos, world.fixed_charge, s.test_charge_softening, s.coulomb_k)
        )


class MagneticField:
    """B_z of a World from a current density grid (electrons plus battery currents)."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def grid(self, world, current: np.ndarray) -> np.ndarray:
        """B_z at every cell centre from current density `current` (cols, rows, 2)."""
        s, m = self.settings, world.materials
        kernel = grid_kernel(m.cols, m.rows, s.conductor_softening, s.coulomb_k)
        return kernel.curl(current) / s.speed_of_light**2

    def electron_current(self, world) -> np.ndarray:
        """Electron current q*v deposited onto cell centres (cloud-in-cell), (cols, rows, 2)."""
        m, p = world.materials, world.particles
        jx = deposit(p.pos, p.charge * p.vel[:, 0], m.cols, m.rows)
        jy = deposit(p.pos, p.charge * p.vel[:, 1], m.cols, m.rows)
        return np.stack([jx, jy], axis=-1)
