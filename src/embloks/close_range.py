"""Close-range forces between electrons, and between electrons and ions.

The grid fields (maxwell.py) smooth every charge over about a cell, so two
electrons barely repel within a cell and can sit on top of each other. Within
electron_cutoff this adds what the grid leaves out: electrons repel as compact
line charges (smeared over only electron_softening), and ions attract them as
charge spread evenly over their cells. Electrons evenly filling a conductor then
feel no net close-range force, as in a real metal; crowding or gaps get pushed
back. This is the particle-particle part of a P3M scheme, with the Maxwell grid as
its particle-mesh part.

Everything is tabulated once: the force the grid really applies between two
charges (measured with the Maxwell code itself, averaged over where the charges
sit in their cells), the correction for a point charge, and that correction
integrated over a uniform square (an ion). The force loop itself is compiled
with Numba.
"""

from functools import lru_cache

import numpy as np
from numba import njit
from scipy import ndimage

from embloks.config import Settings
from embloks.fields import CloudInCell
from embloks.maxwell import Maxwell

# Samples (uniform in r^2, cutoff included) of the radial point-charge tables.
_PAIR_SAMPLES = 4096
# Samples per cell (per axis) of the table of a uniform square.
_SQUARE_SAMPLES = 32
# Measuring the grid's force: source positions per axis within a cell, directions,
# and radial spacing (cells).
_SOURCE_SAMPLES = 8
_DIRECTIONS = 16
_RADIAL_STEP = 0.01
# Below this r^2 / softening^2 the compact kernel uses its Taylor series.
_SERIES_BELOW = 1e-4


def compact_g(r2: np.ndarray, softening: float) -> np.ndarray:
    """g with field g d at offset d from a unit line charge smeared as a Gaussian.

    In 2D (div E = 4 pi rho) a Gaussian of width eps has the field
    2 d / r^2 (1 - exp(-r^2 / eps^2)): plain 2 / r far away, zero at r = 0.
    """
    x = np.asarray(r2) / softening**2
    small = x < _SERIES_BELOW
    x_safe = np.where(small, 1.0, x)
    closed = -2 * np.expm1(-x_safe) / x_safe
    series = 2 * (1 - x / 2 + x * x / 6)
    return np.where(small, series, closed) / softening**2


@lru_cache(maxsize=4)
def grid_force(passes: int, reach: float) -> tuple[np.ndarray, np.ndarray]:
    """Radii (0 .. reach) and the radial force between two unit charges through the grid.

    Measured with the Maxwell code: one charge deposited, smoothed, its static
    field solved and read back at the other charge. Averaged over where the
    source sits in its cell and over directions, since the grid's force depends
    a little on both.
    """
    n = 2 * int(np.ceil(reach)) + 24
    maxwell = Maxwell(n, n, Settings(smoothing_passes=passes, coulomb_k=1.0))
    radii = np.linspace(0, reach, int(round(reach / _RADIAL_STEP)) + 1)
    angles = 2 * np.pi * (np.arange(_DIRECTIONS) + 0.5) / _DIRECTIONS
    u = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    total = np.zeros(len(radii))
    for i in range(_SOURCE_SAMPLES):
        for j in range(_SOURCE_SAMPLES):
            source = n // 2 + (np.array([i, j]) + 0.5) / _SOURCE_SAMPLES
            maxwell.ex[:] = 0
            maxwell.ey[:] = 0
            maxwell.enforce_gauss(maxwell.smoothed(CloudInCell(source[None], n, n).deposit(np.ones(1))))
            points = source + radii[:, None, None] * u
            field = maxwell.gather(points.reshape(-1, 2))[:, :2].reshape(len(radii), _DIRECTIONS, 2)
            total += (field * u).sum(axis=-1).mean(axis=1)
    return radii, total / _SOURCE_SAMPLES**2


@lru_cache(maxsize=8)
def radial_tables(softening: float, passes: int, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """The close-range correction of a unit charge: g (field g d) and its potential,
    sampled at r^2 = cutoff^2 i / _PAIR_SAMPLES, plus a zero sample past the cutoff.

    The correction is the compact force minus the grid's. Its potential is the
    integral of that force inwards from the cutoff, so it is zero there and the
    field is exactly its gradient.
    """
    r, f_grid = grid_force(passes, cutoff)
    f_near = r * compact_g(r * r, softening) - f_grid
    steps = 0.5 * (f_near[1:] + f_near[:-1]) * np.diff(r)
    potential = np.append(np.cumsum(steps[::-1])[::-1], 0.0)
    r2 = np.linspace(0, cutoff**2, _PAIR_SAMPLES + 1)
    rs = np.sqrt(r2)
    # The grid's force grows linearly from r = 0, so f / r is finite there.
    grid_g = np.interp(rs, r, f_grid) / np.where(rs > 0, rs, 1.0)
    grid_g[0] = f_grid[1] / r[1]
    g = compact_g(r2, softening) - grid_g
    return np.append(g, 0.0), np.append(np.interp(rs, r, potential), 0.0)


def _radial(table: np.ndarray, r2: np.ndarray, cutoff: float) -> np.ndarray:
    """Linear interpolation in a radial table at squared distances r2; zero beyond the cutoff."""
    x = np.minimum(r2 * (_PAIR_SAMPLES / cutoff**2), _PAIR_SAMPLES)
    i = x.astype(np.int64)
    return np.where(r2 < cutoff**2, table[i] + (table[i + 1] - table[i]) * (x - i), 0.0)


def point_correction(d: np.ndarray, softening: float, passes: int, cutoff: float) -> np.ndarray:
    """Close-range potential and field (..., 3) of a unit charge at offsets d (..., 2)."""
    g, potential = radial_tables(softening, passes, cutoff)
    r2 = d[..., 0] ** 2 + d[..., 1] ** 2
    return np.concatenate(
        [_radial(potential, r2, cutoff)[..., None], _radial(g, r2, cutoff)[..., None] * d], axis=-1
    )


@lru_cache(maxsize=8)
def square_table(softening: float, passes: int, cutoff: float) -> tuple[np.ndarray, float]:
    """point_correction of a unit charge spread evenly over a cell, tabulated.

    Returns the table (n, n, 3) over offsets from the cell centre in [-half, half]^2
    (spacing 1 / _SQUARE_SAMPLES), and `half`. The square is integrated with the
    midpoint rule: a sliding sum of the point-charge values along each axis.
    """
    m = _SQUARE_SAMPLES
    h = 1 / m
    half = float(np.ceil(cutoff) + 1)
    # Point values at offsets z_k = -half - 0.5 + (k + 0.5) h; table entry i
    # (offset -half + i h) sums z_i .. z_(i + m - 1) along each axis.
    z = -half - 0.5 + (np.arange(int(round((2 * half + 1) * m))) + 0.5) * h
    zx, zy = np.meshgrid(z, z, indexing="ij")
    point = point_correction(np.stack([zx, zy], axis=-1), softening, passes, cutoff)
    total = np.pad(point.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0), (0, 0)))
    n = len(z) - m + 1
    table = total[m:m + n, m:m + n] - total[:n, m:m + n] - total[m:m + n, :n] + total[:n, :n]
    return table * h**2, half


@lru_cache(maxsize=8)
def _ion_stencil(softening: float, passes: int, cutoff: float) -> np.ndarray:
    """Close-range potential of a uniform unit square, averaged over the cell at each
    offset (dc, dr) from it, shape (2 reach + 1,) * 2 (trapezoid rule on the table)."""
    square, half = square_table(softening, passes, cutoff)
    m = _SQUARE_SAMPLES
    reach = int(np.ceil(cutoff))
    weights = np.ones(m + 1)
    weights[[0, -1]] = 0.5
    weights /= m
    stencil = np.zeros((2 * reach + 1,) * 2)
    for dc in range(-reach, reach + 1):
        for dr in range(-reach, reach + 1):
            a, b = int(round((dc - 0.5 + half) * m)), int(round((dr - 0.5 + half) * m))
            stencil[dc + reach, dr + reach] = weights @ square[a : a + m + 1, b : b + m + 1, 0] @ weights
    return stencil


class CloseRange:
    """Close-range forces on electrons and their potential, on top of the grid fields."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._key = None

    @property
    def enabled(self) -> bool:
        return self.settings.electron_cutoff > 0

    def _params(self) -> tuple[float, int, float]:
        s = self.settings
        return s.electron_softening, s.smoothing_passes, s.electron_cutoff

    def force(self, world) -> np.ndarray:
        """Close-range force on each electron (N, 2). Opposite and equal between
        electrons, so momentum is conserved."""
        s, p = self.settings, world.particles
        g, _ = radial_tables(*self._params())
        half = square_table(*self._params())[1]
        return _forces(
            p.pos, p.charge, world.materials.density, _square_field(*self._params()), half,
            g, s.electron_cutoff, s.coulomb_k, s.coulomb_k * s.cell_charge,
        )

    def _ion_potential(self, world) -> np.ndarray:
        """Close-range potential of all ions, averaged over each cell (cols, rows)."""
        s, m = self.settings, world.materials
        key = (id(world), m.version, *self._params(), s.coulomb_k, s.cell_charge)
        if key != self._key:
            stencil = _ion_stencil(*self._params())
            self._ions = s.coulomb_k * s.cell_charge * ndimage.correlate(
                m.density, stencil, mode="constant"
            )
            self._key = key
        return self._ions

    def potential(self, world, where: np.ndarray | None = None) -> np.ndarray:
        """Close-range part of the potential, averaged over each cell (cols, rows).

        The potential whose gradient `force` adds. Batteries need it: they hold the
        potential difference between their terminals, and electrons crowded at the
        - terminal push harder than the grid potential alone shows. Averaged over
        the cell rather than read at its centre, which is sharp: it would jump with
        every electron passing close by.

        `where` (cols, rows) limits the work to the cells that will be read; the
        result is zero elsewhere.
        """
        potential = self._ion_potential(world) + self.charges_potential(
            world.materials, world.particles.pos, world.particles.charge, where)
        return potential if where is None else np.where(where, potential, 0.0)

    def charges_potential(self, materials, pos: np.ndarray, charge: np.ndarray,
                          where: np.ndarray | None = None) -> np.ndarray:
        """Close-range potential of point charges like electrons, averaged over each cell
        (cols, rows). With `where`, only exact on those cells."""
        s, m = self.settings, materials
        square, half = square_table(*self._params())
        reach = int(np.ceil(s.electron_cutoff))
        if where is not None:
            # Only charges within reach of a requested cell contribute to it.
            near = ndimage.binary_dilation(where, np.ones((2 * reach + 1,) * 2, dtype=bool))
            cell = np.floor(pos).astype(np.int64)
            inside = near[cell[:, 0], cell[:, 1]]
            pos, charge = pos[inside], charge[inside]
        steps = np.arange(-reach, reach + 1)
        offsets = np.stack(np.meshgrid(steps, steps, indexing="ij"), axis=-1).reshape(-1, 1, 2)
        cells = np.floor(pos).astype(np.int64) + offsets
        c, r = cells[..., 0], cells[..., 1]
        hit = (c >= 0) & (c < m.cols) & (r >= 0) & (r < m.rows)
        # A charge's potential averaged over a cell is that of the cell, spread evenly,
        # at the charge: the square table.
        d = (pos - (cells + 0.5))[hit]
        values = _bilinear(square[..., 0], (d + half) * _SQUARE_SAMPLES)
        total = np.bincount(
            (c * m.rows + r)[hit], values * np.broadcast_to(charge, c.shape)[hit],
            minlength=m.cols * m.rows,
        )
        return s.coulomb_k * total.reshape(m.cols, m.rows)


@lru_cache(maxsize=8)
def _square_field(softening: float, passes: int, cutoff: float) -> np.ndarray:
    """The square table's field components as one contiguous array, for the kernel."""
    return np.ascontiguousarray(square_table(softening, passes, cutoff)[0][..., 1:])


def _bilinear(table: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Bilinear interpolation in a 2D table at fractional indices (N, 2)."""
    i = np.clip(np.floor(index).astype(np.int64), 0, np.array(table.shape) - 2)
    f = index - i
    fx, fy = f[:, 0], f[:, 1]
    ix, iy = i[:, 0], i[:, 1]
    return (
        table[ix, iy] * (1 - fx) * (1 - fy)
        + table[ix + 1, iy] * fx * (1 - fy)
        + table[ix, iy + 1] * (1 - fx) * fy
        + table[ix + 1, iy + 1] * fx * fy
    )


@njit(cache=True)
def _forces(pos, charge, ions, square, half, g, cutoff, k, k_ion):  # pragma: no cover
    """Close-range forces (N, 2): electron pairs within the cutoff and the ions nearby.

    Electrons are sorted by cell, so each one only checks the cells within reach.
    Every pair is handled once, from the electron in the lower-numbered cell.
    """
    n = len(pos)
    cols, rows = ions.shape
    reach = int(np.ceil(cutoff))
    cutoff2 = cutoff * cutoff
    scale = (len(g) - 2) / cutoff2
    m = square.shape[0]
    samples = (m - 1) / (2 * half)
    # Counting sort of the electrons by cell.
    cell = np.empty(n, np.int64)
    start = np.zeros(cols * rows + 1, np.int64)
    for i in range(n):
        c = min(max(int(np.floor(pos[i, 0])), 0), cols - 1)
        r = min(max(int(np.floor(pos[i, 1])), 0), rows - 1)
        cell[i] = c * rows + r
        start[cell[i] + 1] += 1
    for i in range(cols * rows):
        start[i + 1] += start[i]
    order = np.empty(n, np.int64)
    fill = start[:-1].copy()
    for i in range(n):
        order[fill[cell[i]]] = i
        fill[cell[i]] += 1

    force = np.zeros((n, 2))
    for i in range(n):
        x, y = pos[i, 0], pos[i, 1]
        ci, ri = cell[i] // rows, cell[i] % rows
        for c in range(max(ci - reach, 0), min(ci + reach + 1, cols)):
            for r in range(max(ri - reach, 0), min(ri + reach + 1, rows)):
                if ions[c, r] > 0:
                    # The ion of cell (c, r): the square table at the offset from its centre.
                    u = (x - c - 0.5 + half) * samples
                    v = (y - r - 0.5 + half) * samples
                    a = min(max(int(u), 0), m - 2)
                    b = min(max(int(v), 0), m - 2)
                    fu, fv = u - a, v - b
                    for axis in range(2):
                        value = (square[a, b, axis] * (1 - fu) * (1 - fv)
                                 + square[a + 1, b, axis] * fu * (1 - fv)
                                 + square[a, b + 1, axis] * (1 - fu) * fv
                                 + square[a + 1, b + 1, axis] * fu * fv)
                        force[i, axis] += k_ion * ions[c, r] * charge[i] * value
                other = c * rows + r
                if other < cell[i]:
                    continue
                for slot in range(start[other], start[other + 1]):
                    j = order[slot]
                    if other == cell[i] and j <= i:
                        continue
                    dx, dy = x - pos[j, 0], y - pos[j, 1]
                    d2 = dx * dx + dy * dy
                    if d2 >= cutoff2:
                        continue
                    t = d2 * scale
                    s = int(t)
                    f = k * charge[i] * charge[j] * (g[s] + (g[s + 1] - g[s]) * (t - s))
                    force[i, 0] += f * dx
                    force[i, 1] += f * dy
                    force[j, 0] -= f * dx
                    force[j, 1] -= f * dy
    return force
