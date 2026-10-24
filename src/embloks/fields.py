from functools import lru_cache

import numpy as np
from scipy import fft, ndimage, special
from scipy.spatial import cKDTree

from embloks.config import Settings

# Bound on (points x sources) per vectorised block, to keep memory use flat.
_BLOCK_SIZE = 1_000_000

# Below this r / softening, the softened kernels use their Taylor series: the
# closed forms divide by r and, for the field, lose digits to cancellation.
_SERIES_BELOW = 1e-2


def softened_inv_r(r2: np.ndarray, softening: float) -> np.ndarray:
    """erf(r / eps) / r at squared distances r2: the potential of a unit charge
    smeared as a Gaussian, density exp(-r^2 / eps^2) / (pi^1.5 eps^3).

    Plain 1 / r far away (within 0.05% beyond 2.5 eps), finite 2 / (sqrt(pi) eps) at r = 0.
    """
    x = np.sqrt(r2) / softening
    small = x < _SERIES_BELOW
    x_safe = np.where(small, 1.0, x)
    closed = special.erf(x_safe) / x_safe
    series = 2 / np.sqrt(np.pi) * (1 - x * x / 3)
    return np.where(small, series, closed) / softening


def softened_inv_r3(r2: np.ndarray, softening: float) -> np.ndarray:
    """g with field k q d g at offset d from a Gaussian-smeared unit charge: -grad of
    softened_inv_r is d (erf(x) - 2 x exp(-x^2) / sqrt(pi)) / r^3, x = r / eps.

    Plain 1 / r^3 far away, and zero field at r = 0: a charge never pushes itself.
    """
    x2 = r2 / softening**2
    x = np.sqrt(x2)
    small = x < _SERIES_BELOW
    x_safe = np.where(small, 1.0, x)
    closed = (special.erf(x_safe) - 2 / np.sqrt(np.pi) * x_safe * np.exp(-x_safe * x_safe)) / x_safe**3
    series = 4 / (3 * np.sqrt(np.pi)) * (1 - 0.6 * x2)
    return np.where(small, series, closed) / softening**3


def coulomb_field(
    points: np.ndarray,
    sources: np.ndarray,
    charges: np.ndarray,
    softening: float,
    k: float = 1.0,
) -> np.ndarray:
    """Electric field at `points` (M, 2) from point charges at `sources` (N, 2).

    Each charge is smeared as a Gaussian of size ~eps (see softened_inv_r3): this
    avoids the 1/r^2 blow-up, and makes a charge's field at its own position
    exactly zero (so particles never push themselves).
    """
    field = np.zeros((len(points), 2))
    if len(points) == 0 or len(sources) == 0:
        return field
    rows_per_block = max(1, _BLOCK_SIZE // len(sources))
    for start in range(0, len(points), rows_per_block):
        block = points[start : start + rows_per_block]
        dx = block[:, 0, None] - sources[None, :, 0]
        dy = block[:, 1, None] - sources[None, :, 1]
        weight = charges * softened_inv_r3(dx * dx + dy * dy, softening)
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

    V = k q erf(r / eps) / r, the potential whose gradient is coulomb_field.
    """
    potential = np.zeros(len(points))
    if len(points) == 0 or len(sources) == 0:
        return potential
    rows_per_block = max(1, _BLOCK_SIZE // len(sources))
    for start in range(0, len(points), rows_per_block):
        block = points[start : start + rows_per_block]
        dx = block[:, 0, None] - sources[None, :, 0]
        dy = block[:, 1, None] - sources[None, :, 1]
        potential[start : start + rows_per_block] = k * (
            charges * softened_inv_r(dx * dx + dy * dy, softening)
        ).sum(axis=1)
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
    B_z = (k / c^2) * sum q (v x r)_z g(r), with r = point - source and g as in coulomb_field.
    Screen coordinates have y pointing down, so positive B_z points INTO the screen.
    """
    bz = np.zeros(len(points))
    if len(points) == 0 or len(sources) == 0:
        return bz
    rows_per_block = max(1, _BLOCK_SIZE // len(sources))
    for start in range(0, len(points), rows_per_block):
        block = points[start : start + rows_per_block]
        dx = block[:, 0, None] - sources[None, :, 0]
        dy = block[:, 1, None] - sources[None, :, 1]
        cross = currents[None, :, 0] * dy - currents[None, :, 1] * dx
        bz[start : start + rows_per_block] = (cross * softened_inv_r3(dx * dx + dy * dy, softening)).sum(axis=1)
    return k / c**2 * bz


def _close_potential(r2: np.ndarray, softening: float, grid_softening: float, cutoff: float) -> np.ndarray:
    """Close-range correction potential of a unit charge at squared distances r2.

    The Ewald split: the grid fields apply erf(r / grid_softening) / r, this adds
    the rest of erf(r / softening) / r, i.e. erfc(r / grid_softening) / r -
    erfc(r / softening) / r (just the erfc term for point charges). It decays like a
    Gaussian, so it is dropped beyond the cutoff.
    """
    near = softened_inv_r(r2, softening) - softened_inv_r(r2, grid_softening)
    return np.where(r2 < cutoff**2, near, 0.0)


def _close_field(d: np.ndarray, softening: float, grid_softening: float, cutoff: float) -> np.ndarray:
    """Close-range correction field (..., 2) of a unit charge at offsets d: -grad _close_potential."""
    r2 = d[..., 0] ** 2 + d[..., 1] ** 2
    near = softened_inv_r3(r2, softening) - softened_inv_r3(r2, grid_softening)
    return np.where(r2 < cutoff**2, near, 0.0)[..., None] * d


# Samples (uniform in r^2, cutoff included) of the tabulated pair kernel: erf and
# exp per pair cost several times more than reading this back (error ~1e-5 of the peak).
_PAIR_SAMPLES = 4096


@lru_cache(maxsize=8)
def _pair_table(softening: float, grid_softening: float, cutoff: float) -> np.ndarray:
    """g with _close_field = g(r^2) d, sampled at r^2 = cutoff^2 i / _PAIR_SAMPLES.

    One extra zero sample after the cutoff, so interpolating at it stays in bounds.
    """
    r2 = np.linspace(0, cutoff**2, _PAIR_SAMPLES + 1)
    g = softened_inv_r3(r2, softening) - softened_inv_r3(r2, grid_softening)
    return np.append(g, 0.0)


def _pair_field(d: np.ndarray, softening: float, grid_softening: float, cutoff: float) -> np.ndarray:
    """_close_field (N, 2) at offsets d (N, 2) within the cutoff, from _pair_table."""
    table = _pair_table(softening, grid_softening, cutoff)
    x = np.minimum((d[:, 0] ** 2 + d[:, 1] ** 2) * (_PAIR_SAMPLES / cutoff**2), _PAIR_SAMPLES)
    i = x.astype(np.int64)
    g = table[i] + (table[i + 1] - table[i]) * (x - i)
    return g[:, None] * d


def _close_range(d: np.ndarray, softening: float, grid_softening: float, cutoff: float) -> np.ndarray:
    """Close-range potential and field together (..., 3), for tabulating."""
    potential = _close_potential((d * d).sum(axis=-1), softening, grid_softening, cutoff)
    return np.concatenate([potential[..., None], _close_field(d, softening, grid_softening, cutoff)], axis=-1)


# Samples per cell (per axis) of the close-range table of a uniform square.
_SQUARE_SAMPLES = 32


@lru_cache(maxsize=8)
def _square_table(softening: float, grid_softening: float, cutoff: float) -> tuple[np.ndarray, float]:
    """_close_range of a unit charge spread evenly over a cell, tabulated.

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
    point = _close_range(np.stack([zx, zy], axis=-1), softening, grid_softening, cutoff)
    total = np.pad(point.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0), (0, 0)))
    n = len(z) - m + 1
    table = total[m:m + n, m:m + n] - total[:n, m:m + n] - total[m:m + n, :n] + total[:n, :n]
    return table * h**2, half


class CloseRange:
    """Close-range forces on electrons and their potential, on top of the grid fields.

    The grid smears every charge over about a cell, so two electrons barely repel
    within a cell and can sit on top of each other. Within electron_cutoff this
    corrects that (see _close_range): electrons repel as compact charges
    (electron_softening), and ions attract them as charge spread evenly over
    their cells. Electrons evenly filling a conductor then feel no net
    close-range force, as in a real metal; crowding or gaps get pushed back.

    The ions' part only depends on which cells nearby conduct. It is tabulated
    over a cell once per distinct neighbourhood, until the materials change.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._key = None

    @property
    def enabled(self) -> bool:
        return self.settings.electron_cutoff > 0

    def _softenings(self) -> tuple[float, float, float]:
        s = self.settings
        return s.electron_softening, s.conductor_softening, s.electron_cutoff

    def _update_ions(self, world) -> None:
        s, m = self.settings, world.materials
        key = (id(world), m.version, s.electron_softening, s.conductor_softening,
               s.electron_cutoff, s.coulomb_k, s.cell_charge)
        if key == self._key:
            return
        square, half = _square_table(*self._softenings())
        n = _SQUARE_SAMPLES
        reach = int(np.ceil(s.electron_cutoff))
        padded = np.pad(m.conductive, reach)
        offsets = [(dc, dr) for dc in range(-reach, reach + 1) for dr in range(-reach, reach + 1)]
        # Each cell's neighbourhood as a bit pattern: one bit per conducting neighbour.
        pattern = np.zeros((m.cols, m.rows), dtype=np.int64)
        for bit, (dc, dr) in enumerate(offsets):
            near = padded[reach + dc : reach + dc + m.cols, reach + dr : reach + dr + m.rows]
            pattern |= near.astype(np.int64) << bit
        unique, self._pattern = np.unique(pattern, return_inverse=True)
        self._pattern = self._pattern.reshape(m.cols, m.rows)
        # Table rows j = 0..n sample the cell at x = j / n; offset x - 0.5 - dc from that
        # neighbour's centre is square-table row j + (half - 0.5 - dc) n.
        self._tables = np.zeros((len(unique), n + 1, n + 1, 3))
        for bit, (dc, dr) in enumerate(offsets):
            a, b = int(round((half - 0.5 - dc) * n)), int(round((half - 0.5 - dr) * n))
            has = (unique >> bit) & 1 == 1
            self._tables[has] += square[a : a + n + 1, b : b + n + 1]
        self._tables *= s.coulomb_k * s.cell_charge
        # Ion potential averaged over each cell (trapezoid rule on the table samples).
        weights = np.ones(n + 1)
        weights[[0, -1]] = 0.5
        weights /= n
        cell_mean = np.einsum("uij,i,j->u", self._tables[..., 0], weights, weights)
        self._ion_potential = cell_mean[self._pattern]
        self._square, self._half = square, half
        self._key = key

    def _ions(self, world, points: np.ndarray) -> np.ndarray:
        """Close-range potential and field (N, 3) of all ions at points inside the grid."""
        self._update_ions(world)
        cell = np.floor(points).astype(np.int64)
        table = self._pattern[cell[:, 0], cell[:, 1]]
        return _lookup(self._tables, table, (points - cell) * _SQUARE_SAMPLES)

    def force(self, world) -> np.ndarray:
        """Close-range force on each electron (N, 2). Opposite and equal between
        electrons, so momentum is conserved."""
        s, p = self.settings, world.particles
        force = p.charge[:, None] * self._ions(world, p.pos)[:, 1:]
        if len(p) < 2:
            return force
        tree = cKDTree(p.pos, balanced_tree=False, compact_nodes=False)
        pairs = tree.query_pairs(s.electron_cutoff, output_type="ndarray")
        if len(pairs) == 0:
            return force
        i, j = pairs[:, 0], pairs[:, 1]
        field = _pair_field(p.pos[i] - p.pos[j], *self._softenings())
        pair = s.coulomb_k * (p.charge[i] * p.charge[j])[:, None] * field
        for axis in (0, 1):
            force[:, axis] += np.bincount(i, pair[:, axis], minlength=len(p))
            force[:, axis] -= np.bincount(j, pair[:, axis], minlength=len(p))
        return force

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
        s, m, p = self.settings, world.materials, world.particles
        self._update_ions(world)
        reach = int(np.ceil(s.electron_cutoff))
        pos = p.pos
        charge = p.charge
        if where is not None:
            # Only electrons within reach of a requested cell contribute to it.
            near = ndimage.binary_dilation(where, np.ones((2 * reach + 1,) * 2, dtype=bool))
            cols, rows = p.cells()
            pos, charge = pos[near[cols, rows]], charge[near[cols, rows]]
        steps = np.arange(-reach, reach + 1)
        offsets = np.stack(np.meshgrid(steps, steps, indexing="ij"), axis=-1).reshape(-1, 1, 2)
        cells = np.floor(pos).astype(np.int64) + offsets
        c, r = cells[..., 0], cells[..., 1]
        hit = (c >= 0) & (c < m.cols) & (r >= 0) & (r < m.rows)
        # An electron's potential averaged over a cell is that of the cell, spread
        # evenly, at the electron: the square table.
        d = (pos - (cells + 0.5))[hit]
        index = (d + self._half) * _SQUARE_SAMPLES
        values = _lookup(self._square[None], np.zeros(len(d), dtype=np.int64), index)[:, 0]
        electrons = np.bincount(
            (c * m.rows + r)[hit], (values * np.broadcast_to(charge, c.shape)[hit]),
            minlength=m.cols * m.rows,
        )
        potential = self._ion_potential + s.coulomb_k * electrons.reshape(m.cols, m.rows)
        return potential if where is None else np.where(where, potential, 0.0)


def _lookup(tables: np.ndarray, which: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Bilinear interpolation in tables[which] (n, n, d) at fractional indices (N, 2)."""
    i = np.clip(np.floor(index).astype(np.int64), 0, tables.shape[1] - 2)
    f = index - i
    fx, fy = f[:, 0, None], f[:, 1, None]
    ix, iy = i[:, 0], i[:, 1]
    return (
        tables[which, ix, iy] * (1 - fx) * (1 - fy)
        + tables[which, ix + 1, iy] * fx * (1 - fy)
        + tables[which, ix, iy + 1] * (1 - fx) * fy
        + tables[which, ix + 1, iy + 1] * fx * fy
    )


# Period (cells) of the grid the assignment correction is computed on. The correction
# is short-ranged (below 0.1% of the kernel beyond 20 cells, ~1e-4 at 60), so offsets
# of half the period or more are dropped.
_CORRECTION_PERIOD = 128
# Aliases summed per axis in the optimal influence function: beyond 2 they are
# negligible for any softening of a few tenths of a cell or more.
_ALIASES = 2


def _reference_spectra(qx: np.ndarray, qy: np.ndarray, softening: float) -> tuple[np.ndarray, ...]:
    """Continuous 2D Fourier transforms of the softened field (x, y) and potential of
    a unit charge: the potential erf(r / eps) / r transforms to 2 pi erfc(q eps / 2) / q,
    and the field is -i q times that. Zero at q = 0 (only the potential has a pole there;
    it cancels in the correction).
    """
    q = np.hypot(qx, qy)
    zero = q == 0
    q_safe = np.where(zero, 1.0, q)
    potential = np.where(zero, 0.0, 2 * np.pi * special.erfc(q * softening / 2) / q_safe)
    return -1j * qx * potential, -1j * qy * potential, potential


@lru_cache(maxsize=8)
def _assignment_correction(softening: float) -> tuple[np.ndarray, ...]:
    """What the optimal influence function adds to the sampled kernels: (field x,
    field y, potential), each (_CORRECTION_PERIOD,) * 2 indexed by offset modulo the period.

    Cloud-in-cell assignment smears each charge once when depositing and once more
    when reading back, weakening the field at short range. Hockney & Eastwood's
    optimal influence function compensates for it, aliasing included, in the
    least-squares sense averaged over particle positions:

        K_opt(k) = sum_m U(k_m)^2 R(k_m) / (sum_m U(k_m)^2)^2,   k_m = k + 2 pi m

    with U the cloud-in-cell window (a sinc^2 per axis) and R the reference
    spectrum. The sampled kernel already has the spectrum sum_m R(k_m); the
    difference transforms to a short-ranged real-space correction.
    """
    n = _CORRECTION_PERIOD
    k = 2 * np.pi * np.fft.fftfreq(n)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    weighted = [0.0, 0.0, 0.0]
    sampled = [0.0, 0.0, 0.0]
    for mx in range(-_ALIASES, _ALIASES + 1):
        for my in range(-_ALIASES, _ALIASES + 1):
            qx, qy = kx + 2 * np.pi * mx, ky + 2 * np.pi * my
            window2 = (np.sinc(qx / (2 * np.pi)) * np.sinc(qy / (2 * np.pi))) ** 4
            for i, reference in enumerate(_reference_spectra(qx, qy, softening)):
                weighted[i] = weighted[i] + window2 * reference
                sampled[i] = sampled[i] + reference
    # sum_m U(k_m)^2 in closed form, per axis: 1 - 2/3 sin^2(k / 2).
    windows = (1 - 2 / 3 * np.sin(kx / 2) ** 2) * (1 - 2 / 3 * np.sin(ky / 2) ** 2)
    return tuple(np.fft.ifft2(w / windows**2 - r).real for w, r in zip(weighted, sampled))


def field_kernels(
    cols: int, rows: int, softening: float, k: float = 1.0, corrected: bool = True
) -> tuple[np.ndarray, ...]:
    """Real-space kernels (field x, field y, potential) at offsets -(n - 1) .. n - 1 per
    axis (offset 0 at index n - 1): a unit charge's softened field and potential,
    plus the assignment correction if `corrected`.
    """
    dx, dy = np.meshgrid(np.arange(-(cols - 1), cols), np.arange(-(rows - 1), rows), indexing="ij")
    r2 = (dx**2 + dy**2).astype(float)
    inv_r3 = softened_inv_r3(r2, softening)
    kernels = [dx * inv_r3, dy * inv_r3, softened_inv_r(r2, softening)]
    if corrected:
        half = _CORRECTION_PERIOD // 2
        inside = (np.abs(dx) < half) & (np.abs(dy) < half)
        for kernel, correction in zip(kernels, _assignment_correction(softening)):
            kernel += np.where(inside, correction[dx % _CORRECTION_PERIOD, dy % _CORRECTION_PERIOD], 0.0)
    return tuple(k * kernel for kernel in kernels)


class GridKernel:
    """Field and potential at every cell centre from charges on cell centres, via FFT.

    Uncorrected, equivalent to coulomb_field / coulomb_potential over all cells, but
    O(n log n). Corrected (the default), tuned for charges deposited and read back by
    cloud-in-cell (see _assignment_correction). The kernel transforms are computed
    once per grid size and softening.
    """

    def __init__(self, cols: int, rows: int, softening: float, k: float = 1.0, corrected: bool = True):
        self.cols, self.rows = cols, rows
        # Circular convolution is exact for the outputs we keep when the transform
        # is at least as long as the kernel (2n - 1): nothing wraps into them.
        self.shape = (
            fft.next_fast_len(2 * cols - 1, real=True),
            fft.next_fast_len(2 * rows - 1, real=True),
        )
        kx, ky, kv = field_kernels(cols, rows, softening, k, corrected)
        self._kx = fft.rfft2(kx, s=self.shape)
        self._ky = fft.rfft2(ky, s=self.shape)
        self._kv = fft.rfft2(kv, s=self.shape)
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
        # Combine both terms before transforming back (it's linear): one inverse FFT.
        return fft.irfft2(jx * self._ky - jy * self._kx, s=self.shape)[self._keep]


@lru_cache(maxsize=8)
def grid_kernel(cols: int, rows: int, softening: float, k: float = 1.0, corrected: bool = True) -> GridKernel:
    return GridKernel(cols, rows, softening, k, corrected)


class CloudInCell:
    """Bilinear (cloud-in-cell) weights of particles onto the 4 nearest cell centres.

    Computed once per particle configuration, then reused to deposit charge and
    current and to read fields back. Using identical weights both ways makes the
    grid force on a particle from its own charge cancel exactly (the Coulomb and
    Biot-Savart kernels are antisymmetric). Corners outside the grid are clamped
    onto the edge cells, so weights per particle still sum to 1.
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
        """Spread one value per particle over the grid, shape (cols, rows). Conserves the sum."""
        size = self.shape[0] * self.shape[1]
        flat = np.bincount(self.index.ravel(), (self.weight * values).ravel(), minlength=size)
        return flat.reshape(self.shape)

    def gather(self, grid: np.ndarray) -> np.ndarray:
        """Read a (cols, rows, d) grid at the particles, shape (particles, d)."""
        flat = grid.reshape(self.shape[0] * self.shape[1], -1)
        return (flat[self.index] * self.weight[..., None]).sum(axis=0)


def deposit(positions: np.ndarray, charges: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """Spread point charges over the 4 nearest cell centres (see CloudInCell)."""
    return CloudInCell(positions, cols, rows).deposit(charges)


def interpolate(grid: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Read a (cols, rows, d) grid at positions with the same weights `deposit` uses."""
    return CloudInCell(positions, *grid.shape[:2]).gather(grid)


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


class ElectricField:
    """Electric field of a World: ions, electrons and fixed test charges.

    `at` is exact (direct sums). `grid` is the fast version, used for display and
    for forces: all charges go through FFT convolution on the grid. Electrons are
    deposited onto cell centres by cloud-in-cell, and ions spread over their cells
    (see ion_charge) are cloud-in-cell deposits too, so both use the kernel with the
    assignment correction. Test charges sit exactly on cell centres: nothing to
    correct, so their part is exact. Ions and test charges are cached until they change.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._static_key = None
        self._static_grid: np.ndarray | None = None
        self._static_potential: np.ndarray | None = None
        self.close_range = CloseRange(settings)

    def _kernel(self, world, softening: float, corrected: bool = True) -> GridKernel:
        m = world.materials
        return grid_kernel(m.cols, m.rows, softening, self.settings.coulomb_k, corrected)

    def _update_static(self, world) -> None:
        """(Re)compute the field and potential of ions and test charges if they changed."""
        key = (id(world), world.materials.version, world.fixed_version)
        if key == self._static_key:
            return
        s, m = self.settings, world.materials
        ion_kernel = self._kernel(world, s.conductor_softening)
        ions = ion_charge(m, s.cell_charge)
        fixed_kernel = self._kernel(world, s.test_charge_softening, corrected=False)
        fixed = deposit(world.fixed_pos, world.fixed_charge, m.cols, m.rows)
        self._static_grid = ion_kernel.field(ions) + fixed_kernel.field(fixed)
        self._static_potential = ion_kernel.potential(ions) + fixed_kernel.potential(fixed)
        self._static_key = key

    def static_grid(self, world) -> np.ndarray:
        """Field of ions and test charges at every cell centre, shape (cols, rows, 2)."""
        self._update_static(world)
        return self._static_grid

    def _electron_charge(self, world, cic: CloudInCell | None) -> np.ndarray:
        m, p = world.materials, world.particles
        cic = cic or CloudInCell(p.pos, m.cols, m.rows)
        return cic.deposit(p.charge)

    def electron_grid(self, world, cic: CloudInCell | None = None) -> np.ndarray:
        """Approximate electron field at every cell centre, shape (cols, rows, 2).

        Pass `cic` (weights of the current electron positions) to reuse it.
        """
        kernel = self._kernel(world, self.settings.conductor_softening)
        return kernel.field(self._electron_charge(world, cic))

    def grid(self, world, cic: CloudInCell | None = None) -> np.ndarray:
        """Total field at every cell centre, shape (cols, rows, 2)."""
        return self.static_grid(world) + self.electron_grid(world, cic)

    def potential_grid(self, world, where: np.ndarray | None = None) -> np.ndarray:
        """Total potential at every cell centre, shape (cols, rows); zero far away.

        The close-range part (see CloseRange.potential) is only added on the cells
        in `where`, if given.
        """
        self._update_static(world)
        kernel = self._kernel(world, self.settings.conductor_softening)
        potential = self._static_potential + kernel.potential(self._electron_charge(world, None))
        if self.close_range.enabled:
            potential = potential + self.close_range.potential(world, where)
        return potential

    def at(self, world, points: np.ndarray) -> np.ndarray:
        """Exact total field at arbitrary points (M, 2), all sources summed directly."""
        return self._sum_sources(world, points, coulomb_field)

    def potential_at(self, world, points: np.ndarray) -> np.ndarray:
        """Exact electric potential at arbitrary points (M,), zero far from all charges."""
        return self._sum_sources(world, points, coulomb_potential)

    def _sum_sources(self, world, points: np.ndarray, kernel) -> np.ndarray:
        s = self.settings
        ion_grid = ion_charge(world.materials, s.cell_charge)
        ions = np.argwhere(ion_grid != 0) + 0.5
        ion_charges = ion_grid[ion_grid != 0]
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

    def electron_current(self, world, cic: CloudInCell | None = None) -> np.ndarray:
        """Electron current q*v deposited onto cell centres (cloud-in-cell), (cols, rows, 2)."""
        m, p = world.materials, world.particles
        cic = cic or CloudInCell(p.pos, m.cols, m.rows)
        return np.stack(
            [cic.deposit(p.charge * p.vel[:, 0]), cic.deposit(p.charge * p.vel[:, 1])], axis=-1
        )


def poynting(e_grid: np.ndarray, bz: np.ndarray, settings: Settings) -> np.ndarray:
    """Poynting vector S = E x B / mu0 per cell, shape (cols, rows, 2): energy flow.

    With E in the plane and B = (0, 0, B_z), E x B = (E_y B_z, -E_x B_z) is back
    in the plane. 1/mu0 = c^2 / (4 pi k) in our units.
    """
    inv_mu0 = settings.speed_of_light**2 / (4 * np.pi * settings.coulomb_k)
    return inv_mu0 * np.stack([e_grid[..., 1] * bz, -e_grid[..., 0] * bz], axis=-1)
