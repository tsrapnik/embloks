"""The fields: Maxwell's equations stepped in time on a staggered grid (FDTD).

This is genuine 2D electromagnetism (TE mode: Ex, Ey in the plane, Bz out of it),
as in particle-in-cell plasma codes. Charges are infinitely long lines in z, so
static fields fall off as 1/r instead of 1/r^2.

    dBz/dt = -(dEy/dx - dEx/dy)
    dE/dt  = c^2 curl(Bz) - J / eps0          (eps0 = 1 / (4 pi k))

Layout (Yee grid; nodes are cell centres, as everywhere else in the app):
    Ex on vertical cell edges, between nodes along x:   shape (nx + 1, ny)
    Ey on horizontal cell edges, between nodes along y:  shape (nx, ny + 1)
    Bz on cell corners:                                   shape (nx + 1, ny + 1)

Every tick E is corrected so that Gauss's law div E = rho / eps0 holds exactly
(divergence cleaning). The electrostatic part of E is then instantaneous, as in
the Coulomb gauge, and only the part made by changing currents travels at c.
That also keeps the field right when charge appears or jumps without a current
(painting, battery transfers). A margin around the visible grid with gradually
increasing damping absorbs outgoing waves.
"""

import numpy as np
from scipy import fft

from embloks.config import Settings
from embloks.fields import CloudInCell, ion_charge


class Maxwell:
    def __init__(self, cols: int, rows: int, settings: Settings):
        self.cols, self.rows = cols, rows
        self.settings = settings
        self.margin = m = settings.absorbing_margin
        if m < settings.smoothing_passes:
            raise ValueError("absorbing_margin must be at least smoothing_passes")
        self.nx, self.ny = nx, ny = cols + 2 * m, rows + 2 * m
        self.ex = np.zeros((nx + 1, ny))
        self.ey = np.zeros((nx, ny + 1))
        self.bz = np.zeros((nx + 1, ny + 1))
        self.potential = np.zeros((nx, ny))
        # Eigenvalues of the 5-point Laplacian with zero boundary values, diagonal in
        # the type-I sine transform: Poisson's equation is solved exactly in one go.
        kx = 2 * np.cos(np.pi * np.arange(1, nx + 1) / (nx + 1)) - 2
        ky = 2 * np.cos(np.pi * np.arange(1, ny + 1) / (ny + 1)) - 2
        self._inv_laplacian = 1 / (kx[:, None] + ky[None, :])
        self._curl = np.zeros((nx + 1, ny + 1))
        self._damping_key = None

    @classmethod
    def for_world(cls, world, settings: Settings) -> "Maxwell":
        """Fields for `world`, starting from rest: only the static E of its charges."""
        m = world.materials
        maxwell = cls(m.cols, m.rows, settings)
        maxwell.enforce_gauss(maxwell.sources(world)[0])
        return maxwell

    # --- Geometry -------------------------------------------------------------------

    def visible(self, array: np.ndarray) -> np.ndarray:
        """The part of a node array (nx, ny, ...) covering the visible grid."""
        m = self.margin
        return array[m : m + self.cols, m : m + self.rows]

    def _damping(self, dt: float):
        """Per-step damping factors for Ex, Ey, Bz: 1 inside, falling in the margin."""
        if self._damping_key == dt:
            return self._damp
        m, nx, ny = self.margin, self.nx, self.ny

        def depth(coords: np.ndarray, n: int) -> np.ndarray:
            # 0 inside the visible nodes (margin .. n - 1 - margin), 1 at the outer edge.
            outside = np.maximum(np.maximum(m - coords, coords - (n - 1 - m)), 0)
            return np.minimum(outside / m, 1.0)

        x_nodes, y_nodes = np.arange(nx), np.arange(ny)
        x_edges, y_edges = np.arange(nx + 1) - 0.5, np.arange(ny + 1) - 0.5
        strength = self.settings.absorbing_strength

        def factor(x: np.ndarray, y: np.ndarray, nx_: int, ny_: int) -> np.ndarray:
            d = np.maximum(depth(x, nx_)[:, None], depth(y, ny_)[None, :])
            return np.exp(-strength * d**3 * dt)

        self._damp = (
            factor(x_edges, y_nodes, nx, ny),
            factor(x_nodes, y_edges, nx, ny),
            factor(x_edges, y_edges, nx, ny),
        )
        self._damping_key = dt
        return self._damp

    # --- Field operations -----------------------------------------------------------

    def _poisson(self, rhs: np.ndarray) -> np.ndarray:
        """phi with (discrete) laplacian(phi) = rhs and phi = 0 just outside the grid."""
        return fft.idstn(fft.dstn(rhs, type=1) * self._inv_laplacian, type=1)

    def divergence(self) -> np.ndarray:
        return (self.ex[1:] - self.ex[:-1]) + (self.ey[:, 1:] - self.ey[:, :-1])

    def _gradient_subtract(self, phi: np.ndarray) -> None:
        """E -= grad(phi), with phi = 0 outside the grid."""
        self.ex[1:-1] -= phi[1:] - phi[:-1]
        self.ex[0] -= phi[0]
        self.ex[-1] += phi[-1]
        self.ey[:, 1:-1] -= phi[:, 1:] - phi[:, :-1]
        self.ey[:, 0] -= phi[:, 0]
        self.ey[:, -1] += phi[:, -1]

    def enforce_gauss(self, rho: np.ndarray) -> None:
        """Correct E so div E = rho / eps0 exactly, keeping its wave (curl) part.

        Also stores the electric potential of rho (Coulomb gauge), which the
        batteries measure. Poisson's equation is linear, so the correction
        poisson(div E - 4 pi k rho) is poisson(div E) plus that potential.
        """
        self.potential = self._poisson(-4 * np.pi * self.settings.coulomb_k * rho)
        self._gradient_subtract(self._poisson(self.divergence()) + self.potential)

    def _curl_e(self) -> np.ndarray:
        """dEy/dx - dEx/dy at the corners (E is zero beyond the grid). Reuses a buffer."""
        curl = self._curl
        curl[:-1] = self.ey
        curl[-1] = 0.0
        curl[1:] -= self.ey
        curl[:, :-1] -= self.ex
        curl[:, 1:] += self.ex
        return curl

    def step(self, rho: np.ndarray, current: np.ndarray, dt: float) -> None:
        """Advance E and B by dt. `rho` (nx, ny) and `current` (nx, ny, 2) live on nodes.

        The scheme is stable while light crosses under 1/sqrt(2) cell per step (CFL), so
        a fast speed of light gets split into smaller steps. Gauss's law is enforced
        once, at the end: the correction is a gradient, which has no curl, so it
        doesn't change B or the waves in between.
        """
        s = self.settings
        substeps = max(1, int(np.ceil(s.speed_of_light * dt * np.sqrt(2) / 0.9)))
        h = dt / substeps
        c2 = s.speed_of_light**2
        four_pi_k = 4 * np.pi * s.coulomb_k
        # Currents from nodes onto the edges where E lives (boundary edges get none).
        jx = np.zeros_like(self.ex)
        jx[1:-1] = 0.5 * (current[:-1, :, 0] + current[1:, :, 0])
        jy = np.zeros_like(self.ey)
        jy[:, 1:-1] = 0.5 * (current[:, :-1, 1] + current[:, 1:, 1])
        jx *= four_pi_k * h
        jy *= four_pi_k * h
        damp_ex, damp_ey, damp_bz = self._damping(h)
        for _ in range(substeps):
            # Leapfrog: half a step of B, a full step of E, another half step of B.
            self.bz -= 0.5 * h * self._curl_e()
            self.ex += h * c2 * (self.bz[:, 1:] - self.bz[:, :-1]) - jx
            self.ey -= h * c2 * (self.bz[1:] - self.bz[:-1]) + jy
            self.bz -= 0.5 * h * self._curl_e()
            self.ex *= damp_ex
            self.ey *= damp_ey
            self.bz *= damp_bz
        self.enforce_gauss(rho)

    # --- Coupling to the world ------------------------------------------------------

    def smoothed(self, values: np.ndarray) -> np.ndarray:
        """Visible-grid node values (cols, rows, ...) smoothed and placed on the full grid.

        Smoothing runs on the visible grid plus a border as wide as it spreads,
        which gives the same result as smoothing the whole (otherwise zero) grid.
        """
        k = self.settings.smoothing_passes
        values = np.pad(values, [(k, k), (k, k)] + [(0, 0)] * (values.ndim - 2))
        for _ in range(k):
            values = _smooth(values)
        out = np.zeros((self.nx, self.ny) + values.shape[2:])
        m = self.margin - k
        out[m : m + values.shape[0], m : m + values.shape[1]] = values
        return out

    def sources(self, world, battery_current: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Smoothed charge (nx, ny) and current (nx, ny, 2) of the world on the full grid.

        All charge counts: ions, test charges and electrons; the current is the
        electrons' q*v plus the batteries' internal currents.
        """
        s, m, p = self.settings, world.materials, world.particles
        cic = CloudInCell(p.pos, m.cols, m.rows)
        values = np.zeros((m.cols, m.rows, 3))
        values[..., 0] = ion_charge(m, s.cell_charge) + cic.deposit(p.charge)
        if len(world.fixed_charge):
            values[..., 0] += CloudInCell(world.fixed_pos, m.cols, m.rows).deposit(world.fixed_charge)
        values[..., 1:] = cic.deposit(p.charge[:, None] * p.vel)
        if battery_current is not None:
            values[..., 1:] += battery_current
        full = self.smoothed(values)
        return full[..., 0], full[..., 1:]

    def static_potential(self, rho: np.ndarray) -> np.ndarray:
        """Potential (cols, rows) of visible-grid node charges `rho`, smoothed like all
        charge, on its own: the fields themselves are left alone."""
        return self.visible(self._poisson(-4 * np.pi * self.settings.coulomb_k * self.smoothed(rho)))

    def advance(self, world, dt: float, battery_current: np.ndarray) -> None:
        """One tick: deposit charge and current from the world, then step the fields."""
        self.step(*self.sources(world, battery_current), dt)

    def gather(self, positions: np.ndarray) -> np.ndarray:
        """Ex, Ey, Bz at particle positions (in visible-grid cell coordinates), (N, 3).

        Each component is interpolated bilinearly from its own staggered grid. (First
        averaging onto cell centres would hide an odd/even "checkerboard" of charge from
        the particles: it would feel no force, nothing would push back, and it grows.)
        """
        m = self.margin
        x, y = positions[:, 0], positions[:, 1]
        # Fractional array indices: Ex sits on vertical cell edges (x integer, y centred),
        # Ey on horizontal edges, Bz on corners; node i is cell centre i - m + 0.5.
        return np.stack([
            _bilinear(self.ex, x + m, y - 0.5 + m),
            _bilinear(self.ey, x - 0.5 + m, y + m),
            _bilinear(self.bz, x + m, y + m),
        ], axis=-1)

    def node_fields(self) -> tuple[np.ndarray, np.ndarray]:
        """E (cols, rows, 2) and Bz (cols, rows) at the visible cell centres."""
        e = np.stack(
            [0.5 * (self.ex[:-1] + self.ex[1:]), 0.5 * (self.ey[:, :-1] + self.ey[:, 1:])], axis=-1
        )
        b = 0.25 * (self.bz[:-1, :-1] + self.bz[1:, :-1] + self.bz[:-1, 1:] + self.bz[1:, 1:])
        return self.visible(e), self.visible(b)

    def visible_potential(self) -> np.ndarray:
        return self.visible(self.potential)


def _bilinear(array: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a 2D array at fractional indices (gx, gy)."""
    i = np.clip(np.floor(gx).astype(np.int64), 0, array.shape[0] - 2)
    j = np.clip(np.floor(gy).astype(np.int64), 0, array.shape[1] - 2)
    fx, fy = gx - i, gy - j
    return (
        array[i, j] * (1 - fx) * (1 - fy)
        + array[i + 1, j] * fx * (1 - fy)
        + array[i, j + 1] * (1 - fx) * fy
        + array[i + 1, j + 1] * fx * fy
    )


def _smooth(array: np.ndarray) -> np.ndarray:
    """One pass of the 1-2-1 binomial filter along x and y, zero beyond the edges.

    Smears charge and current over neighbouring cells, the grid's counterpart of a
    softening length: without it, single electrons in a one-cell-wide wire make the
    field vary wildly within the cell. Conserves charge that stays clear of the edges.
    """
    for axis in (0, 1):
        a = np.moveaxis(array, axis, 0)
        out = 0.5 * a
        out[1:] += 0.25 * a[:-1]
        out[:-1] += 0.25 * a[1:]
        array = np.moveaxis(out, 0, axis)
    return array
