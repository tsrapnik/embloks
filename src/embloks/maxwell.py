"""Dynamic fields: Maxwell's equations stepped in time on a staggered grid (FDTD).

This is genuine 2D electromagnetism (TE mode: Ex, Ey in the plane, Bz out of it),
as in particle-in-cell plasma codes. Charges are infinitely long lines in z, so
static fields fall off as 1/r instead of 1/r^2 as in the instant-field mode.

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
from embloks.fields import CloudInCell


class Maxwell:
    def __init__(self, cols: int, rows: int, settings: Settings):
        self.cols, self.rows = cols, rows
        self.settings = settings
        self.margin = m = settings.absorbing_margin
        self.nx, self.ny = nx, ny = cols + 2 * m, rows + 2 * m
        self.ex = np.zeros((nx + 1, ny))
        self.ey = np.zeros((nx, ny + 1))
        self.bz = np.zeros((nx + 1, ny + 1))
        self.potential = np.zeros((nx, ny))
        # Eigenvalues of the 5-point Laplacian with zero boundary values, diagonal in
        # the type-I sine transform: Poisson's equation is solved exactly in one go.
        kx = 2 * np.cos(np.pi * np.arange(1, nx + 1) / (nx + 1)) - 2
        ky = 2 * np.cos(np.pi * np.arange(1, ny + 1) / (ny + 1)) - 2
        self._laplacian = kx[:, None] + ky[None, :]
        self._damping_key = None

    # --- Geometry -------------------------------------------------------------------

    def visible(self, array: np.ndarray) -> np.ndarray:
        """The part of a node array (nx, ny, ...) covering the visible grid."""
        m = self.margin
        return array[m : m + self.cols, m : m + self.rows]

    def embed(self, array: np.ndarray) -> np.ndarray:
        """A visible-grid node array placed inside a zero node array (nx, ny, ...)."""
        out = np.zeros((self.nx, self.ny) + array.shape[2:])
        self.visible(out)[...] = array
        return out

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
        return fft.idstn(fft.dstn(rhs, type=1) / self._laplacian, type=1)

    def divergence(self) -> np.ndarray:
        return (self.ex[1:] - self.ex[:-1]) + (self.ey[:, 1:] - self.ey[:, :-1])

    def _gradient_subtract(self, phi: np.ndarray) -> None:
        """E -= grad(phi), with phi = 0 outside the grid."""
        padded = np.pad(phi, 1)
        self.ex -= padded[1:, 1:-1] - padded[:-1, 1:-1]
        self.ey -= padded[1:-1, 1:] - padded[1:-1, :-1]

    def enforce_gauss(self, rho: np.ndarray) -> None:
        """Correct E so div E = rho / eps0 exactly, keeping its wave (curl) part.

        Also stores the electric potential of rho (Coulomb gauge), which the
        batteries measure.
        """
        four_pi_k = 4 * np.pi * self.settings.coulomb_k
        self._gradient_subtract(self._poisson(self.divergence() - four_pi_k * rho))
        self.potential = self._poisson(-four_pi_k * rho)

    def _curl_e(self) -> np.ndarray:
        """dEy/dx - dEx/dy at the corners."""
        ey = np.pad(self.ey, ((1, 1), (0, 0)))
        ex = np.pad(self.ex, ((0, 0), (1, 1)))
        return (ey[1:] - ey[:-1]) - (ex[:, 1:] - ex[:, :-1])

    def step(self, rho: np.ndarray, current: np.ndarray, dt: float) -> None:
        """Advance E and B by dt. `rho` (nx, ny) and `current` (nx, ny, 2) live on nodes.

        The scheme is stable while light crosses under 1/sqrt(2) cell per step (CFL), so
        a fast speed of light gets split into smaller steps.
        """
        courant = self.settings.speed_of_light * dt * np.sqrt(2)
        substeps = max(1, int(np.ceil(courant / 0.9)))
        for _ in range(substeps):
            self._step(rho, current, dt / substeps)

    def _step(self, rho: np.ndarray, current: np.ndarray, dt: float) -> None:
        c2 = self.settings.speed_of_light**2
        four_pi_k = 4 * np.pi * self.settings.coulomb_k
        # Currents from nodes onto the edges where E lives (boundary edges get none).
        jx = np.zeros_like(self.ex)
        jx[1:-1] = 0.5 * (current[:-1, :, 0] + current[1:, :, 0])
        jy = np.zeros_like(self.ey)
        jy[:, 1:-1] = 0.5 * (current[:, :-1, 1] + current[:, 1:, 1])
        # Leapfrog: half a step of B, a full step of E, another half step of B.
        self.bz -= 0.5 * dt * self._curl_e()
        self.ex += dt * (c2 * (self.bz[:, 1:] - self.bz[:, :-1]) - four_pi_k * jx)
        self.ey += dt * (-c2 * (self.bz[1:] - self.bz[:-1]) - four_pi_k * jy)
        self.bz -= 0.5 * dt * self._curl_e()
        damp_ex, damp_ey, damp_bz = self._damping(dt)
        self.ex *= damp_ex
        self.ey *= damp_ey
        self.bz *= damp_bz
        self.enforce_gauss(rho)

    # --- Coupling to the world ------------------------------------------------------

    def charge(self, world, cic: CloudInCell) -> np.ndarray:
        """All charge (ions, test charges, electrons) on the full node grid."""
        s, m, p = self.settings, world.materials, world.particles
        rho = m.conductive * s.cell_charge + cic.deposit(p.charge)
        if len(world.fixed_charge):
            rho = rho + CloudInCell(world.fixed_pos, m.cols, m.rows).deposit(world.fixed_charge)
        return self.embed(rho)

    def advance(self, world, dt: float, battery_current: np.ndarray) -> None:
        """One tick: deposit charge and current from the world, then step the fields."""
        m, p = world.materials, world.particles
        cic = CloudInCell(p.pos, m.cols, m.rows)
        current = np.stack(
            [cic.deposit(p.charge * p.vel[:, 0]), cic.deposit(p.charge * p.vel[:, 1])], axis=-1
        )
        rho = self.charge(world, cic)
        current = self.embed(current + battery_current)
        for _ in range(self.settings.dynamic_smoothing_passes):
            rho, current = _smooth(rho), _smooth(current)
        self.step(rho, current, dt)

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
    """One pass of the 1-2-1 binomial filter along x and y (charge is conserved).

    Smears charge and current over neighbouring cells, like the softening length of
    the instant-field mode: without it, single electrons in a one-cell-wide wire
    make the field vary wildly within the cell.
    """
    for axis in (0, 1):
        n = array.shape[axis]
        padded = np.pad(array, [(1, 1) if a == axis else (0, 0) for a in range(array.ndim)])
        left, middle, right = (np.take(padded, np.arange(k, k + n), axis=axis) for k in (0, 1, 2))
        array = 0.25 * left + 0.5 * middle + 0.25 * right
    return array
