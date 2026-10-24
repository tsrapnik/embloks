import numpy as np

from embloks.battery import Stack, find_stacks
from embloks.config import Settings
from embloks.connectivity import compute_connectivity
from embloks.materials import Material, Materials, Orientation
from embloks.particles import Particles

# Electrons spawn spread over their cell (see fill_pattern), offset by at most this
# much (cells), so a fresh conductor is not a perfect lattice.
SPAWN_JITTER = 0.02


class World:
    """Materials plus the charges they carry, kept consistent while painting.

    Every conductor cell holds one fixed ion of charge +cell_charge, spread evenly
    over the cell (see fields.ion_charge), and, when neutral, `carriers_per_cell` electrons of charge -cell_charge/n each.
    """

    def __init__(self, cols: int, rows: int, settings: Settings, rng: np.random.Generator | None = None):
        self.settings = settings
        self.rng = rng or np.random.default_rng()
        self.materials = Materials(cols, rows)
        self.particles = Particles()
        # Per battery stack: charge owed but not yet moved (less than one electron),
        # and smoothed current, for display.
        self.pump_charge: dict[Stack, float] = {}
        self.pump_current: dict[Stack, float] = {}
        # The same pumped charge per time, averaged only briefly: the battery current the
        # fields see (the slow average would lag an AC source by much of a period).
        self.pump_flow: dict[Stack, float] = {}
        self._update_structure()
        # Fixed test charges, at most one per cell, at cell centres.
        self.fixed_pos = np.empty((0, 2))
        self.fixed_charge = np.empty(0)
        # Bumped whenever test charges change, so cached fields know to rebuild.
        self.fixed_version = 0

    @property
    def electron_charge(self) -> float:
        return -self.settings.cell_charge / self.settings.carriers_per_cell

    @property
    def electron_mass(self) -> float:
        # Mass shrinks with charge, so charge/mass and thus the dynamics don't depend on n.
        return -self.electron_charge / self.settings.carrier_charge_to_mass

    def paint(
        self,
        cols: np.ndarray,
        rows: np.ndarray,
        material: Material,
        resistance: float = 0.0,
        emf: float = 0.0,
        orientation: Orientation = Orientation.RIGHT,
        frequency: float = 0.0,
    ) -> None:
        was_conductive = self.materials.conductive
        old_labels = self.connectivity.labels
        self.materials.paint(cols, rows, material, resistance, emf, orientation, frequency)
        self._update_structure()

        is_conductive = self.materials.conductive
        removed = was_conductive & ~is_conductive
        added = is_conductive & ~was_conductive
        if removed.any():
            self._remove_carriers(removed, old_labels)
        if added.any():
            self._spawn_carriers(added)

    def clear(self) -> None:
        self.materials.clear()
        self.particles = Particles()
        self._update_structure()
        self.fixed_pos = np.empty((0, 2))
        self.fixed_charge = np.empty(0)
        self.fixed_version += 1

    def _update_structure(self) -> None:
        """Recompute islands and battery stacks after the materials changed."""
        self.connectivity = compute_connectivity(self.materials)
        self.stacks = find_stacks(self.materials, self.settings.battery_resistance)
        # Distance along the wires from every battery terminal, for choosing which
        # electron a battery takes: the nearest one *along the wire*, never one
        # that is merely close in a straight line (e.g. just across the battery).
        terminals = sorted(
            {t for st in self.stacks for t in (st.low, st.high) if self._is_conductor(t)}
        )
        distances = self.connectivity.distances_from(terminals)
        self.terminal_distance = dict(zip(terminals, distances))
        # Forget pump state of stacks that no longer exist.
        for state in (self.pump_charge, self.pump_current, self.pump_flow):
            for stack in set(state) - set(self.stacks):
                del state[stack]

    def _is_conductor(self, cell: tuple[int, int]) -> bool:
        c, r = cell
        m = self.materials
        return 0 <= c < m.cols and 0 <= r < m.rows and bool(m.conductive[c, r])

    def place_charges(self, cols: np.ndarray, rows: np.ndarray, charge: float) -> None:
        """Put a fixed test charge at the centre of each given cell, replacing any there."""
        self.remove_charges(cols, rows)
        cells = np.unique(np.stack([cols, rows], axis=-1), axis=0)
        self.fixed_pos = np.concatenate([self.fixed_pos, cells + 0.5])
        self.fixed_charge = np.concatenate([self.fixed_charge, np.full(len(cells), charge)])
        self.fixed_version += 1

    def charges_in(self, cols: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Mask over the fixed test charges: which sit in one of the given cells."""
        fixed_cells = np.floor(self.fixed_pos).astype(np.int64)
        n_rows = self.materials.rows
        return np.isin(fixed_cells[:, 0] * n_rows + fixed_cells[:, 1], cols * n_rows + rows)

    def remove_charges(self, cols: np.ndarray, rows: np.ndarray) -> None:
        """Remove the fixed test charges in the given cells."""
        hit = self.charges_in(cols, rows)
        self.fixed_pos = self.fixed_pos[~hit]
        self.fixed_charge = self.fixed_charge[~hit]
        self.fixed_version += 1

    def _spawn_carriers(self, cells: np.ndarray) -> None:
        n = self.settings.carriers_per_cell
        corners = np.repeat(np.argwhere(cells), n, axis=0)
        offsets = np.tile(fill_pattern(n), (int(cells.sum()), 1))
        positions = corners + offsets + self._jitter(corners.shape)
        self.particles.add(positions, self.electron_charge, self.electron_mass)

    def _jitter(self, shape: tuple[int, ...]) -> np.ndarray:
        return self.rng.uniform(-SPAWN_JITTER, SPAWN_JITTER, size=shape)

    def _remove_carriers(self, removed: np.ndarray, old_labels: np.ndarray) -> None:
        """Take away the electrons that belonged to the erased cells.

        Per island, n electrons leave for each erased cell: first those sitting in
        erased cells, then the ones nearest to the erased cells. Any electrons
        still in erased cells after that move to the nearest remaining cell of
        their old island, or vanish if nothing of it is left.
        """
        p = self.particles
        n = self.settings.carriers_per_cell
        cell_c, cell_r = p.cells()
        particle_island = old_labels[cell_c, cell_r]
        in_removed = removed[cell_c, cell_r]
        keep = np.ones(len(p), dtype=bool)

        for island in np.unique(old_labels[removed]):
            removed_here = removed & (old_labels == island)
            quota = n * int(removed_here.sum())
            members = particle_island == island
            displaced = np.flatnonzero(members & in_removed)

            keep[displaced[:quota]] = False
            leftover = displaced[quota:]
            if len(displaced) < quota:
                others = np.flatnonzero(members & ~in_removed)
                centres = np.argwhere(removed_here) + 0.5
                distance = _nearest_distance(p.pos[others], centres)
                keep[others[np.argsort(distance)[: quota - len(displaced)]]] = False

            if len(leftover):
                targets = self.materials.conductive & (old_labels == island)
                if targets.any():
                    self._move_to_nearest_cell(leftover, targets)
                else:
                    keep[leftover] = False

        p.keep(keep)

    def _move_to_nearest_cell(self, indices: np.ndarray, cells: np.ndarray) -> None:
        p = self.particles
        centres = np.argwhere(cells) + 0.5
        d2 = ((p.pos[indices, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        nearest = centres[np.argmin(d2, axis=1)]
        p.pos[indices] = nearest + self._jitter(nearest.shape)
        p.vel[indices] = 0.0


def fill_pattern(n: int) -> np.ndarray:
    """n points spread evenly over a unit cell (n, 2): a rank-1 lattice.

    Columns and rows are each n evenly spaced values, so per axis the points
    deposit 3/4 on the cell centre and 1/8 on each neighbour, like the ion
    (fields.spread_over_cells): a freshly painted conductor is almost field-free.
    """
    i = np.arange(n)
    # Step through the rows by a generator coprime to n near n / golden ratio^2,
    # so successive points land far apart.
    step = min((g for g in range(1, n + 1) if np.gcd(g, n) == 1), key=lambda g: abs(g - 0.382 * n))
    return np.stack([(i + 0.5) / n, ((i * step) % n + 0.5) / n], axis=-1)


def _nearest_distance(points: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Distance from each point to the closest of `centres`."""
    d2 = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(d2.min(axis=1))
