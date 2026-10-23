import numpy as np

from embloks.config import Settings
from embloks.connectivity import compute_connectivity
from embloks.materials import Material, Materials, Orientation
from embloks.particles import Particles

# Electrons spawn on their cell's ion, offset by at most this much (cells). On the
# ion they cancel it exactly, so a fresh conductor is field-free; the small jitter
# lets several electrons in one cell separate instead of moving as one stack.
SPAWN_JITTER = 0.02


class World:
    """Materials plus the charges they carry, kept consistent while painting.

    Every conductive cell holds one fixed ion of charge +e at its centre and,
    when neutral, `carriers_per_cell` electrons of charge -e/n each.
    """

    def __init__(self, cols: int, rows: int, settings: Settings, rng: np.random.Generator | None = None):
        self.settings = settings
        self.rng = rng or np.random.default_rng()
        self.materials = Materials(cols, rows)
        self.particles = Particles()
        self.connectivity = compute_connectivity(self.materials)
        # Fixed test charges, at most one per cell, at cell centres.
        self.fixed_pos = np.empty((0, 2))
        self.fixed_charge = np.empty(0)
        # Bumped whenever test charges change, so cached fields know to rebuild.
        self.fixed_version = 0

    @property
    def electron_charge(self) -> float:
        return -self.settings.elementary_charge / self.settings.carriers_per_cell

    @property
    def electron_mass(self) -> float:
        # Mass shrinks with charge, so charge/mass and thus the dynamics don't depend on n.
        return self.settings.electron_mass / self.settings.carriers_per_cell

    def paint(
        self,
        cols: np.ndarray,
        rows: np.ndarray,
        material: Material,
        resistance: float = 0.0,
        emf: float = 0.0,
        orientation: Orientation = Orientation.RIGHT,
    ) -> None:
        was_conductive = self.materials.conductive
        old_labels = self.connectivity.labels
        self.materials.paint(cols, rows, material, resistance, emf, orientation)
        self.connectivity = compute_connectivity(self.materials)

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
        self.connectivity = compute_connectivity(self.materials)
        self.fixed_pos = np.empty((0, 2))
        self.fixed_charge = np.empty(0)
        self.fixed_version += 1

    def place_charges(self, cols: np.ndarray, rows: np.ndarray, charge: float) -> None:
        """Put a fixed test charge at the centre of each given cell, replacing any there."""
        self.remove_charges(cols, rows)
        cells = np.unique(np.stack([cols, rows], axis=-1), axis=0)
        self.fixed_pos = np.concatenate([self.fixed_pos, cells + 0.5])
        self.fixed_charge = np.concatenate([self.fixed_charge, np.full(len(cells), charge)])
        self.fixed_version += 1

    def remove_charges(self, cols: np.ndarray, rows: np.ndarray) -> None:
        """Remove the fixed test charges in the given cells."""
        fixed_cells = np.floor(self.fixed_pos).astype(np.int64)
        n_rows = self.materials.rows
        hit = np.isin(fixed_cells[:, 0] * n_rows + fixed_cells[:, 1], cols * n_rows + rows)
        self.fixed_pos = self.fixed_pos[~hit]
        self.fixed_charge = self.fixed_charge[~hit]
        self.fixed_version += 1

    def _spawn_carriers(self, cells: np.ndarray) -> None:
        n = self.settings.carriers_per_cell
        centres = np.repeat(np.argwhere(cells), n, axis=0) + 0.5
        self.particles.add(centres + self._jitter(centres.shape), self.electron_charge, self.electron_mass)

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


def _nearest_distance(points: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Distance from each point to the closest of `centres`."""
    d2 = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(d2.min(axis=1))
