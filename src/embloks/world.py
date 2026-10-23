import numpy as np

from embloks.config import Settings
from embloks.connectivity import compute_connectivity
from embloks.materials import Material, Materials, Orientation
from embloks.particles import Particles

# Electrons spawn this far from their cell's edges.
SPAWN_MARGIN = 0.1


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

    def _spawn_carriers(self, cells: np.ndarray) -> None:
        n = self.settings.carriers_per_cell
        corners = np.repeat(np.argwhere(cells), n, axis=0).astype(float)
        offsets = self.rng.uniform(SPAWN_MARGIN, 1 - SPAWN_MARGIN, size=corners.shape)
        self.particles.add(corners + offsets, self.electron_charge, self.electron_mass)

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
        jitter = self.rng.uniform(-0.5 + SPAWN_MARGIN, 0.5 - SPAWN_MARGIN, size=nearest.shape)
        p.pos[indices] = nearest + jitter
        p.vel[indices] = 0.0


def _nearest_distance(points: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Distance from each point to the closest of `centres`."""
    d2 = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(d2.min(axis=1))
