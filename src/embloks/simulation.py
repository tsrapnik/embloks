"""One tick of physics: batteries, electrons and fields, in that order."""

import numpy as np

from embloks import battery, physics
from embloks.close_range import CloseRange
from embloks.config import Settings
from embloks.fields import CloudInCell
from embloks.maxwell import Maxwell


class Simulation:
    """The physics of a World: the fields, close-range forces and batteries.

    The fields start from rest: only the static field of the charges present.
    """

    def __init__(self, world, settings: Settings):
        self.world = world
        self.settings = settings
        self.maxwell = Maxwell.for_world(world, settings)
        self.close_range = CloseRange(settings)
        self._fields_key = self._scene_key()
        self._stiffness_key = None
        self._stiffness: dict = {}

    def _scene_key(self) -> tuple:
        return (self.world.materials.version, self.world.fixed_version)

    def sync(self) -> None:
        """Bring E and the potential up to date after painting, even while paused.

        Ticks keep Gauss's law themselves; this covers charge that appeared or
        vanished since (painted conductors, test charges).
        """
        if self._scene_key() != self._fields_key:
            self.maxwell.enforce_gauss(self.maxwell.sources(self.world)[0])
            self._fields_key = self._scene_key()

    def potential(self, where: np.ndarray | None = None) -> np.ndarray:
        """Electric potential at the cell centres, close-range part included.

        The close-range part is only added on the cells in `where`, if given
        (see CloseRange.potential).
        """
        self.sync()
        potential = self.maxwell.visible_potential()
        if self.close_range.enabled:
            potential = potential + self.close_range.potential(self.world, where)
        return potential

    def _potential_of(self, pos: np.ndarray, charge: np.ndarray) -> np.ndarray:
        """Potential (cols, rows) of the given point charges alone."""
        m = self.world.materials
        potential = self.maxwell.static_potential(CloudInCell(pos, m.cols, m.rows).deposit(charge))
        if self.close_range.enabled:
            potential = potential + self.close_range.charges_potential(m, pos, charge)
        return potential

    def stiffness(self) -> dict:
        """The batteries' terminal stiffness (battery.terminal_stiffness), until repainted."""
        s, m = self.settings, self.world.materials
        key = (m.version, s.smoothing_passes, s.electron_softening, s.electron_cutoff, s.coulomb_k)
        if key != self._stiffness_key:
            self._stiffness = battery.terminal_stiffness(self.world, self._potential_of)
            self._stiffness_key = key
        return self._stiffness

    def step(self, dt: float, time: float) -> int:
        """Advance everything by dt from simulation `time`; returns the electron substeps."""
        s, w = self.settings, self.world
        potential = self.potential(battery.probe_cells(w))
        battery.pump(w, potential, dt, s, time, self.stiffness())
        return physics.step(w, self.maxwell, self.close_range, dt, s)
