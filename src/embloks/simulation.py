"""One tick of physics: battery feedback, electrons, then fields."""

import numpy as np

from embloks import battery, physics
from embloks.close_range import CloseRange
from embloks.config import Settings
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

    def step(self, dt: float, time: float) -> int:
        """Advance everything by dt from simulation `time`; returns the electron substeps."""
        s, w = self.settings, self.world
        potential = self.potential(battery.probe_cells(w))
        battery.trim(w, potential, physics.current_density(w), dt, s, time)
        return physics.step(w, self.maxwell, self.close_range, dt, s, time)
