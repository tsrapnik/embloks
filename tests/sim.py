"""A world-level simulation loop for tests: what App.step does, without the window."""

from embloks.simulation import Simulation


class Sim(Simulation):
    def __init__(self, world):
        super().__init__(world, world.settings)
        self.time = 0.0

    def e_field(self):
        return self.maxwell.node_fields()[0]

    def step(self) -> int:
        substeps = super().step(self.settings.dt, self.time)
        self.time += self.settings.dt
        return substeps

    def run(self, seconds: float) -> "Sim":
        for _ in range(int(round(seconds / self.settings.dt))):
            self.step()
        return self
