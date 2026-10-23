import pygame

from embloks import render
from embloks.config import Settings
from embloks.grid import Grid

DENSITY_UP_KEYS = {pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS}
DENSITY_DOWN_KEYS = {pygame.K_MINUS, pygame.K_KP_MINUS}


class App:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.settings.window_width, self.settings.window_height)
        )
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        self.hovered_cell: tuple[int, int] | None = None
        self.reset_scene()

    def reset_scene(self) -> None:
        """Rebuild the grid at the current density and restart the simulation."""
        s = self.settings
        self.grid = Grid.fit(s.grid_density, s.window_width, s.window_height)
        self.sim_time = 0.0
        self.tick_count = 0
        # Real time not yet consumed by physics ticks.
        self._accumulator = 0.0

    def change_density(self, factor: float) -> None:
        s = self.settings
        new = round(s.grid_density * factor)
        if new == s.grid_density:
            new += 1 if factor > 1 else -1
        new = max(s.min_grid_density, min(s.max_grid_density, new))
        if new != s.grid_density:
            s.grid_density = new
            self.reset_scene()

    def run(self) -> None:
        while self.running:
            frame_seconds = self.clock.tick(self.settings.fps) / 1000
            self.handle_events()
            self.advance(frame_seconds)
            self.render()
        pygame.quit()

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key in DENSITY_UP_KEYS:
                    self.change_density(self.settings.grid_density_step)
                elif event.key in DENSITY_DOWN_KEYS:
                    self.change_density(1 / self.settings.grid_density_step)
        self.hovered_cell = self.grid.cell_at(*pygame.mouse.get_pos())

    def advance(self, frame_seconds: float) -> None:
        """Run as many fixed-size physics ticks as fit in the elapsed real time."""
        if self.paused:
            self._accumulator = 0.0
            return
        dt = self.settings.dt
        self._accumulator += frame_seconds
        ticks = 0
        while self._accumulator >= dt and ticks < self.settings.max_ticks_per_frame:
            self.step(dt)
            self._accumulator -= dt
            ticks += 1
        if ticks == self.settings.max_ticks_per_frame:
            # Falling behind: drop the backlog instead of trying to catch up.
            self._accumulator = 0.0

    def step(self, dt: float) -> None:
        # Physics goes here in later steps.
        self.sim_time += dt
        self.tick_count += 1

    def render(self) -> None:
        self.screen.fill(self.settings.background_color)
        render.draw_grid_lines(self.screen, self.grid, self.settings)
        render.draw_hover(self.screen, self.grid, self.hovered_cell, self.settings)
        pygame.display.flip()
        self.update_caption()

    def update_caption(self) -> None:
        state = "paused" if self.paused else "running"
        cell = "-" if self.hovered_cell is None else f"{self.hovered_cell}"
        pygame.display.set_caption(
            f"Embloks | {self.clock.get_fps():.0f} fps | t={self.sim_time:.2f} | {state}"
            f" | grid {self.grid.cols}x{self.grid.rows} | cell {cell}"
        )


def main() -> None:
    App().run()
