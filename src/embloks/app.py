import pygame

from embloks.config import Settings


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
        self.sim_time = 0.0
        self.tick_count = 0
        # Real time not yet consumed by physics ticks.
        self._accumulator = 0.0

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
        pygame.display.flip()
        state = "paused" if self.paused else "running"
        pygame.display.set_caption(
            f"Embloks | {self.clock.get_fps():.0f} fps | t={self.sim_time:.2f} | {state}"
        )


def main() -> None:
    App().run()
