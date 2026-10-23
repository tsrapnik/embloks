import numpy as np
import pygame

from embloks import render
from embloks.config import Settings
from embloks.grid import Grid
from embloks.materials import Material, Materials, Orientation
from embloks.ui import Brush, line_cells

DENSITY_UP_KEYS = {pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS}
DENSITY_DOWN_KEYS = {pygame.K_MINUS, pygame.K_KP_MINUS}
TOOL_KEYS = {
    pygame.K_1: Material.VACUUM,
    pygame.K_2: Material.CONDUCTOR,
    pygame.K_3: Material.BATTERY,
}
PAINT_BUTTON = 1
ERASE_BUTTON = 3


class App:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.settings.window_width, self.settings.window_height)
        )
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, self.settings.hud_font_size)
        self.running = True
        self.paused = False
        self.show_hud = True
        self.brush = Brush(self.settings)
        self.material_layer = render.MaterialLayer()
        self.hovered_cell: tuple[int, int] | None = None
        self.reset_scene()

    def reset_scene(self) -> None:
        """Rebuild the grid at the current density and restart the simulation."""
        s = self.settings
        self.grid = Grid.fit(s.grid_density, s.window_width, s.window_height)
        self.materials = Materials(self.grid.cols, self.grid.rows)
        self.sim_time = 0.0
        self.tick_count = 0
        # Real time not yet consumed by physics ticks.
        self._accumulator = 0.0
        # Mouse button held down for the current stroke, and the last cell it painted.
        self._stroke_button: int | None = None
        self._stroke_cell: tuple[int, int] | None = None

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
                self.handle_key(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (PAINT_BUTTON, ERASE_BUTTON):
                self._stroke_button = event.button
                self._stroke_cell = None
                self.stroke_to(self.grid.cell_at(*event.pos))
            elif event.type == pygame.MOUSEMOTION and self._stroke_button is not None:
                self.stroke_to(self.grid.cell_at(*event.pos))
            elif event.type == pygame.MOUSEBUTTONUP and event.button == self._stroke_button:
                self._stroke_button = None
            elif event.type == pygame.MOUSEWHEEL:
                self.brush.resize(event.y)
        self.hovered_cell = self.grid.cell_at(*pygame.mouse.get_pos())

    def handle_key(self, key: int) -> None:
        brush = self.brush
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key in DENSITY_UP_KEYS:
            self.change_density(self.settings.grid_density_step)
        elif key in DENSITY_DOWN_KEYS:
            self.change_density(1 / self.settings.grid_density_step)
        elif key in TOOL_KEYS:
            brush.tool = TOOL_KEYS[key]
        elif key == pygame.K_RIGHTBRACKET:
            brush.resize(1)
        elif key == pygame.K_LEFTBRACKET:
            brush.resize(-1)
        elif key == pygame.K_r:
            brush.rotate()
        elif key == pygame.K_UP:
            brush.adjust_value(1)
        elif key == pygame.K_DOWN:
            brush.adjust_value(-1)
        elif key == pygame.K_c:
            self.materials.clear()
        elif key == pygame.K_h:
            self.show_hud = not self.show_hud

    def stroke_to(self, cell: tuple[int, int] | None) -> None:
        """Paint from the previous stroke cell to `cell`, leaving no gaps on fast drags."""
        if cell is None:
            # Off the grid: the next on-grid cell starts a fresh segment.
            self._stroke_cell = None
            return
        path = line_cells(self._stroke_cell or cell, cell)
        footprints = [self.brush.footprint(c, self.grid) for c in path]
        cols = np.concatenate([f[0] for f in footprints])
        rows = np.concatenate([f[1] for f in footprints])
        if self._stroke_button == ERASE_BUTTON:
            self.materials.paint(cols, rows, Material.VACUUM)
        else:
            b = self.brush
            self.materials.paint(cols, rows, b.tool, b.resistance, b.emf, b.orientation)
        self._stroke_cell = cell

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
        s = self.settings
        self.screen.fill(s.background_color)
        self.material_layer.draw(self.screen, self.grid, self.materials, s)
        render.draw_grid_lines(self.screen, self.grid, s)
        if self.hovered_cell is not None:
            cells = self.brush.footprint(self.hovered_cell, self.grid)
            render.draw_brush_outline(self.screen, self.grid, cells, s)
        if self.show_hud:
            render.draw_hud(self.screen, self.font, self.hud_lines(), s)
        pygame.display.flip()
        self.update_caption()

    def hud_lines(self) -> list[str]:
        b = self.brush
        tool = f"Tool: {b.tool.name.lower()}"
        if b.tool == Material.CONDUCTOR:
            tool += f"   resistance {b.resistance:g}"
        elif b.tool == Material.BATTERY:
            tool += f"   EMF {b.emf:g}   + side {b.orientation.name.lower()}"
        lines = [
            tool,
            f"Brush radius {b.radius}",
            "1/2/3 tool   Up/Down value   R rotate   [ ] / wheel brush   C clear   H hide",
        ]
        if self.hovered_cell is not None:
            lines.append(f"Cell {self.hovered_cell}: {self.describe_cell(*self.hovered_cell)}")
        return lines

    def describe_cell(self, col: int, row: int) -> str:
        m = self.materials
        kind = Material(m.kind[col, row])
        if kind == Material.CONDUCTOR:
            return f"conductor, resistance {m.resistance[col, row]:g}"
        if kind == Material.BATTERY:
            side = Orientation(m.orientation[col, row]).name.lower()
            return f"battery, EMF {m.emf[col, row]:g}, + side {side}"
        return "vacuum"

    def update_caption(self) -> None:
        state = "paused" if self.paused else "running"
        pygame.display.set_caption(
            f"Embloks | {self.clock.get_fps():.0f} fps | t={self.sim_time:.2f} | {state}"
            f" | grid {self.grid.cols}x{self.grid.rows}"
        )


def main() -> None:
    App().run()
