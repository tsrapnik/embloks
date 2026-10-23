import numpy as np
import pygame

from embloks import render
from embloks.config import Settings
from embloks.fields import ElectricField
from embloks.grid import Grid
from embloks.materials import Material, Orientation
from embloks.ui import Brush, Tool, line_cells
from embloks.world import World

DENSITY_UP_KEYS = {pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS}
DENSITY_DOWN_KEYS = {pygame.K_MINUS, pygame.K_KP_MINUS}
TOOL_KEYS = {
    pygame.K_1: Tool.VACUUM,
    pygame.K_2: Tool.CONDUCTOR,
    pygame.K_3: Tool.BATTERY,
    pygame.K_4: Tool.CHARGE,
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
        self.show_particles = True
        self.show_e_field = True
        self.electric = ElectricField(self.settings)
        self.brush = Brush(self.settings)
        self.material_layer = render.MaterialLayer()
        self.hovered_cell: tuple[int, int] | None = None
        self.reset_scene()

    def reset_scene(self) -> None:
        """Rebuild the grid at the current density and restart the simulation."""
        s = self.settings
        self.grid = Grid.fit(s.grid_density, s.window_width, s.window_height)
        self.world = World(self.grid.cols, self.grid.rows, s)
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
            self.world.clear()
        elif key == pygame.K_p:
            self.show_particles = not self.show_particles
        elif key == pygame.K_e:
            self.show_e_field = not self.show_e_field
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
        b = self.brush
        erasing = self._stroke_button == ERASE_BUTTON
        if b.tool == Tool.CHARGE:
            # The charge tool erases only test charges, never materials.
            if erasing:
                self.world.remove_charges(cols, rows)
            else:
                self.world.place_charges(cols, rows, b.charge)
        elif erasing:
            self.world.paint(cols, rows, Material.VACUUM)
        else:
            self.world.paint(cols, rows, b.tool.material, b.resistance, b.emf, b.orientation)
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
        self.material_layer.draw(
            self.screen, self.grid, self.world.materials, s, show_ions=self.show_particles
        )
        render.draw_grid_lines(self.screen, self.grid, s)
        e_grid = self.electric.grid(self.world)
        if self.show_e_field:
            cells = render.arrow_sample_cells(self.grid, s)
            render.draw_field_arrows(self.screen, self.grid, cells, e_grid[cells], s)
        if self.show_particles:
            render.draw_electrons(self.screen, self.grid, self.world.particles, s)
        render.draw_fixed_charges(
            self.screen, self.grid, self.world.fixed_pos, self.world.fixed_charge, s
        )
        if self.hovered_cell is not None:
            cells = self.brush.footprint(self.hovered_cell, self.grid)
            render.draw_brush_outline(self.screen, self.grid, cells, s)
        if self.show_hud:
            render.draw_hud(self.screen, self.font, self.hud_lines(e_grid), s)
        pygame.display.flip()
        self.update_caption()

    def hud_lines(self, e_grid: np.ndarray) -> list[str]:
        b = self.brush
        tool = f"Tool: {b.tool.label}"
        if b.tool == Tool.CONDUCTOR:
            tool += f"   resistance {b.resistance:g}"
        elif b.tool == Tool.BATTERY:
            tool += f"   EMF {b.emf:g}   + side {b.orientation.name.lower()}"
        elif b.tool == Tool.CHARGE:
            tool += f"   charge {b.charge:+g} e"
        lines = [
            tool,
            f"Brush radius {b.radius}   Electrons {len(self.world.particles)}"
            f"   Islands {self.world.connectivity.count}"
            f"   Test charges {len(self.world.fixed_charge)}",
            "1-4 tool   Up/Down value   R rotate   [ ] / wheel brush   C clear",
            "E field arrows   P particles   H hide",
        ]
        if self.hovered_cell is not None:
            col, row = self.hovered_cell
            lines.append(f"Cell {self.hovered_cell}: {self.describe_cell(col, row)}")
            ex, ey = e_grid[col, row]
            lines.append(f"E = ({ex:+.3g}, {ey:+.3g})   |E| = {np.hypot(ex, ey):.3g}")
        return lines

    def describe_cell(self, col: int, row: int) -> str:
        m = self.world.materials
        kind = Material(m.kind[col, row])
        fixed_cells = np.floor(self.world.fixed_pos).astype(int)
        here = (fixed_cells[:, 0] == col) & (fixed_cells[:, 1] == row)
        charge = f", test charge {self.world.fixed_charge[here][0]:+g} e" if here.any() else ""
        if kind == Material.VACUUM:
            return "vacuum" + charge
        if kind == Material.CONDUCTOR:
            text = f"conductor, resistance {m.resistance[col, row]:g}"
        else:
            side = Orientation(m.orientation[col, row]).name.lower()
            text = f"battery, EMF {m.emf[col, row]:g}, + side {side}"
        cell_c, cell_r = self.world.particles.cells()
        electrons = np.count_nonzero((cell_c == col) & (cell_r == row))
        island = self.world.connectivity.labels[col, row]
        return f"{text}, island {island}, {electrons} electron(s){charge}"

    def update_caption(self) -> None:
        state = "paused" if self.paused else "running"
        pygame.display.set_caption(
            f"Embloks | {self.clock.get_fps():.0f} fps | t={self.sim_time:.2f} | {state}"
            f" | grid {self.grid.cols}x{self.grid.rows}"
        )


def main() -> None:
    App().run()
