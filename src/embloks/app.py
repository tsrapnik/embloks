import math
import textwrap
import time

import numpy as np
import pygame

from embloks import battery, physics, render, scenes, storage
from embloks.config import Settings
from embloks.fields import CloudInCell, ElectricField, MagneticField, poynting
from embloks.grid import Grid
from embloks.layers import Layer, LayerPanel
from embloks.maxwell import Maxwell
from embloks.materials import Material, Orientation
from embloks.ui import Brush, Tool, line_cells
from embloks.world import World

DENSITY_UP_KEYS = {pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS}
DENSITY_DOWN_KEYS = {pygame.K_MINUS, pygame.K_KP_MINUS}
SCENE_KEYS = {getattr(pygame, f"K_F{n + 1}"): n for n in range(len(scenes.SCENES))}
DIGIT_KEYS = {getattr(pygame, f"K_{n}"): n for n in range(10)}
TOOL_KEYS = {
    pygame.K_1: Tool.CONDUCTOR,
    pygame.K_2: Tool.BATTERY,
    pygame.K_3: Tool.CHARGE,
}
# Keys that show/hide a display layer, by layer id.
LAYER_KEYS = {pygame.K_e: "E", pygame.K_b: "B", pygame.K_s: "S", pygame.K_p: "P"}
SLOWER_KEYS = {pygame.K_COMMA, pygame.K_LESS}
FASTER_KEYS = {pygame.K_PERIOD, pygame.K_GREATER}
# Info panel messages wrap at this many characters.
MESSAGE_WIDTH = 48
# H cycles the info panel through these.
HUD_MODES = ("full", "status", "hidden")
KEY_HELP = [
    ("Left drag", "paint with tool"),
    ("Right drag", "erase: test charges if one is under the brush, else material"),
    ("1  2  3", "tool: conductor / battery / test charge"),
    ("Up  Down", "tool value: resistance / EMF / charge"),
    ("R", "rotate battery"),
    ("F", "battery frequency: DC or AC"),
    ("M", "field mode: instant / dynamic (Maxwell: induction, waves)"),
    ("[  ]  wheel", "brush size"),
    ("E  B  S  P", "show E arrows / B colours / S arrows / particles"),
    ("L", "layers panel (top right): show, scale, opacity, log/linear"),
    ("Space", "pause"),
    ("<  >", "slower / faster simulation (the , and . keys)"),
    ("+  -", "grid density (clears)"),
    ("C", "clear all"),
    ("Ctrl / Shift + 0-9", "save to / load from slot"),
    ("F1 - F7", "examples: battery + resistor / screening / series / capacitor / parallel /"
                " transformer / antenna"),
    ("H", "panel: full / status only / hidden"),
]
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
        self.hud_mode = HUD_MODES[0]
        self.sim_speed = self.settings.default_sim_speed
        self.layers = default_layers()
        self.layer_panel = LayerPanel(list(self.layers.values()), self.font, self.settings)
        self.electric = ElectricField(self.settings)
        self.magnetic = MagneticField(self.settings)
        self.brush = Brush(self.settings)
        self.material_layer = render.MaterialLayer()
        self.hovered_cell: tuple[int, int] | None = None
        # Short feedback (e.g. "saved to slot 2") and when it disappears (pygame ticks, ms).
        self.message = ""
        self._message_until = 0
        # Low-pass filters for the info panel's numbers: world-wide and hovered-cell ones.
        self.status_readout = ReadoutFilter(self.settings.readout_smoothing_seconds)
        self.cell_readout = ReadoutFilter(self.settings.readout_smoothing_seconds)
        self.reset_scene()

    def reset_scene(self) -> None:
        """Rebuild the grid at the current density and restart the simulation."""
        s = self.settings
        grid = Grid.fit(s.grid_density, s.window_width, s.window_height)
        self.install_world(World(grid.cols, grid.rows, s))

    def install_world(self, world: World) -> None:
        """Make `world` the current scene, sizing the grid to fit it, and restart time."""
        s = self.settings
        self.world = world
        self.grid = Grid.for_shape(world.materials.cols, world.materials.rows,
                                   s.window_width, s.window_height)
        # Keep +/- stepping from the new size.
        s.grid_density = self.grid.cols
        # Conventional current per cell, averaged over current_smoothing_time.
        self.current = np.zeros((self.grid.cols, self.grid.rows, 2))
        self.sim_time = 0.0
        self.tick_count = 0
        # Real time not yet consumed by physics ticks.
        self._accumulator = 0.0
        # Mouse button held down for the current stroke, and the last cell it painted.
        self._stroke_button: int | None = None
        self._stroke_cell: tuple[int, int] | None = None
        # Whether the current erase stroke removes test charges instead of materials.
        self._erase_charges = False
        # Time-averaged E and B for display, the simulation time they were last updated,
        # and what the static charges looked like then (painting resets the averages).
        self._display: dict[str, np.ndarray] = {}
        self._display_time = 0.0
        self._display_key = None
        # Time-stepped fields in dynamic mode (None: instant fields).
        self.maxwell = self.make_maxwell() if s.dynamic_fields else None

    def change_density(self, factor: float) -> None:
        s = self.settings
        new = round(s.grid_density * factor)
        if new == s.grid_density:
            new += 1 if factor > 1 else -1
        new = max(s.min_grid_density, min(s.max_grid_density, new))
        if new != s.grid_density:
            s.grid_density = new
            self.reset_scene()

    def reset_layers(self) -> None:
        """Put every display layer back to its start-up settings (same Layer objects)."""
        for key, fresh in default_layers().items():
            for name in ("visible", "gain", "opacity", "log_scale", "auto"):
                setattr(self.layers[key], name, getattr(fresh, name))

    def load_scene(self, index: int) -> None:
        scene = scenes.SCENES[index]
        scenes.load_scene(self, scene)
        self.show_message(f"F{index + 1} {scene.title}: {scene.hint}")

    def save_slot(self, slot: int) -> None:
        path = storage.slot_path(self.settings.save_dir, slot)
        storage.save(self, path)
        self.show_message(f"Saved slot {slot}  ({path})")

    def load_slot(self, slot: int) -> None:
        path = storage.slot_path(self.settings.save_dir, slot)
        if not path.exists():
            self.show_message(f"Slot {slot} is empty  (save with Ctrl+{slot})")
            return
        try:
            storage.load(self, path)
        except (OSError, ValueError, KeyError) as error:
            self.show_message(f"Could not load slot {slot}: {error}")
            return
        self.show_message(f"Loaded slot {slot}")

    def show_message(self, text: str) -> None:
        self.message = text
        self._message_until = pygame.time.get_ticks() + int(1000 * self.settings.message_seconds)

    def change_speed(self, steps: int) -> None:
        speeds = self.settings.sim_speeds
        index = min(range(len(speeds)), key=lambda i: abs(speeds[i] - self.sim_speed))
        self.sim_speed = speeds[max(0, min(len(speeds) - 1, index + steps))]

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
                self.handle_key(event.key, event.mod)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (PAINT_BUTTON, ERASE_BUTTON):
                if event.button == PAINT_BUTTON and self.layer_panel.handle_click(event.pos):
                    continue
                if not self.over_panel(event.pos):
                    self.start_stroke(event.button, self.grid.cell_at(*event.pos))
            elif event.type == pygame.MOUSEMOTION and self._stroke_button is not None:
                self.stroke_to(self.grid.cell_at(*event.pos))
            elif event.type == pygame.MOUSEBUTTONUP and event.button == self._stroke_button:
                self._stroke_button = None
            elif event.type == pygame.MOUSEWHEEL:
                self.brush.resize(event.y)
        mouse = pygame.mouse.get_pos()
        self.hovered_cell = None if self.over_panel(mouse) else self.grid.cell_at(*mouse)

    def over_panel(self, pos: tuple[int, int]) -> bool:
        rect = self.layer_panel.rect
        return self.layer_panel.visible and rect is not None and rect.collidepoint(pos)

    def handle_key(self, key: int, mod: int = 0) -> None:
        brush = self.brush
        if key in DIGIT_KEYS and mod & (pygame.KMOD_CTRL | pygame.KMOD_SHIFT):
            slot = DIGIT_KEYS[key]
            if mod & pygame.KMOD_CTRL:
                self.save_slot(slot)
            else:
                self.load_slot(slot)
        elif key in SCENE_KEYS:
            self.load_scene(SCENE_KEYS[key])
        elif key == pygame.K_ESCAPE:
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
        elif key in LAYER_KEYS:
            layer = self.layers[LAYER_KEYS[key]]
            layer.visible = not layer.visible
        elif key == pygame.K_m:
            self.toggle_dynamic_fields()
        elif key == pygame.K_f:
            brush.step_frequency()
        elif key == pygame.K_l:
            self.layer_panel.visible = not self.layer_panel.visible
        elif key in SLOWER_KEYS | FASTER_KEYS:
            self.change_speed(1 if key in FASTER_KEYS else -1)
        elif key == pygame.K_h:
            self.hud_mode = HUD_MODES[(HUD_MODES.index(self.hud_mode) + 1) % len(HUD_MODES)]

    def start_stroke(self, button: int, cell: tuple[int, int] | None) -> None:
        """Begin painting or erasing.

        An erase stroke that starts with a test charge under the brush removes only
        test charges, for the whole drag; otherwise it erases materials. Deciding once
        per stroke keeps a drag from removing a charge and then the wire beneath it.
        """
        self._stroke_button = button
        self._stroke_cell = None
        self._erase_charges = False
        if button == ERASE_BUTTON and cell is not None:
            cols, rows = self.brush.footprint(cell, self.grid)
            self._erase_charges = self.world.charges_in(cols, rows).any()
        self.stroke_to(cell)

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
        if self._stroke_button == ERASE_BUTTON:
            if self._erase_charges:
                self.world.remove_charges(cols, rows)
            else:
                self.world.paint(cols, rows, Material.VACUUM)
        elif b.tool == Tool.CHARGE:
            self.world.place_charges(cols, rows, b.charge)
        else:
            self.world.paint(
                cols, rows, b.tool.material, b.resistance, b.emf, b.orientation, b.frequency
            )
        self._stroke_cell = cell

    def advance(self, frame_seconds: float) -> None:
        """Run as many fixed-size physics ticks as fit in the elapsed (sped-up) time."""
        if self.paused:
            self._accumulator = 0.0
            return
        dt = self.settings.dt
        self._accumulator += frame_seconds * self.sim_speed
        ticks = 0
        while self._accumulator >= dt and ticks < self.settings.max_ticks_per_frame:
            self.step(dt)
            self._accumulator -= dt
            ticks += 1
        if ticks == self.settings.max_ticks_per_frame:
            # Falling behind: drop the backlog instead of trying to catch up.
            self._accumulator = 0.0

    def make_maxwell(self) -> Maxwell:
        """Dynamic fields for the current world, starting from rest (only static E)."""
        w, m = self.world, self.world.materials
        maxwell = Maxwell(m.cols, m.rows, self.settings)
        maxwell.enforce_gauss(maxwell.charge(w, CloudInCell(w.particles.pos, m.cols, m.rows)))
        return maxwell

    def toggle_dynamic_fields(self) -> None:
        s = self.settings
        s.dynamic_fields = not s.dynamic_fields
        self.maxwell = self.make_maxwell() if s.dynamic_fields else None
        self._display = {}
        if s.dynamic_fields:
            self.show_message(f"Dynamic fields: Maxwell's equations in 2D, light speed "
                              f"{s.speed_of_light:g} cells per time unit")
        else:
            self.show_message("Instant fields: Coulomb and Biot-Savart, no induction or waves")

    def potential_grid(self, where: np.ndarray | None = None) -> np.ndarray:
        """Electric potential at the cell centres in the current field mode.

        Only exact on the cells in `where`, if given (see ElectricField.potential_grid).
        """
        if self.maxwell is None:
            return self.electric.potential_grid(self.world, where)
        potential = self.maxwell.visible_potential()
        if self.electric.close_range.enabled:
            potential = potential + self.electric.close_range.potential(self.world, where)
        return potential

    def step(self, dt: float) -> None:
        s = self.settings
        probes = battery.probe_cells(self.world)
        battery.pump(self.world, self.potential_grid(probes), dt, s, self.sim_time)
        physics.step(self.world, self.electric, dt, s, self.maxwell)
        alpha = min(1.0, dt / s.current_smoothing_time)
        self.current += alpha * (physics.current_density(self.world) - self.current)
        self.sim_time += dt
        self.tick_count += 1

    def display_fields(self) -> tuple[np.ndarray, np.ndarray]:
        """E and B on the grid, averaged over display_smoothing_time of simulated time.

        The average advances with simulated time, so it holds still while paused.
        It restarts from the instantaneous fields whenever charges are painted or the
        field mode changes, so edits show up immediately even while paused.
        """
        if self.maxwell is not None:
            e, b = self.maxwell.node_fields()
        else:
            e = self.electric.grid(self.world)
            # B from the averaged current: instantaneous electron jitter would flicker.
            b = self.magnetic.grid(self.world, self.current + battery.current_grid(self.world))
        w = self.world
        key = (id(w), w.materials.version, w.fixed_version, self.maxwell is None)
        tau = self.settings.display_smoothing_time
        if not self._display or key != self._display_key or tau <= 0:
            self._display = {"E": e, "B": b}
        else:
            alpha = 1 - np.exp(-(self.sim_time - self._display_time) / tau)
            for name, value in (("E", e), ("B", b)):
                self._display[name] = self._display[name] + alpha * (value - self._display[name])
        self._display_key = key
        self._display_time = self.sim_time
        return self._display["E"], self._display["B"]

    def display_e_field(self) -> np.ndarray:
        return self.display_fields()[0]

    def render(self) -> None:
        s, layers = self.settings, self.layers
        self.screen.fill(s.background_color)
        self.material_layer.draw(
            self.screen, self.grid, self.world.materials, s, show_ions=layers["P"].visible
        )
        e_grid, b_grid = self.display_fields()
        if layers["B"].visible:
            render.draw_magnetic_field(self.screen, self.grid, b_grid, s, layers["B"])
        render.draw_grid_lines(self.screen, self.grid, s)
        s_grid = poynting(e_grid, b_grid, s)
        cells = render.arrow_sample_cells(self.grid, s)
        if layers["E"].visible:
            render.draw_field_arrows(
                self.screen, self.grid, cells, e_grid[cells], s, layers["E"],
                (s.e_field_min, s.e_field_max), (s.e_arrow_dim_color, s.e_arrow_color),
            )
        if layers["S"].visible:
            render.draw_field_arrows(
                self.screen, self.grid, cells, s_grid[cells], s, layers["S"],
                (s.s_field_min, s.s_field_max), (s.s_arrow_dim_color, s.s_arrow_color),
            )
        if layers["P"].visible:
            render.draw_electrons(self.screen, self.grid, self.world.particles, s)
        render.draw_fixed_charges(
            self.screen, self.grid, self.world.fixed_pos, self.world.fixed_charge, s
        )
        if self.hovered_cell is not None:
            cells = self.brush.footprint(self.hovered_cell, self.grid)
            render.draw_brush_outline(self.screen, self.grid, cells, s)
        if self.hud_mode != "hidden":
            render.draw_hud(self.screen, self.font, self.hud_sections(e_grid, b_grid, s_grid), s)
        self.layer_panel.draw(self.screen)
        pygame.display.flip()
        self.update_caption()

    def hud_sections(
        self, e_grid: np.ndarray, b_grid: np.ndarray, s_grid: np.ndarray
    ) -> list[list[tuple[str, str]]]:
        """Info panel content: sections of (label, value) rows."""
        # Readings restart from the exact value when the scene is edited or the mouse
        # moves to another cell, so nothing slides over from a stale value.
        w = self.world
        scene_key = (id(w), w.materials.version, w.fixed_version)
        self.status_readout.begin(scene_key)
        self.cell_readout.begin((scene_key, self.hovered_cell))
        sections = [self.status_rows(e_grid)]
        if self.hovered_cell is not None:
            sections.append(self.cell_rows(*self.hovered_cell, e_grid, b_grid, s_grid))
        if self.hud_mode == "full":
            sections.append(KEY_HELP)
        return sections

    def status_rows(self, e_grid: np.ndarray) -> list[tuple[str, str]]:
        b = self.brush
        tool = b.tool.label
        if b.tool == Tool.CONDUCTOR:
            tool += f"  ·  resistance {b.resistance:g}"
        elif b.tool == Tool.BATTERY:
            tool += f"  ·  EMF {b.emf:g}  ·  {describe_frequency(b.frequency)}"
            tool += f"  ·  + side {b.orientation.name.lower()}"
        elif b.tool == Tool.CHARGE:
            tool += f"  ·  {b.charge:+g} e"
        w = self.world
        rows = []
        if self.message and pygame.time.get_ticks() < self._message_until:
            # Wrapped, so a long message doesn't stretch the panel across the screen.
            for i, text in enumerate(textwrap.wrap(self.message, MESSAGE_WIDTH)):
                rows.append(("»" if i == 0 else "", text))
        rows += [
            ("Tool", tool),
            ("Brush", f"radius {b.radius}"),
            ("Speed", f"{self.sim_speed:g}× (simulated time per second)" + ("  ·  PAUSED" if self.paused else "")),
            ("Fields", f"dynamic: Maxwell, light speed {self.settings.speed_of_light:g}"
             if self.maxwell is not None else "instant: Coulomb + Biot-Savart"),
            (
                "World",
                f"{plural(len(w.particles), 'electron')}"
                f"  ·  {plural(w.connectivity.count, 'island')}"
                f"  ·  {plural(len(w.fixed_charge), 'test charge')}",
            ),
        ]
        conductive = w.materials.conductive
        if conductive.any():
            mean_e = self.status_readout("E in wires", np.hypot(*e_grid[conductive].T).mean())
            rows.append(("|E| in wires", f"{mean_e:.4f} (mean)"))
        return rows

    def cell_rows(
        self, col: int, row: int, e_grid: np.ndarray, b_grid: np.ndarray, s_grid: np.ndarray
    ) -> list[tuple[str, str]]:
        w, m = self.world, self.world.materials
        probes = battery.probe_cells(w)
        probes[col, row] = True
        potential = self.potential_grid(probes)
        kind = Material(m.kind[col, row])
        if kind == Material.CONDUCTOR:
            cell_c, cell_r = w.particles.cells()
            electrons = np.count_nonzero((cell_c == col) & (cell_r == row))
            what = (
                f"conductor  ·  resistance {m.resistance[col, row]:g}"
                f"  ·  island {w.connectivity.labels[col, row]}  ·  {plural(electrons, 'electron')}"
            )
        elif kind == Material.BATTERY:
            side = Orientation(m.orientation[col, row]).name.lower()
            what = (f"battery  ·  EMF {m.emf[col, row]:g}  ·  "
                    f"{describe_frequency(m.frequency[col, row])}  ·  + side {side}")
        else:
            what = "vacuum"
        fixed_cells = np.floor(w.fixed_pos).astype(int)
        here = (fixed_cells[:, 0] == col) & (fixed_cells[:, 1] == row)
        if here.any():
            what += f"  ·  test charge {w.fixed_charge[here][0]:+g} e"

        smooth = self.cell_readout
        ex, ey = smooth("E", e_grid[col, row])
        sx, sy = smooth("S", s_grid[col, row])
        # Coordinates go in the value, not the label: a changing label width would shift
        # every value in the panel.
        rows = [("Cell", f"{col}, {row}  ·  {what}")]
        if kind == Material.BATTERY:
            rows.append(("Stack", self.describe_stack((col, row), potential)))
        # Fixed columns (x, y, |magnitude|, note) so numbers never shift as they change.
        bz = smooth("B", b_grid[col, row])
        rows += [
            ("V", (number(smooth("V", potential[col, row])),)),
            ("E", (number(ex), number(ey), "|E|", number(np.hypot(ex, ey), signed=False))),
            ("B", ("", "", "|B|", number(abs(bz), signed=False), describe_b(bz))),
            ("S", (number(sx), number(sy), "|S|", number(np.hypot(sx, sy), signed=False))),
        ]
        if kind == Material.CONDUCTOR:
            ix, iy = smooth("I", self.current[col, row])
            rows.append(("I", (number(ix), number(iy), "|I|", number(np.hypot(ix, iy), signed=False),
                               "this cell")))
            total, width = self.wire_current(col, row)
            total = smooth("wire I", total)
            rows.append(("Wire I", ("", "", "|I|", number(abs(total), signed=False),
                                    f"across {plural(width, 'cell')} of wire")))
        return rows

    def wire_current(self, col: int, row: int) -> tuple[float, int]:
        """Total current through the wire's cross-section at a cell, like a clamp meter.

        Takes the direction the current mostly flows in at this cell, then sums that
        component over the unbroken run of conductor cells across it.
        """
        m = self.world.materials
        along = 0 if abs(self.current[col, row, 0]) >= abs(self.current[col, row, 1]) else 1
        across = (0, 1) if along == 0 else (1, 0)
        cells = [(col, row)]
        for sign in (1, -1):
            c, r = col + sign * across[0], row + sign * across[1]
            while 0 <= c < m.cols and 0 <= r < m.rows and m.conductive[c, r]:
                cells.append((c, r))
                c, r = c + sign * across[0], r + sign * across[1]
        cols, rows = np.array(cells).T
        return float(self.current[cols, rows, along].sum()), len(cells)

    def describe_stack(self, cell: tuple[int, int], potential: np.ndarray) -> str:
        stack = next(st for st in self.world.stacks if cell in st.cells)
        v_low = battery.terminal_potential(self.world, potential, stack.low)
        v_high = battery.terminal_potential(self.world, potential, stack.high)
        kind = "AC amplitude" if stack.is_ac else "EMF"
        if v_low is None or v_high is None:
            return f"{kind} {abs(stack.emf):g}  ·  not connected"
        # Report from the stack's own + terminal, so both numbers come out positive.
        sign = 1 if stack.emf >= 0 else -1
        voltage = self.cell_readout("terminal V", sign * (v_high - v_low))
        current = self.cell_readout("battery I", sign * self.world.pump_current.get(stack, 0.0))
        return (f"{kind} {abs(stack.emf):g}  ·  terminals {number(voltage)} V"
                f"  ·  current {number(current)}")

    def update_caption(self) -> None:
        state = "paused" if self.paused else "running"
        pygame.display.set_caption(
            f"Embloks | {self.clock.get_fps():.0f} fps | t={self.sim_time:.2f} | {state}"
            f" | grid {self.grid.cols}x{self.grid.rows}"
        )


class ReadoutFilter:
    """Exponential low-pass filter for numbers shown in the info panel.

    Works in real time, so readings settle at a readable pace at any simulation
    speed. Call begin() once per frame with a key describing what is being
    measured; when the key changes, all readings restart from their exact value.
    """

    def __init__(self, seconds: float, clock=time.perf_counter):
        self.seconds = seconds
        self.clock = clock
        self._values: dict[str, np.ndarray] = {}
        self._key = None
        self._last_time: float | None = None
        self._alpha = 1.0

    def begin(self, key) -> None:
        now = self.clock()
        if key != self._key or self._last_time is None or self.seconds <= 0:
            self._values.clear()
            self._alpha = 1.0
        else:
            self._alpha = 1 - math.exp(-(now - self._last_time) / self.seconds)
        self._key = key
        self._last_time = now

    def __call__(self, name: str, value):
        value = np.asarray(value, dtype=float)
        previous = self._values.get(name)
        filtered = value if previous is None else previous + self._alpha * (value - previous)
        self._values[name] = filtered
        return float(filtered) if filtered.ndim == 0 else filtered


def default_layers() -> dict[str, Layer]:
    return {
        "E": Layer("E field"),
        # B and S grow with the current (S with current squared), so they scale themselves;
        # a fixed scale would saturate the whole screen for strong currents. B is linear:
        # on a log scale a loop's weak return field outside looks nearly as strong as inside.
        "B": Layer("B field", auto=True, log_scale=False),
        # Off by default: Poynting arrows sit on the same cells as the E arrows.
        "S": Layer("S energy flow", visible=False, auto=True),
        "P": Layer("Particles", adjustable=False),
    }


def describe_b(bz: float) -> str:
    """Direction of B_z in words. Screen y points down, so +z points into the screen."""
    if bz == 0:
        return ""
    return "into the screen" if bz > 0 else "out of the screen"


def number(value: float, signed: bool = True) -> str:
    """A reading with a stable layout: 4 decimals from 0.001 to 1000, else scientific.

    With a fixed number of decimals the decimal point stays in place as the value
    changes (the info panel right-aligns numbers in fixed-width columns).
    """
    sign = "+" if signed else ""
    magnitude = abs(value)
    if magnitude == 0 or 1e-3 <= magnitude < 1e3:
        return f"{value:{sign}.4f}"
    return f"{value:{sign}.2e}"


def describe_frequency(frequency: float) -> str:
    return "DC" if frequency == 0 else f"AC {frequency:g} cycles/time"


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


def main() -> None:
    App().run()
