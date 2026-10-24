//! The app's state and behaviour, independent of the platform: keys, painting
//! strokes, the fixed-timestep loop, smoothed display fields and the info panel's
//! content. The web front end feeds it events and draws what it holds.

use std::collections::HashMap;
use std::rc::Rc;

use ndarray::{Array2, Array3};

use crate::battery;
use crate::brush::{Brush, Tool, line_cells};
use crate::fields::poynting;
use crate::format::{describe_b, describe_frequency, number, plural};
use crate::grid::Grid;
use crate::layers::{LayerId, Layers};
use crate::materials::{Cell, Material, Orientation, PaintProps};
use crate::physics;
use crate::scenes::SCENES;
use crate::settings::Settings;
use crate::simulation::Simulation;
use crate::storage::PhysicsSettings;
use crate::world::World;

/// Info panel messages wrap at this many characters.
pub const MESSAGE_WIDTH: usize = 48;

pub const KEY_HELP: [(&str, &str); 16] = [
    ("Left drag", "paint with tool"),
    ("Right drag", "erase: test charges if one is under the brush, else material"),
    ("1  2  3", "tool: conductor / battery / test charge"),
    ("Up  Down", "tool value: resistance / EMF / charge"),
    ("R", "rotate battery"),
    ("F", "battery frequency: DC or AC"),
    ("[  ]  wheel", "brush size"),
    ("E  B  S  P", "show E arrows / B colours / S arrows / particles"),
    ("L", "layers panel (top right): show, scale, opacity, log/linear"),
    ("Space", "pause"),
    ("<  >", "slower / faster simulation (the , and . keys)"),
    ("+  -", "grid density (clears)"),
    ("C", "clear all"),
    ("Ctrl+Shift / Shift + 0-9", "save to / load from slot"),
    ("F1 - F7", "examples: battery + resistor / screening / series / capacitor / parallel / transformer / antenna"),
    ("H", "panel: full / status only / hidden"),
];

/// Seconds since some fixed moment; real time for the readout filters and messages.
pub type Clock = Rc<dyn Fn() -> f64>;

pub fn system_clock() -> Clock {
    let start = web_time::Instant::now();
    Rc::new(move || start.elapsed().as_secs_f64())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HudMode {
    Full,
    Status,
    Hidden,
}

impl HudMode {
    fn next(self) -> Self {
        match self {
            HudMode::Full => HudMode::Status,
            HudMode::Status => HudMode::Hidden,
            HudMode::Hidden => HudMode::Full,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Key {
    /// A printable key, lower case.
    Char(char),
    Digit(u8),
    /// F1, F2, ...
    Function(u8),
    Up,
    Down,
    Space,
    Escape,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mods {
    pub ctrl: bool,
    pub shift: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Button {
    Paint,
    Erase,
}

/// Work a key asks of the platform (saves live in browser storage, which is async).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Effect {
    SaveSlot(u8),
    LoadSlot(u8),
}

/// A value in the info panel: one string, or fields laid out in fixed columns.
#[derive(Clone, Debug, PartialEq)]
pub enum HudValue {
    Text(String),
    Fields(Vec<String>),
}

pub type HudSection = Vec<(String, HudValue)>;

/// World-wide numbers for the status section.
#[derive(Clone, Debug, Default)]
pub struct StatusReading {
    pub mean_e_in_wires: Option<f64>,
}

/// Numbers for the cell under the mouse.
#[derive(Clone, Debug, Default)]
pub struct CellReading {
    pub electrons: usize,
    pub potential: f64,
    pub e: [f64; 2],
    pub b: f64,
    pub s: [f64; 2],
    /// Conductive cells: the smoothed current here, and summed across the wire.
    pub current: Option<[f64; 2]>,
    pub wire_current: Option<(f64, usize)>,
    /// Battery cells: their stack's readings.
    pub stack: Option<StackReading>,
}

#[derive(Clone, Debug, Default)]
pub struct StackReading {
    pub emf: f64,
    pub is_ac: bool,
    /// Measured from the stack's own + terminal, so it comes out positive; None
    /// unless both terminals are wired.
    pub voltage: Option<f64>,
    pub current: f64,
    pub gain: f64,
}

/// Exponential low-pass filter for numbers shown in the info panel.
///
/// Works in real time, so readings settle at a readable pace at any simulation
/// speed. Call begin() once per frame with a key describing what is being
/// measured; when the key changes, all readings restart from their exact value.
pub struct ReadoutFilter {
    pub seconds: f64,
    pub clock: Clock,
    pub values: HashMap<String, Vec<f64>>,
    key: Option<String>,
    last_time: Option<f64>,
    alpha: f64,
}

impl ReadoutFilter {
    pub fn new(seconds: f64, clock: Clock) -> Self {
        Self { seconds, clock, values: HashMap::new(), key: None, last_time: None, alpha: 1.0 }
    }

    pub fn begin(&mut self, key: String) {
        let now = (self.clock)();
        match self.last_time {
            Some(last) if self.key.as_ref() == Some(&key) && self.seconds > 0.0 => {
                self.alpha = 1.0 - (-(now - last) / self.seconds).exp();
            }
            _ => {
                self.values.clear();
                self.alpha = 1.0;
            }
        }
        self.key = Some(key);
        self.last_time = Some(now);
    }

    pub fn filter(&mut self, name: &str, value: &[f64]) -> Vec<f64> {
        let filtered: Vec<f64> = match self.values.get(name) {
            Some(previous) => previous.iter().zip(value).map(|(p, v)| p + self.alpha * (v - p)).collect(),
            None => value.to_vec(),
        };
        self.values.insert(name.to_string(), filtered.clone());
        filtered
    }

    pub fn scalar(&mut self, name: &str, value: f64) -> f64 {
        self.filter(name, &[value])[0]
    }

    pub fn pair(&mut self, name: &str, value: [f64; 2]) -> [f64; 2] {
        let v = self.filter(name, &value);
        [v[0], v[1]]
    }
}

pub struct App {
    pub settings: Settings,
    /// Canvas size in pixels.
    pub width: i32,
    pub height: i32,
    pub paused: bool,
    pub hud_mode: HudMode,
    pub sim_speed: f64,
    pub layers: Layers,
    pub layer_panel_visible: bool,
    pub brush: Brush,
    pub hovered_cell: Option<Cell>,
    /// Short feedback (e.g. "saved to slot 2") and when it disappears (clock seconds).
    pub message: String,
    message_until: f64,
    // Low-pass filters for the info panel's numbers: world-wide and hovered-cell ones.
    pub status_readout: ReadoutFilter,
    pub cell_readout: ReadoutFilter,
    pub clock: Clock,
    pub grid: Grid,
    /// The physics: world, fields, close-range forces and batteries.
    pub sim: Simulation,
    /// Conventional current per cell, averaged over current_smoothing_time.
    pub current: Array3<f64>,
    pub tick_count: u64,
    /// Bumped whenever a new world is installed (scene, density, loaded save), so a
    /// GPU backend knows to start over rather than update.
    pub world_generation: u64,
    /// Real time not yet consumed by physics ticks.
    accumulator: f64,
    /// Mouse button held down for the current stroke, and the last cell it painted.
    stroke_button: Option<Button>,
    stroke_cell: Option<Cell>,
    /// Whether the current erase stroke removes test charges instead of materials.
    erase_charges: bool,
    /// Time-averaged E and B for display, the simulation time they were last updated,
    /// and what the static charges looked like then (painting resets the averages).
    display: Option<(Array3<f64>, Array2<f64>)>,
    display_time: f64,
    display_key: Option<(u64, u64)>,
    seed: u64,
}

impl App {
    pub fn new(settings: Settings, width: i32, height: i32) -> Self {
        Self::with_clock(settings, width, height, system_clock(), 0)
    }

    pub fn with_clock(settings: Settings, width: i32, height: i32, clock: Clock, seed: u64) -> Self {
        let grid = Grid::fit(settings.grid_density, width, height);
        let world = World::new(grid.cols, grid.rows, settings.clone(), seed);
        let mut app = Self {
            width,
            height,
            paused: false,
            hud_mode: HudMode::Full,
            sim_speed: settings.default_sim_speed,
            layers: Layers::default(),
            layer_panel_visible: true,
            brush: Brush::new(&settings),
            hovered_cell: None,
            message: String::new(),
            message_until: 0.0,
            status_readout: ReadoutFilter::new(settings.readout_smoothing_seconds, clock.clone()),
            cell_readout: ReadoutFilter::new(settings.readout_smoothing_seconds, clock.clone()),
            clock,
            grid,
            current: Array3::zeros((grid.cols, grid.rows, 2)),
            sim: Simulation::new(world.clone()),
            tick_count: 0,
            world_generation: 0,
            accumulator: 0.0,
            stroke_button: None,
            stroke_cell: None,
            erase_charges: false,
            display: None,
            display_time: 0.0,
            display_key: None,
            seed,
            settings,
        };
        app.install_world(world);
        app
    }

    pub fn world(&self) -> &World {
        &self.sim.world
    }

    pub fn world_mut(&mut self) -> &mut World {
        &mut self.sim.world
    }

    /// A fresh seed for the next world's random electron jitter.
    pub fn next_seed(&mut self) -> u64 {
        self.seed = self.seed.wrapping_add(1);
        self.seed
    }

    /// Rebuild the grid at the current density and restart the simulation.
    pub fn reset_scene(&mut self) {
        let grid = Grid::fit(self.settings.grid_density, self.width, self.height);
        let seed = self.next_seed();
        self.install_world(World::new(grid.cols, grid.rows, self.settings.clone(), seed));
    }

    /// Make `world` the current scene, sizing the grid to fit it, and restart time.
    pub fn install_world(&mut self, world: World) {
        let m = &world.materials;
        self.grid = Grid::for_shape(m.cols, m.rows, self.width, self.height);
        // Keep +/- stepping from the new size.
        self.settings.grid_density = self.grid.cols as u32;
        self.current = Array3::zeros((m.cols, m.rows, 2));
        self.tick_count = 0;
        self.world_generation += 1;
        self.accumulator = 0.0;
        self.stroke_button = None;
        self.stroke_cell = None;
        self.erase_charges = false;
        self.display = None;
        self.display_time = 0.0;
        self.display_key = None;
        self.sim = Simulation::new(world);
    }

    /// The canvas changed size: keep the scene, re-fit its grid.
    pub fn resize(&mut self, width: i32, height: i32) {
        (self.width, self.height) = (width, height);
        self.grid = Grid::for_shape(self.grid.cols, self.grid.rows, width, height);
    }

    pub fn change_density(&mut self, factor: f64) {
        let s = &mut self.settings;
        let mut new = (s.grid_density as f64 * factor).round() as i64;
        if new == s.grid_density as i64 {
            new += if factor > 1.0 { 1 } else { -1 };
        }
        let new = new.clamp(s.min_grid_density as i64, s.max_grid_density as i64) as u32;
        if new != s.grid_density {
            s.grid_density = new;
            self.reset_scene();
        }
    }

    pub fn load_scene(&mut self, index: usize) {
        let scene = &SCENES[index];
        PhysicsSettings::default().apply(&mut self.settings);
        (scene.settings)(&mut self.settings);
        let seed = self.next_seed();
        self.install_world(World::new(scene.cols, scene.rows, self.settings.clone(), seed));
        (scene.build)(self.world_mut());
        self.layers = Layers::default();
        (scene.layers)(&mut self.layers);
        self.sim_speed = scene.sim_speed;
        self.show_message(&format!("F{} {}: {}", index + 1, scene.title, scene.hint));
    }

    /// The scene as bytes for save slot `slot` (the platform stores them).
    pub fn save_slot(&mut self, slot: u8) -> Vec<u8> {
        let bytes = crate::storage::save(self);
        self.show_message(&format!("Saved slot {slot}"));
        bytes
    }

    /// Load save slot `slot` from its stored bytes, if there are any.
    pub fn load_slot(&mut self, slot: u8, bytes: Option<&[u8]>) {
        let Some(bytes) = bytes else {
            self.show_message(&format!("Slot {slot} is empty  (save with Ctrl+Shift+{slot})"));
            return;
        };
        match crate::storage::load(self, bytes) {
            Ok(()) => self.show_message(&format!("Loaded slot {slot}")),
            Err(error) => self.show_message(&format!("Could not load slot {slot}: {error}")),
        }
    }

    pub fn show_message(&mut self, text: &str) {
        self.message = text.to_string();
        self.message_until = (self.clock)() + self.settings.message_seconds;
    }

    pub fn change_speed(&mut self, steps: i32) {
        let speeds = &self.settings.sim_speeds;
        let index = (0..speeds.len())
            .min_by(|&a, &b| (speeds[a] - self.sim_speed).abs().total_cmp(&(speeds[b] - self.sim_speed).abs()))
            .unwrap() as i32;
        self.sim_speed = speeds[(index + steps).clamp(0, speeds.len() as i32 - 1) as usize];
    }

    // --- Input ----------------------------------------------------------------------

    pub fn handle_key(&mut self, key: Key, mods: Mods) -> Option<Effect> {
        match key {
            // Ctrl+Shift saves: browsers keep plain Ctrl+digit for switching tabs.
            Key::Digit(slot) if mods.ctrl => return Some(Effect::SaveSlot(slot)),
            Key::Digit(slot) if mods.shift => return Some(Effect::LoadSlot(slot)),
            Key::Function(n) if (1..=SCENES.len() as u8).contains(&n) => self.load_scene(n as usize - 1),
            Key::Escape => self.cancel_stroke(),
            Key::Space => self.paused = !self.paused,
            Key::Char('+' | '=') => self.change_density(self.settings.grid_density_step),
            Key::Char('-') => self.change_density(1.0 / self.settings.grid_density_step),
            Key::Digit(1) => self.brush.tool = Tool::Conductor,
            Key::Digit(2) => self.brush.tool = Tool::Battery,
            Key::Digit(3) => self.brush.tool = Tool::Charge,
            Key::Char(']') => self.brush.resize(1),
            Key::Char('[') => self.brush.resize(-1),
            Key::Char('r') => self.brush.rotate(),
            Key::Up => self.brush.adjust_value(1),
            Key::Down => self.brush.adjust_value(-1),
            Key::Char('c') => self.world_mut().clear(),
            Key::Char('e') => self.toggle_layer(LayerId::E),
            Key::Char('b') => self.toggle_layer(LayerId::B),
            Key::Char('s') => self.toggle_layer(LayerId::S),
            Key::Char('p') => self.toggle_layer(LayerId::P),
            Key::Char('f') => self.brush.step_frequency(),
            Key::Char('l') => self.layer_panel_visible = !self.layer_panel_visible,
            Key::Char(',' | '<') => self.change_speed(-1),
            Key::Char('.' | '>') => self.change_speed(1),
            Key::Char('h') => self.hud_mode = self.hud_mode.next(),
            _ => {}
        }
        None
    }

    fn toggle_layer(&mut self, id: LayerId) {
        let layer = &mut self.layers[id];
        layer.visible = !layer.visible;
    }

    /// A button went down at canvas pixel (x, y).
    pub fn pointer_down(&mut self, button: Button, x: f64, y: f64) {
        self.start_stroke(button, self.grid.cell_at(x, y));
    }

    /// The pointer moved to (x, y): hover, and paint if a stroke is going on.
    pub fn pointer_move(&mut self, x: f64, y: f64) {
        let cell = self.grid.cell_at(x, y);
        self.hovered_cell = cell;
        if self.stroke_button.is_some() {
            self.stroke_to(cell);
        }
    }

    pub fn pointer_up(&mut self, button: Button) {
        if self.stroke_button == Some(button) {
            self.stroke_button = None;
        }
    }

    pub fn pointer_leave(&mut self) {
        self.hovered_cell = None;
    }

    pub fn wheel(&mut self, steps: i32) {
        self.brush.resize(steps);
    }

    pub fn cancel_stroke(&mut self) {
        self.stroke_button = None;
        self.stroke_cell = None;
    }

    pub fn stroking(&self) -> bool {
        self.stroke_button.is_some()
    }

    /// Begin painting or erasing.
    ///
    /// An erase stroke that starts with a test charge under the brush removes only
    /// test charges, for the whole drag; otherwise it erases materials. Deciding once
    /// per stroke keeps a drag from removing a charge and then the wire beneath it.
    pub fn start_stroke(&mut self, button: Button, cell: Option<Cell>) {
        self.stroke_button = Some(button);
        self.stroke_cell = None;
        self.erase_charges = false;
        if let (Button::Erase, Some(cell)) = (button, cell) {
            let cells = self.brush.footprint(cell, self.grid.cols, self.grid.rows);
            self.erase_charges = self.world().charges_in(&cells).iter().any(|&hit| hit);
        }
        self.stroke_to(cell);
    }

    /// Paint from the previous stroke cell to `cell`, leaving no gaps on fast drags.
    pub fn stroke_to(&mut self, cell: Option<Cell>) {
        let Some(cell) = cell else {
            // Off the grid: the next on-grid cell starts a fresh segment.
            self.stroke_cell = None;
            return;
        };
        let (cols, rows) = (self.grid.cols, self.grid.rows);
        let cells: Vec<Cell> = line_cells(self.stroke_cell.unwrap_or(cell), cell)
            .into_iter()
            .flat_map(|c| self.brush.footprint(c, cols, rows))
            .collect();
        let b = self.brush.clone();
        let erase_charges = self.erase_charges;
        if self.stroke_button == Some(Button::Erase) {
            let world = self.world_mut();
            if erase_charges {
                world.remove_charges(&cells);
            } else {
                world.paint(&cells, Material::Vacuum, PaintProps::default());
            }
        } else if let Some(material) = b.tool.material() {
            let props = PaintProps {
                resistance: b.resistance(),
                emf: b.emf(),
                orientation: b.orientation,
                frequency: b.frequency(),
                density: 1.0,
            };
            self.world_mut().paint(&cells, material, props);
        } else {
            self.world_mut().place_charges(&cells, b.charge());
        }
        self.stroke_cell = Some(cell);
    }

    // --- Time -----------------------------------------------------------------------

    /// How many fixed-size physics ticks fit in the elapsed (sped-up) time.
    pub fn take_ticks(&mut self, frame_seconds: f64) -> usize {
        if self.paused {
            self.accumulator = 0.0;
            return 0;
        }
        let dt = self.settings.dt;
        let max = self.settings.max_ticks_per_frame as usize;
        self.accumulator += frame_seconds * self.sim_speed;
        let mut ticks = 0;
        while self.accumulator >= dt && ticks < max {
            self.accumulator -= dt;
            ticks += 1;
        }
        if ticks == max {
            // Falling behind: drop the backlog instead of trying to catch up.
            self.accumulator = 0.0;
        }
        ticks
    }

    /// Run as many physics ticks as fit in the elapsed (sped-up) time.
    pub fn advance(&mut self, frame_seconds: f64) -> usize {
        let ticks = self.take_ticks(frame_seconds);
        for _ in 0..ticks {
            self.step();
        }
        ticks
    }

    pub fn step(&mut self) {
        self.sim.step();
        let alpha = (self.settings.dt / self.settings.current_smoothing_time).min(1.0);
        let fresh = physics::current_density(self.world());
        self.current.zip_mut_with(&fresh, |c, f| *c += alpha * (f - *c));
        self.tick_count += 1;
    }

    pub fn sim_time(&self) -> f64 {
        self.sim.time
    }

    /// Electric potential at the cell centres (see Simulation::potential).
    pub fn potential_grid(&mut self, where_: Option<&Array2<bool>>) -> Array2<f64> {
        self.sim.potential(where_)
    }

    /// E and B on the grid, averaged over display_smoothing_time of simulated time.
    ///
    /// The average advances with simulated time, so it holds still while paused.
    /// It restarts from the instantaneous fields whenever charges are painted, so
    /// edits show up immediately even while paused.
    pub fn display_fields(&mut self) -> (Array3<f64>, Array2<f64>) {
        self.sim.sync();
        let (e, b) = self.sim.maxwell.node_fields();
        let key = self.world().scene_key();
        let tau = self.settings.display_smoothing_time;
        let time = self.sim.time;
        match &mut self.display {
            Some((de, db)) if self.display_key == Some(key) && tau > 0.0 => {
                let alpha = 1.0 - (-(time - self.display_time) / tau).exp();
                de.zip_mut_with(&e, |d, v| *d += alpha * (v - *d));
                db.zip_mut_with(&b, |d, v| *d += alpha * (v - *d));
            }
            _ => self.display = Some((e, b)),
        }
        self.display_key = Some(key);
        self.display_time = time;
        self.display.clone().unwrap()
    }

    // --- Readings -------------------------------------------------------------------

    /// Total current through the wire's cross-section at a cell, like a clamp meter.
    ///
    /// Takes the direction the current mostly flows in at this cell, then sums that
    /// component over the unbroken run of conductor cells across it.
    pub fn wire_current(&self, col: usize, row: usize) -> (f64, usize) {
        let m = &self.world().materials;
        let along = if self.current[(col, row, 0)].abs() >= self.current[(col, row, 1)].abs() { 0 } else { 1 };
        let across: (i64, i64) = if along == 0 { (0, 1) } else { (1, 0) };
        let mut total = self.current[(col, row, along)];
        let mut width = 1;
        for sign in [1, -1] {
            let (mut c, mut r) = (col as i64 + sign * across.0, row as i64 + sign * across.1);
            while m.in_bounds(c, r) && m.is_conductive((c as usize, r as usize)) {
                total += self.current[(c as usize, r as usize, along)];
                width += 1;
                (c, r) = (c + sign * across.0, r + sign * across.1);
            }
        }
        (total, width)
    }

    /// The status numbers, from the CPU simulation.
    pub fn status_reading(&self, e_grid: &Array3<f64>) -> StatusReading {
        let m = &self.world().materials;
        let wires: Vec<f64> = m
            .kind
            .indexed_iter()
            .filter(|&(_, &k)| k == Material::Conductor as u8)
            .map(|((c, r), _)| e_grid[(c, r, 0)].hypot(e_grid[(c, r, 1)]))
            .collect();
        let mean_e_in_wires = (!wires.is_empty()).then(|| wires.iter().sum::<f64>() / wires.len() as f64);
        StatusReading { mean_e_in_wires }
    }

    /// The hovered cell's numbers, from the CPU simulation.
    pub fn cell_reading(&mut self, (col, row): Cell, e: &Array3<f64>, b: &Array2<f64>, s: &Array3<f64>) -> CellReading {
        let mut probes = battery::probe_cells(self.world());
        probes[(col, row)] = true;
        let potential = self.potential_grid(Some(&probes));
        let w = self.world();
        let electrons = (0..w.particles.len()).filter(|&i| w.particles.cell(i) == (col as i64, row as i64)).count();
        let conductive = w.materials.is_conductive((col, row));
        let stack = (w.materials.material((col, row)) == Material::Battery)
            .then(|| w.stacks.iter().find(|st| st.cells.contains(&(col, row))).cloned())
            .flatten()
            .map(|stack| {
                let low = battery::terminal_potential(w, &potential, stack.low);
                let high = battery::terminal_potential(w, &potential, stack.high);
                // Report from the stack's own + terminal, so both numbers come out positive.
                let sign = if stack.emf >= 0.0 { 1.0 } else { -1.0 };
                StackReading {
                    emf: stack.emf.abs(),
                    is_ac: stack.is_ac(),
                    voltage: low.zip(high).map(|(l, h)| sign * (h - l)),
                    current: sign * battery::stack_current(w, &stack, &self.current),
                    gain: w.battery_gain.get(&stack).copied().unwrap_or(1.0),
                }
            });
        CellReading {
            electrons,
            potential: potential[(col, row)],
            e: [e[(col, row, 0)], e[(col, row, 1)]],
            b: b[(col, row)],
            s: [s[(col, row, 0)], s[(col, row, 1)]],
            current: conductive.then(|| [self.current[(col, row, 0)], self.current[(col, row, 1)]]),
            wire_current: conductive.then(|| self.wire_current(col, row)),
            stack,
        }
    }

    /// Info panel content from the CPU simulation (see hud_sections).
    pub fn hud(&mut self) -> Vec<HudSection> {
        let (e, b) = self.display_fields();
        let s = poynting(&e, &b, &self.settings);
        let status = self.status_reading(&e);
        let cell = self.hovered_cell.map(|cell| (cell, self.cell_reading(cell, &e, &b, &s)));
        self.hud_sections(&status, cell.as_ref().map(|(c, r)| (*c, r)))
    }

    // --- Info panel -----------------------------------------------------------------

    /// Info panel content: sections of (label, value) rows.
    pub fn hud_sections(&mut self, status: &StatusReading, cell: Option<(Cell, &CellReading)>) -> Vec<HudSection> {
        // Readings restart from the exact value when the scene is edited or the mouse
        // moves to another cell, so nothing slides over from a stale value.
        let scene_key = format!("{:?}", self.world().scene_key());
        self.status_readout.begin(scene_key.clone());
        self.cell_readout.begin(format!("{scene_key} {:?}", cell.map(|(c, _)| c)));
        let mut sections = vec![self.status_rows(status)];
        if let Some((cell, reading)) = cell {
            sections.push(self.cell_rows(cell, reading));
        }
        if self.hud_mode == HudMode::Full {
            sections.push(KEY_HELP.iter().map(|(k, v)| (k.to_string(), HudValue::Text(v.to_string()))).collect());
        }
        sections
    }

    fn status_rows(&mut self, status: &StatusReading) -> HudSection {
        let b = &self.brush;
        let mut tool = b.tool.label().to_string();
        match b.tool {
            Tool::Conductor => tool += &format!("  ·  resistance {}", b.resistance()),
            Tool::Battery => {
                tool += &format!("  ·  EMF {}  ·  {}", b.emf(), describe_frequency(b.frequency()));
                tool += &format!("  ·  + side {}", b.orientation.name());
            }
            Tool::Charge => tool += &format!("  ·  {:+} e", b.charge()),
        }
        let mut rows = vec![];
        if !self.message.is_empty() && (self.clock)() < self.message_until {
            // Wrapped, so a long message doesn't stretch the panel across the screen.
            for (i, line) in wrap(&self.message, MESSAGE_WIDTH).into_iter().enumerate() {
                rows.push((if i == 0 { "»" } else { "" }.to_string(), HudValue::Text(line)));
            }
        }
        let w = self.world();
        let paused = if self.paused { "  ·  PAUSED" } else { "" };
        let text = |s: String| HudValue::Text(s);
        rows.extend([
            ("Tool".to_string(), text(tool)),
            ("Brush".to_string(), text(format!("radius {}", b.radius))),
            ("Speed".to_string(), text(format!("{}× (simulated time per second){paused}", self.sim_speed))),
            ("Light".to_string(), text(format!("{} cells per time unit", self.settings.speed_of_light))),
            (
                "World".to_string(),
                text(format!(
                    "{}  ·  {}  ·  {}",
                    plural(w.particles.len(), "electron"),
                    plural(w.connectivity.count, "island"),
                    plural(w.fixed_charge.len(), "test charge")
                )),
            ),
        ]);
        if let Some(mean_e) = status.mean_e_in_wires {
            let mean_e = self.status_readout.scalar("E in wires", mean_e);
            rows.push(("|E| in wires".to_string(), text(format!("{mean_e:.4} (mean)"))));
        }
        rows
    }

    fn cell_rows(&mut self, (col, row): Cell, reading: &CellReading) -> HudSection {
        let w = &self.sim.world;
        let m = &w.materials;
        let electrons = plural(reading.electrons, "electron");
        let mut what = match m.material((col, row)) {
            Material::Conductor => format!(
                "conductor  ·  resistance {}  ·  island {}  ·  {electrons}",
                m.resistance[(col, row)],
                w.connectivity.labels[(col, row)]
            ),
            Material::Battery => format!(
                "battery  ·  EMF {}  ·  {}  ·  + side {}  ·  {electrons}",
                m.emf[(col, row)],
                describe_frequency(m.frequency[(col, row)]),
                Orientation::from_u8(m.orientation[(col, row)]).name()
            ),
            Material::Vacuum => "vacuum".to_string(),
        };
        let here = (col as f64 + 0.5, row as f64 + 0.5);
        if let Some(i) = w.fixed_pos.iter().position(|p| (p[0], p[1]) == here) {
            what += &format!("  ·  test charge {:+} e", w.fixed_charge[i]);
        }

        let smooth = &mut self.cell_readout;
        let [ex, ey] = smooth.pair("E", reading.e);
        let [sx, sy] = smooth.pair("S", reading.s);
        let fields = |v: Vec<String>| HudValue::Fields(v);
        // Coordinates go in the value, not the label: a changing label width would shift
        // every value in the panel.
        let mut rows = vec![("Cell".to_string(), HudValue::Text(format!("{col}, {row}  ·  {what}")))];
        if let Some(stack) = &reading.stack {
            rows.push(("Stack".to_string(), HudValue::Text(describe_stack(stack, smooth))));
        }
        // Fixed columns (x, y, |magnitude|, note) so numbers never shift as they change.
        let bz = smooth.scalar("B", reading.b);
        let v = smooth.scalar("V", reading.potential);
        rows.extend([
            ("V".to_string(), fields(vec![number(v, true)])),
            ("E".to_string(), fields(vec![number(ex, true), number(ey, true), "|E|".into(), number(ex.hypot(ey), false)])),
            (
                "B".to_string(),
                fields(vec![String::new(), String::new(), "|B|".into(), number(bz.abs(), false), describe_b(bz).into()]),
            ),
            ("S".to_string(), fields(vec![number(sx, true), number(sy, true), "|S|".into(), number(sx.hypot(sy), false)])),
        ]);
        if let (Some(current), Some((total, width))) = (reading.current, reading.wire_current) {
            let [ix, iy] = smooth.pair("I", current);
            rows.push((
                "I".to_string(),
                fields(vec![number(ix, true), number(iy, true), "|I|".into(), number(ix.hypot(iy), false), "this cell".into()]),
            ));
            let total = smooth.scalar("wire I", total);
            rows.push((
                "Wire I".to_string(),
                fields(vec![
                    String::new(),
                    String::new(),
                    "|I|".into(),
                    number(total.abs(), false),
                    format!("across {} of wire", plural(width, "cell")),
                ]),
            ));
        }
        rows
    }

    /// Title line: frame rate, simulated time, state and grid size.
    pub fn caption(&self, fps: f64) -> String {
        let state = if self.paused { "paused" } else { "running" };
        format!(
            "Embloks | {fps:.0} fps | t={:.2} | {state} | grid {}x{}",
            self.sim.time, self.grid.cols, self.grid.rows
        )
    }
}

fn describe_stack(stack: &StackReading, smooth: &mut ReadoutFilter) -> String {
    let kind = if stack.is_ac { "AC amplitude" } else { "EMF" };
    let Some(voltage) = stack.voltage else {
        return format!("{kind} {}  ·  not connected", stack.emf);
    };
    let voltage = smooth.scalar("terminal V", voltage);
    let current = smooth.scalar("battery I", stack.current);
    format!(
        "{kind} {}  ·  terminals {} V  ·  current {}  ·  gain {:.2}",
        stack.emf,
        number(voltage, true),
        number(current, true),
        stack.gain
    )
}

/// Greedy word wrap at `width` characters (like Python's textwrap.wrap).
pub fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines: Vec<String> = vec![];
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && line.chars().count() + 1 + word.chars().count() > width {
            lines.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}
