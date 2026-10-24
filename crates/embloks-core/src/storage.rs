//! Saving and loading the complete simulation state.
//!
//! A save is a deflate-compressed postcard encoding of `SaveFile`: the arrays
//! (materials, electrons, test charges, smoothed current) plus the format version,
//! physics settings, battery gains, view layers and speed.

use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};

use crate::controller::App;
use crate::layers::Layers;
use crate::settings::Settings;
use crate::world::World;

/// 3: the Rust port (Python saves were format 2, .npz).
pub const FORMAT_VERSION: u32 = 3;
const MAGIC: &[u8; 8] = b"EMBLOKS\0";

/// Settings that shape the saved state. They are restored on load, so a scene
/// behaves the same even if the defaults have changed since it was saved.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhysicsSettings {
    pub dt: f64,
    pub cell_charge: f64,
    pub carriers_per_cell: u32,
    pub carrier_charge_to_mass: f64,
    pub drag_per_resistance: f64,
    pub battery_resistance: f64,
    pub battery_density: u32,
    pub current_smoothing_time: f64,
    pub wall_restitution: f64,
    pub max_step: f64,
    pub max_substeps: u32,
    pub coulomb_k: f64,
    pub speed_of_light: f64,
    pub electron_softening: f64,
    pub electron_cutoff: f64,
    pub battery_trim_time: f64,
    pub smoothing_passes: usize,
    pub absorbing_margin: usize,
    pub absorbing_strength: f64,
}

macro_rules! physics_fields {
    ($m:ident) => {
        $m!(dt, cell_charge, carriers_per_cell, carrier_charge_to_mass, drag_per_resistance,
            battery_resistance, battery_density, current_smoothing_time, wall_restitution,
            max_step, max_substeps, coulomb_k, speed_of_light, electron_softening,
            electron_cutoff, battery_trim_time, smoothing_passes, absorbing_margin,
            absorbing_strength)
    };
}

impl PhysicsSettings {
    pub fn of(s: &Settings) -> Self {
        macro_rules! take {
            ($($f:ident),*) => { Self { $($f: s.$f.clone()),* } };
        }
        physics_fields!(take)
    }

    pub fn apply(&self, s: &mut Settings) {
        macro_rules! put {
            ($($f:ident),*) => {{ $(s.$f = self.$f.clone();)* }};
        }
        physics_fields!(put);
    }
}

impl Default for PhysicsSettings {
    fn default() -> Self {
        Self::of(&Settings::default())
    }
}

#[derive(Serialize, Deserialize)]
struct SaveFile {
    format: u32,
    cols: usize,
    rows: usize,
    sim_time: f64,
    sim_speed: f64,
    settings: PhysicsSettings,
    // Stacks are found in a fixed order from the materials, so a list lines up on load.
    battery_gain: Vec<Option<f64>>,
    layers: Layers,
    kind: Vec<u8>,
    resistance: Vec<f64>,
    density: Vec<f64>,
    emf: Vec<f64>,
    orientation: Vec<u8>,
    frequency: Vec<f64>,
    pos: Vec<[f64; 2]>,
    vel: Vec<[f64; 2]>,
    charge: Vec<f64>,
    mass: Vec<f64>,
    fixed_pos: Vec<[f64; 2]>,
    fixed_charge: Vec<f64>,
    current: Vec<f64>,
}

fn flat<T: Clone>(array: &Array2<T>) -> Vec<T> {
    array.as_standard_layout().iter().cloned().collect()
}

/// The app's world, view and physics settings as bytes.
pub fn save(app: &App) -> Vec<u8> {
    let w = &app.sim.world;
    let (m, p) = (&w.materials, &w.particles);
    let file = SaveFile {
        format: FORMAT_VERSION,
        cols: m.cols,
        rows: m.rows,
        sim_time: app.sim.time,
        sim_speed: app.sim_speed,
        settings: PhysicsSettings::of(&app.settings),
        battery_gain: w.stacks.iter().map(|stack| w.battery_gain.get(stack).copied()).collect(),
        layers: app.layers.clone(),
        kind: flat(&m.kind),
        resistance: flat(&m.resistance),
        density: flat(&m.density),
        emf: flat(&m.emf),
        orientation: flat(&m.orientation),
        frequency: flat(&m.frequency),
        pos: p.pos.clone(),
        vel: p.vel.clone(),
        charge: p.charge.clone(),
        mass: p.mass.clone(),
        fixed_pos: w.fixed_pos.clone(),
        fixed_charge: w.fixed_charge.clone(),
        current: app.current.as_standard_layout().iter().copied().collect(),
    };
    let mut bytes = MAGIC.to_vec();
    bytes.extend(miniz_oxide::deflate::compress_to_vec(&postcard::to_allocvec(&file).unwrap(), 6));
    bytes
}

fn array<T>(cols: usize, rows: usize, values: Vec<T>, name: &str) -> Result<Array2<T>, String> {
    Array2::from_shape_vec((cols, rows), values).map_err(|_| format!("{name} has the wrong size"))
}

/// Replace the app's state with the one saved in `bytes`.
pub fn load(app: &mut App, bytes: &[u8]) -> Result<(), String> {
    let body = bytes.strip_prefix(MAGIC).ok_or("not an Embloks save")?;
    let raw = miniz_oxide::inflate::decompress_to_vec(body).map_err(|_| "corrupt save (compression)")?;
    let format: u32 = postcard::take_from_bytes::<u32>(&raw).map_err(|_| "corrupt save")?.0;
    if format != FORMAT_VERSION {
        return Err(format!("unsupported save format {format}"));
    }
    let file: SaveFile = postcard::from_bytes(&raw).map_err(|e| format!("corrupt save ({e})"))?;
    let (cols, rows) = (file.cols, file.rows);

    // Settings first: the world reads them (e.g. carrier charge) while being rebuilt.
    file.settings.apply(&mut app.settings);
    let mut world = World::new(cols, rows, app.settings.clone(), app.next_seed());
    let m = &mut world.materials;
    m.kind = array(cols, rows, file.kind, "kind")?;
    m.resistance = array(cols, rows, file.resistance, "resistance")?;
    m.density = array(cols, rows, file.density, "density")?;
    m.emf = array(cols, rows, file.emf, "emf")?;
    m.orientation = array(cols, rows, file.orientation, "orientation")?;
    m.frequency = array(cols, rows, file.frequency, "frequency")?;
    m.version += 1;
    world.update_structure();
    let p = &mut world.particles;
    let n = file.pos.len();
    if [file.vel.len(), file.charge.len(), file.mass.len()].iter().any(|&len| len != n) {
        return Err("electron arrays differ in length".into());
    }
    (p.pos, p.vel, p.charge, p.mass) = (file.pos, file.vel, file.charge, file.mass);
    (world.fixed_pos, world.fixed_charge) = (file.fixed_pos, file.fixed_charge);
    world.fixed_version += 1;
    for (stack, gain) in world.stacks.clone().into_iter().zip(file.battery_gain) {
        if let Some(gain) = gain {
            world.battery_gain.insert(stack, gain);
        }
    }
    let current = Array3::from_shape_vec((cols, rows, 2), file.current).map_err(|_| "current has the wrong size")?;

    app.install_world(world);
    app.current = current;
    app.sim.time = file.sim_time;
    app.sim_speed = file.sim_speed;
    app.layers = file.layers;
    Ok(())
}
