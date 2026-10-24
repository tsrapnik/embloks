//! Golden comparisons against the original Python implementation (fixtures dumped
//! from it before it was removed: see tests/fixtures/*.json).

use embloks_core::close_range::{self, CloseRange};
use embloks_core::fields::{CloudInCell, spread_over_cells};
use embloks_core::materials::Materials;
use embloks_core::maxwell::Maxwell;
use embloks_core::physics;
use embloks_core::settings::Settings;
use embloks_core::simulation::Simulation;
use embloks_core::world::World;
use ndarray::{Array2, Array3, ArrayD, IxDyn};
use serde_json::Value;

fn load(name: &str) -> Value {
    let path = format!("{}/tests/fixtures/{name}.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn array(value: &Value) -> ArrayD<f64> {
    let shape: Vec<usize> = value["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let data: Vec<f64> = value["data"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    ArrayD::from_shape_vec(IxDyn(&shape), data).unwrap()
}

fn array2(value: &Value) -> Array2<f64> {
    array(value).into_dimensionality().unwrap()
}

fn array3(value: &Value) -> Array3<f64> {
    array(value).into_dimensionality().unwrap()
}

fn points(value: &Value) -> Vec<[f64; 2]> {
    array2(value).rows().into_iter().map(|r| [r[0], r[1]]).collect()
}

/// Largest absolute difference, relative to the largest reference value.
fn relative_error<'a>(actual: impl IntoIterator<Item = &'a f64>, expected: impl IntoIterator<Item = &'a f64>) -> f64 {
    let (mut error, mut scale) = (0.0f64, 0.0f64);
    for (a, e) in actual.into_iter().zip(expected) {
        error = error.max((a - e).abs());
        scale = scale.max(e.abs());
    }
    error / scale.max(1e-300)
}

#[test]
fn poisson_and_gauss_match_scipy() {
    let f = load("maxwell");
    let settings = Settings { absorbing_margin: 4, ..Settings::default() };
    let mut mx = Maxwell::new(10, 8, &settings);
    assert_eq!((mx.nx, mx.ny), (f["nx"].as_u64().unwrap() as usize, f["ny"].as_u64().unwrap() as usize));
    mx.ex = array2(&f["ex0"]);
    mx.ey = array2(&f["ey0"]);
    mx.bz = array2(&f["bz0"]);
    let rho = array2(&f["rho"]);
    mx.enforce_gauss(&rho);
    let gauss = &f["after_gauss"];
    assert!(relative_error(&mx.potential, &array2(&gauss["potential"])) < 1e-12);
    assert!(relative_error(&mx.ex, &array2(&gauss["ex"])) < 1e-12);
    assert!(relative_error(&mx.ey, &array2(&gauss["ey"])) < 1e-12);

    mx.step(&rho, &array3(&f["current"]), settings.dt);
    let after = &f["after_step"];
    for (ours, name) in [(&mx.ex, "ex"), (&mx.ey, "ey"), (&mx.bz, "bz"), (&mx.potential, "potential")] {
        assert!(relative_error(ours, &array2(&after[name])) < 1e-12, "{name}");
    }
}

#[test]
fn deposit_spread_and_smoothing_match_numpy() {
    let f = load("deposit");
    let cic = CloudInCell::new(&points(&f["pos"]), 10, 8);
    let q: Vec<f64> = array(&f["q"]).iter().copied().collect();
    assert!(relative_error(&cic.deposit(&q), &array2(&f["deposit"])) < 1e-12);
    let values = array2(&f["spread_in"]);
    assert!(relative_error(&spread_over_cells(&values), &array2(&f["spread_out"])) < 1e-12);
    let mx = Maxwell::new(10, 8, &Settings { absorbing_margin: 4, ..Settings::default() });
    assert!(relative_error(&mx.smoothed2(&values), &array2(&f["smoothed"])) < 1e-12);
}

#[test]
fn close_range_tables_match_python() {
    let f = load("close_range_tables");
    let s = Settings::default();
    let (r, force) = close_range::grid_force(s.smoothing_passes, s.electron_cutoff);
    assert!(relative_error(&r, &array(&f["grid_force_r"])) < 1e-12);
    assert!(relative_error(&force, &array(&f["grid_force_f"])) < 1e-9);
    let t = close_range::tables(s.electron_softening, s.smoothing_passes, s.electron_cutoff);
    assert!(relative_error(&t.g, &array(&f["g"])) < 1e-9);
    assert!(relative_error(&t.potential, &array(&f["potential"])) < 1e-9);
    assert_eq!(t.half, f["half"].as_f64().unwrap());
    let shape: Vec<usize> = f["square_shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    assert_eq!(t.square.shape(), &shape[..]);
    let sampled = t.square.slice(ndarray::s![..;8, ..;8, ..]).to_owned();
    assert!(relative_error(&sampled, &array(&f["square_sampled"])) < 1e-9);
    assert!(relative_error(&t.ion_stencil, &array(&f["ion_stencil"])) < 1e-9);
}

/// The world exactly as Python built it (positions included, so no RNG differences).
fn world_from(state: &Value, settings: Settings) -> World {
    let (cols, rows) = (state["cols"].as_u64().unwrap() as usize, state["rows"].as_u64().unwrap() as usize);
    let mut world = World::new(cols, rows, settings, 0);
    let m: &mut Materials = &mut world.materials;
    m.kind = array2(&state["kind"]).mapv(|v| v as u8);
    m.resistance = array2(&state["resistance"]);
    m.density = array2(&state["density"]);
    m.emf = array2(&state["emf"]);
    m.orientation = array2(&state["orientation"]).mapv(|v| v as u8);
    m.frequency = array2(&state["frequency"]);
    m.version += 1;
    world.update_structure();
    assert_eq!(world.connectivity.labels.mapv(|v| v as f64), array2(&state["labels"]), "island labels");
    let p = &mut world.particles;
    p.pos = points(&state["pos"]);
    p.vel = points(&state["vel"]);
    p.charge = array(&state["charge"]).iter().copied().collect();
    p.mass = array(&state["mass"]).iter().copied().collect();
    if !state["fixed_charge"]["data"].as_array().unwrap().is_empty() {
        world.fixed_pos = points(&state["fixed_pos"]);
        world.fixed_charge = array(&state["fixed_charge"]).iter().copied().collect();
    }
    world.fixed_version += 1;
    world
}

fn check_scene(name: &str) {
    let f = load(name);
    let settings = Settings { speed_of_light: f["settings"]["speed_of_light"].as_f64().unwrap(), ..Settings::default() };
    let mut sim = Simulation::new(world_from(&f["initial"], settings));
    let potential = sim.potential(None);
    assert!(relative_error(&potential, &array2(&f["initial_potential"])) < 1e-9, "{name}: initial potential");
    let force = CloseRange::new(&sim.world.settings).force(&sim.world);
    let expected = points(&f["initial_close_range_force"]);
    assert!(relative_error(force.iter().flatten(), expected.iter().flatten()) < 1e-9, "{name}: close-range force");

    for _ in 0..f["ticks"].as_u64().unwrap() {
        sim.step();
    }
    // 30 ticks of a many-body system: differences in rounding grow, but stay small.
    let pos = points(&f["final_pos"]);
    let error = sim.world.particles.pos.iter().flatten().zip(pos.iter().flatten()).fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
    assert!(error < 1e-6, "{name}: positions differ by {error}");
    let (e, b) = sim.maxwell.node_fields();
    assert!(relative_error(&e, &array3(&f["final_e"])) < 1e-6, "{name}: E");
    assert!(relative_error(&b, &array2(&f["final_b"])) < 1e-6, "{name}: B");
    assert!(relative_error(&sim.maxwell.visible_potential(), &array2(&f["final_potential"])) < 1e-6, "{name}: V");
    assert!(relative_error(&physics::current_density(&sim.world), &array3(&f["final_current"])) < 1e-5, "{name}: I");
    let gains: Vec<Option<f64>> = sim.world.stacks.iter().map(|s| sim.world.battery_gain.get(s).copied()).collect();
    let expected: Vec<Option<f64>> = f["gains"].as_array().unwrap().iter().map(|g| g.as_f64()).collect();
    assert_eq!(gains.len(), expected.len());
    for (g, e) in gains.iter().zip(&expected) {
        assert!((g.unwrap_or(f64::NAN) - e.unwrap_or(f64::NAN)).abs() < 1e-6 || g == e, "{name}: gains {gains:?} vs {expected:?}");
    }
}

#[test]
fn scene_battery_and_resistor_matches_python() {
    check_scene("scene_f1");
}

#[test]
fn scene_screening_matches_python() {
    check_scene("scene_f2");
}

#[test]
fn scene_series_matches_python() {
    check_scene("scene_f3");
}

#[test]
fn scene_capacitor_matches_python() {
    check_scene("scene_f4");
}

#[test]
fn scene_transformer_matches_python() {
    check_scene("scene_f6");
}

#[test]
fn scene_antenna_matches_python() {
    check_scene("scene_f7");
}
