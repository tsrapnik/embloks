//! The app as a whole: input, time, display, info panel, saves and example scenes.

#![allow(clippy::needless_range_loop, clippy::type_complexity)]

use std::cell::Cell as StdCell;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::rc::Rc;

use embloks_core::controller::{App, Button, Effect, HudMode, Key, Mods, ReadoutFilter};
use embloks_core::fields::poynting;
use embloks_core::layers::LayerId;
use embloks_core::materials::{Cell, Material, Orientation, PaintProps};
use embloks_core::physics::current_density;
use embloks_core::scenes::SCENES;
use embloks_core::settings::Settings;
use embloks_core::storage;

/// A clock the test moves by hand.
fn manual_clock() -> (Rc<StdCell<f64>>, Rc<dyn Fn() -> f64>) {
    let now = Rc::new(StdCell::new(0.0));
    let read = now.clone();
    (now, Rc::new(move || read.get()))
}

fn app() -> App {
    App::with_clock(Settings::default(), 1000, 800, manual_clock().1, 0)
}

fn press(app: &mut App, key: Key) -> Option<Effect> {
    app.handle_key(key, Mods::default())
}

fn centre(app: &App, (col, row): Cell) -> (f64, f64) {
    let (x, y, w, h) = app.grid.cell_rect(col, row);
    ((x + w / 2) as f64, (y + h / 2) as f64)
}

fn down(app: &mut App, cell: Cell, button: Button) {
    let (x, y) = centre(app, cell);
    app.pointer_move(x, y);
    app.pointer_down(button, x, y);
}

fn drag(app: &mut App, cell: Cell) {
    let (x, y) = centre(app, cell);
    app.pointer_move(x, y);
}

fn click(app: &mut App, cell: Cell, button: Button) {
    down(app, cell, button);
    app.pointer_up(button);
}

fn kind(app: &App, cell: Cell) -> Material {
    app.world().materials.material(cell)
}

fn count(app: &App, material: Material) -> usize {
    app.world().materials.kind.iter().filter(|&&k| k == material as u8).count()
}

fn run(app: &mut App, time_units: f64) {
    for _ in 0..(time_units / app.settings.dt).round() as usize {
        app.step();
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn assert_rel(actual: f64, expected: f64, tolerance: f64) {
    assert!((actual - expected).abs() <= tolerance * expected.abs(), "{actual} vs {expected} (rel {tolerance})");
}

// --- Time -------------------------------------------------------------------------------

#[test]
fn fixed_timestep_runs_ticks_matching_elapsed_time() {
    let mut app = app();
    app.sim_speed = 1.0;
    app.advance(app.settings.dt * 3.5);
    assert_eq!(app.tick_count, 3);
}

#[test]
fn speed_multiplies_simulated_time_per_real_second() {
    let mut app = app();
    app.sim_speed = 4.0;
    app.advance(app.settings.dt * 3.5);
    assert_eq!(app.tick_count, 14);
}

#[test]
fn comma_and_period_step_through_speeds() {
    let mut app = app();
    let speeds = app.settings.sim_speeds.clone();
    for _ in 0..20 {
        press(&mut app, Key::Char('.'));
    }
    assert_eq!(app.sim_speed, speeds[speeds.len() - 1]);
    press(&mut app, Key::Char(','));
    assert_eq!(app.sim_speed, speeds[speeds.len() - 2]);
    for _ in 0..20 {
        press(&mut app, Key::Char('<'));
    }
    assert_eq!(app.sim_speed, speeds[0]);
}

#[test]
fn slow_frame_is_capped() {
    let mut app = app();
    app.advance(10.0);
    assert_eq!(app.tick_count, app.settings.max_ticks_per_frame as u64);
}

#[test]
fn space_pauses_physics() {
    let mut app = app();
    press(&mut app, Key::Space);
    assert!(app.paused);
    app.advance(1.0);
    assert_eq!(app.tick_count, 0);
}

#[test]
fn escape_cancels_the_stroke() {
    let mut app = app();
    down(&mut app, (2, 2), Button::Paint);
    press(&mut app, Key::Escape);
    drag(&mut app, (8, 2));
    assert_eq!(count(&app, Material::Conductor), 1);
}

#[test]
fn plus_and_minus_change_density_and_reset_scene() {
    let mut app = app();
    app.advance(app.settings.dt * 2.0);
    let cols = app.grid.cols;
    press(&mut app, Key::Char('='));
    assert!(app.grid.cols > cols);
    assert_eq!(app.sim_time(), 0.0);
    press(&mut app, Key::Char('-'));
    press(&mut app, Key::Char('-'));
    assert!(app.grid.cols < cols);
}

#[test]
fn density_is_clamped() {
    let mut app = app();
    for _ in 0..50 {
        press(&mut app, Key::Char('-'));
    }
    assert_eq!(app.settings.grid_density, app.settings.min_grid_density);
    for _ in 0..50 {
        press(&mut app, Key::Char('='));
    }
    assert_eq!(app.settings.grid_density, app.settings.max_grid_density);
}

// --- Painting ---------------------------------------------------------------------------

#[test]
fn left_drag_paints_a_connected_wire_with_current_tool() {
    let mut app = app();
    down(&mut app, (2, 2), Button::Paint);
    drag(&mut app, (8, 6));
    app.pointer_up(Button::Paint);
    // A 4-connected path from (2,2) to (8,6) has 6 + 4 + 1 cells.
    assert_eq!(count(&app, Material::Conductor), 11);
    assert!(kind(&app, (2, 2)) == Material::Conductor && kind(&app, (8, 6)) == Material::Conductor);
}

#[test]
fn motion_without_button_does_not_paint() {
    let mut app = app();
    drag(&mut app, (3, 3));
    assert_eq!(count(&app, Material::Vacuum), app.grid.cols * app.grid.rows);
}

#[test]
fn right_click_erases() {
    let mut app = app();
    click(&mut app, (4, 4), Button::Paint);
    click(&mut app, (4, 4), Button::Erase);
    assert_eq!(kind(&app, (4, 4)), Material::Vacuum);
}

#[test]
fn battery_tool_paints_with_brush_orientation() {
    let mut app = app();
    press(&mut app, Key::Digit(2));
    press(&mut app, Key::Char('r'));
    down(&mut app, (1, 1), Button::Paint);
    let m = &app.world().materials;
    assert_eq!(kind(&app, (1, 1)), Material::Battery);
    assert_eq!(m.orientation[(1, 1)], Orientation::Down as u8);
    assert_eq!(m.emf[(1, 1)], app.settings.default_emf);
}

#[test]
fn info_panel_works_for_every_kind_of_cell() {
    let mut app = app();
    for (tool, cell) in [(1, (1, 1)), (2, (2, 1)), (3, (5, 5))] {
        press(&mut app, Key::Digit(tool));
        click(&mut app, cell, Button::Paint);
    }
    for cell in [(1, 1), (2, 1), (5, 5), (9, 9)] {
        app.hovered_cell = Some(cell);
        let sections = app.hud();
        assert_eq!(sections.len(), 3);
    }
}

#[test]
fn right_click_removes_test_charge_first_then_material() {
    let mut app = app();
    click(&mut app, (3, 3), Button::Paint); // Conductor (default tool).
    press(&mut app, Key::Digit(3));
    press(&mut app, Key::Down);
    click(&mut app, (3, 3), Button::Paint);
    assert_eq!(app.world().fixed_charge, vec![app.brush.charge()]);
    assert!(app.brush.charge() < app.settings.default_test_charge);
    click(&mut app, (3, 3), Button::Erase);
    assert!(app.world().fixed_charge.is_empty());
    assert_eq!(kind(&app, (3, 3)), Material::Conductor);
    click(&mut app, (3, 3), Button::Erase);
    assert_eq!(kind(&app, (3, 3)), Material::Vacuum);
}

#[test]
fn erase_drag_starting_on_a_charge_never_erases_material() {
    let mut app = app();
    for col in 2..8 {
        click(&mut app, (col, 4), Button::Paint);
    }
    press(&mut app, Key::Digit(3));
    click(&mut app, (2, 4), Button::Paint);
    click(&mut app, (6, 4), Button::Paint);
    down(&mut app, (2, 4), Button::Erase);
    drag(&mut app, (7, 4));
    drag(&mut app, (2, 4));
    app.pointer_up(Button::Erase);
    assert!(app.world().fixed_charge.is_empty());
    assert!((2..8).all(|c| kind(&app, (c, 4)) == Material::Conductor));
}

#[test]
fn f_cycles_the_battery_frequency_and_paints_it() {
    let mut app = app();
    press(&mut app, Key::Digit(2));
    press(&mut app, Key::Char('f'));
    assert!(app.brush.frequency() == app.settings.ac_frequencies[1] && app.brush.frequency() > 0.0);
    click(&mut app, (4, 4), Button::Paint);
    assert_eq!(app.world().materials.frequency[(4, 4)], app.brush.frequency());
    for _ in 0..app.settings.ac_frequencies.len() - 1 {
        press(&mut app, Key::Char('f'));
    }
    assert_eq!(app.brush.frequency(), 0.0); // Back to DC.
}

// --- Display ----------------------------------------------------------------------------

#[test]
fn h_cycles_info_panel() {
    let mut app = app();
    let mut seen = vec![app.hud_mode];
    for _ in 0..3 {
        press(&mut app, Key::Char('h'));
        seen.push(app.hud_mode);
        app.hud();
    }
    assert_eq!(seen, vec![HudMode::Full, HudMode::Status, HudMode::Hidden, HudMode::Full]);
}

#[test]
fn s_toggles_poynting_arrows() {
    let mut app = app();
    assert!(!app.layers[LayerId::S].visible);
    press(&mut app, Key::Char('s'));
    assert!(app.layers[LayerId::S].visible);
    click(&mut app, (3, 3), Button::Paint);
    app.hovered_cell = Some((3, 3));
    app.hud();
}

#[test]
fn l_hides_layer_panel() {
    let mut app = app();
    press(&mut app, Key::Char('l'));
    assert!(!app.layer_panel_visible);
}

#[test]
fn displayed_e_field_is_smoothed_but_follows_edits_immediately() {
    let mut app = app();
    click(&mut app, (10, 10), Button::Paint);
    app.display_fields();
    app.paused = true;
    // Painting while paused (no simulated time passes) must still show up at once.
    let before = app.display_fields().0;
    press(&mut app, Key::Digit(3));
    click(&mut app, (14, 10), Button::Paint);
    let shown = app.display_fields().0;
    assert_eq!(shown, app.sim.maxwell.node_fields().0);
    assert_ne!(shown, before);
    // While running, the display lags behind a sudden change in the electrons.
    app.paused = false;
    app.step();
    let shown = app.display_fields().0;
    assert_ne!(shown, app.sim.maxwell.node_fields().0);
}

#[test]
fn readout_filter_smooths_and_restarts_on_a_new_key() {
    let (now, clock) = manual_clock();
    let mut readout = ReadoutFilter::new(0.5, clock);
    readout.begin("cell A".into());
    assert_eq!(readout.scalar("V", 1.0), 1.0); // First reading is exact.
    now.set(now.get() + 0.5); // One time constant later...
    readout.begin("cell A".into());
    assert!((readout.scalar("V", 0.0) - (-1f64).exp()).abs() < 1e-12); // ...it moved 1 - 1/e of the way.
    assert_eq!(readout.pair("E", [1.0, -2.0]), [1.0, -2.0]); // New name: exact.
    now.set(now.get() + 0.5);
    readout.begin("cell B".into()); // Moving to another cell restarts every reading.
    assert_eq!(readout.scalar("V", 5.0), 5.0);
}

#[test]
fn info_panel_numbers_are_filtered_while_hovering() {
    let (now, clock) = manual_clock();
    let mut app = App::with_clock(Settings::default(), 1000, 800, clock, 0);
    click(&mut app, (3, 3), Button::Paint);
    app.hovered_cell = Some((3, 3));
    app.hud();
    let v = |app: &App| app.cell_readout.values["V"][0];
    let first = v(&app);
    app.world_mut().place_charges(&[(4, 3)], 5.0); // Editing restarts readings...
    app.hud();
    assert_ne!(v(&app), first);
    app.world_mut().particles.pos.iter_mut().for_each(|p| *p = [p[0] + 0.2, p[1] + 0.2]); // ...but motion is smoothed.
    now.set(now.get() + 0.05);
    let before = v(&app);
    let exact = app.potential_grid(None)[(3, 3)];
    app.hud();
    assert!((v(&app) - before).abs() < (exact - before).abs());
}

#[test]
fn wire_current_sums_across_the_wire() {
    let mut app = app();
    for row in 4..7 {
        for col in 2..10 {
            click(&mut app, (col, row), Button::Paint);
        }
    }
    app.current.fill(0.0);
    for c in 2..10 {
        for r in 4..7 {
            app.current[(c, r, 0)] = 0.25; // 0.25 through each of the 3 cells across.
        }
    }
    let (total, width) = app.wire_current(5, 5);
    assert!((total - 0.75).abs() < 1e-12 && width == 3);
    for c in 2..10 {
        for r in 4..7 {
            app.current[(c, r, 0)] = 0.0;
            app.current[(c, r, 1)] = 0.1; // Flowing along y instead: cross-section is a row.
        }
    }
    let (total, width) = app.wire_current(5, 5);
    assert!((total - 0.8).abs() < 1e-12 && width == 8);
}

// --- Saves ------------------------------------------------------------------------------

fn build_circuit(app: &mut App) {
    let cells: Vec<Cell> = embloks_core::scenes::outline(5, 19, 5, 12);
    let w = app.world_mut();
    w.paint(&cells, Material::Conductor, PaintProps { resistance: 0.5, ..PaintProps::default() });
    w.paint(&[(12, 12)], Material::Battery, PaintProps { emf: 3.0, orientation: Orientation::Right, ..PaintProps::default() });
    w.place_charges(&[(30, 20)], -2.0);
    for _ in 0..60 {
        app.step();
    }
}

#[derive(Debug, PartialEq)]
struct Snapshot {
    kind: Vec<u8>,
    resistance: Vec<f64>,
    emf: Vec<f64>,
    orientation: Vec<u8>,
    pos: Vec<[f64; 2]>,
    vel: Vec<[f64; 2]>,
    charge: Vec<f64>,
    mass: Vec<f64>,
    fixed_pos: Vec<[f64; 2]>,
    fixed_charge: Vec<f64>,
    current: Vec<f64>,
    battery_gain: Vec<Option<f64>>,
    sim_time: f64,
    sim_speed: f64,
}

fn snapshot(app: &App) -> Snapshot {
    let (w, m, p) = (app.world(), &app.world().materials, &app.world().particles);
    Snapshot {
        kind: m.kind.iter().copied().collect(),
        resistance: m.resistance.iter().copied().collect(),
        emf: m.emf.iter().copied().collect(),
        orientation: m.orientation.iter().copied().collect(),
        pos: p.pos.clone(),
        vel: p.vel.clone(),
        charge: p.charge.clone(),
        mass: p.mass.clone(),
        fixed_pos: w.fixed_pos.clone(),
        fixed_charge: w.fixed_charge.clone(),
        current: app.current.iter().copied().collect(),
        battery_gain: w.stacks.iter().map(|s| w.battery_gain.get(s).copied()).collect(),
        sim_time: app.sim_time(),
        sim_speed: app.sim_speed,
    }
}

#[test]
fn round_trip_restores_everything() {
    let mut app = app();
    build_circuit(&mut app);
    app.layers[LayerId::S].visible = true;
    app.layers[LayerId::E].opacity = 0.5;
    app.layers[LayerId::B].log_scale = false;
    app.sim_speed = 16.0;
    app.settings.speed_of_light = 3.0;
    let before = snapshot(&app);
    let bytes = storage::save(&app);

    // Wreck the state, including grid size and a physics setting.
    app.change_density(2.0);
    app.world_mut().clear();
    app.layers[LayerId::S].visible = false;
    app.settings.speed_of_light = 9.0;
    app.sim_speed = 1.0;

    storage::load(&mut app, &bytes).unwrap();
    assert_eq!(snapshot(&app), before);
    assert_eq!(app.settings.speed_of_light, 3.0);
    assert!(app.layers[LayerId::S].visible && app.layers[LayerId::E].opacity == 0.5 && !app.layers[LayerId::B].log_scale);
    assert_eq!((app.grid.cols, app.grid.rows), app.world().materials.kind.dim());
    assert_eq!(app.world().stacks.len(), 1);
    // The loaded scene keeps running.
    for _ in 0..10 {
        app.step();
    }
    app.hud();
}

#[test]
fn loading_resizes_the_grid_to_the_saved_scene() {
    let mut app = app();
    build_circuit(&mut app);
    let shape = app.world().materials.kind.dim();
    let bytes = storage::save(&app);
    app.change_density(0.5);
    assert_ne!(app.world().materials.kind.dim(), shape);
    storage::load(&mut app, &bytes).unwrap();
    assert_eq!(app.world().materials.kind.dim(), shape);
    assert_eq!((app.grid.cols, app.grid.rows), shape);
}

/// Press a key, carrying out its save/load effect against in-memory slots.
fn press_with_slots(app: &mut App, key: Key, mods: Mods, slots: &mut HashMap<u8, Vec<u8>>) {
    match app.handle_key(key, mods) {
        Some(Effect::SaveSlot(slot)) => {
            let bytes = app.save_slot(slot);
            slots.insert(slot, bytes);
        }
        Some(Effect::LoadSlot(slot)) => app.load_slot(slot, slots.get(&slot).map(|b| b.as_slice())),
        None => {}
    }
}

#[test]
fn ctrl_shift_digit_saves_and_shift_digit_loads() {
    let mut app = app();
    let mut slots = HashMap::new();
    build_circuit(&mut app);
    let electrons = app.world().particles.len();
    press_with_slots(&mut app, Key::Digit(3), Mods { ctrl: true, shift: true }, &mut slots);
    assert!(slots.contains_key(&3));
    assert!(app.message.contains("Saved slot 3"));
    app.world_mut().clear();
    press_with_slots(&mut app, Key::Digit(3), Mods { ctrl: false, shift: true }, &mut slots);
    assert_eq!(app.world().particles.len(), electrons);
    assert!(app.message.contains("Loaded slot 3"));
    // Plain digits still pick tools.
    press_with_slots(&mut app, Key::Digit(2), Mods::default(), &mut slots);
    assert_eq!(app.brush.tool.label(), "battery");
}

#[test]
fn empty_or_incompatible_slots_give_a_message() {
    let mut app = app();
    let mut slots = HashMap::new();
    press_with_slots(&mut app, Key::Digit(5), Mods { ctrl: false, shift: true }, &mut slots);
    assert!(app.message.contains("empty"));
    slots.insert(6, b"EMBLOKS\0not a save".to_vec());
    press_with_slots(&mut app, Key::Digit(6), Mods { ctrl: false, shift: true }, &mut slots);
    assert!(app.message.contains("Could not load"));
}

#[test]
fn ac_frequency_and_light_speed_survive_save_and_load() {
    let mut app = app();
    app.load_scene(6); // Antenna: AC source, slow light.
    for _ in 0..10 {
        app.step();
    }
    let frequency = app.world().materials.frequency.clone();
    let bytes = storage::save(&app);
    app.load_scene(0); // DC, default light speed.
    storage::load(&mut app, &bytes).unwrap();
    assert_eq!(app.settings.speed_of_light, 2.0);
    assert_eq!(app.world().materials.frequency, frequency);
    assert!(app.world().stacks[0].is_ac());
    for _ in 0..10 {
        app.step();
    }
}

// --- Example scenes ---------------------------------------------------------------------

#[test]
fn every_scene_loads_from_its_f_key_and_runs() {
    for index in 0..SCENES.len() {
        let mut app = app();
        press(&mut app, Key::Function(index as u8 + 1));
        assert!(app.message.contains(SCENES[index].title));
        assert!(!app.world().particles.is_empty());
        run(&mut app, 1.0);
        app.hovered_cell = Some((20, 15));
        app.hud();
    }
}

#[test]
fn loading_a_scene_resets_settings_and_layers() {
    let mut app = app();
    app.settings.speed_of_light = 9.0;
    app.layers[LayerId::E].opacity = 0.1;
    app.layers[LayerId::S].visible = true;
    app.load_scene(1); // Screening: default layers except B hidden.
    assert_eq!(app.settings.speed_of_light, Settings::default().speed_of_light);
    assert!(app.layers[LayerId::E].opacity == 1.0 && !app.layers[LayerId::S].visible);
    assert!(!app.layers[LayerId::B].visible);
}

#[test]
fn scenes_keep_their_grid_whatever_the_canvas_shape() {
    let mut app = App::with_clock(Settings::default(), 1920, 1080, manual_clock().1, 0);
    app.load_scene(3);
    assert_eq!((app.grid.cols, app.grid.rows), (40, 32));
    assert!(app.grid.pixel_height() <= 1080 && app.grid.pixel_width() <= 1920);
}

fn mean_x(current: &ndarray::Array3<f64>, cols: std::ops::Range<usize>, row: usize) -> f64 {
    mean(&cols.map(|c| current[(c, row, 0)]).collect::<Vec<_>>())
}

#[test]
fn textbook_circuit_sends_energy_into_the_resistor() {
    let mut app = app();
    app.load_scene(0);
    let m = app.world().materials.clone();
    let resistor: Vec<Cell> = m
        .kind
        .indexed_iter()
        .filter(|&(cell, &k)| k == Material::Conductor as u8 && m.resistance[cell] == 5.0)
        .map(|(cell, _)| cell)
        .collect();
    let col = resistor.iter().map(|c| c.0).min().unwrap();
    let rows: Vec<usize> = resistor.iter().filter(|c| c.0 == col).map(|c| c.1).collect();
    run(&mut app, 40.0);
    let mut flow = ndarray::Array3::<f64>::zeros((m.cols, m.rows, 2));
    let mut current = vec![];
    for i in 0..(20.0 / app.settings.dt) as usize {
        app.step();
        if i % 3 == 0 {
            let (e, b) = app.sim.maxwell.node_fields();
            flow += &poynting(&e, &b, &app.settings);
        }
        current.push(mean_x(&current_density(app.world()), 13..27, 7));
    }
    // Time-averaged S points into the resistor from inside the loop. In 2D a loop is a
    // solenoid: B, and with it S, is nearly zero outside it.
    assert!(rows.iter().all(|&r| flow[(col - 1, r, 0)] > 0.0));
    // (Beyond the smoothing width of the current, ~2 cells.)
    let outside: f64 = rows.iter().map(|&r| flow[(col + 3, r, 0)].abs()).sum();
    let inside: f64 = rows.iter().map(|&r| flow[(col - 1, r, 0)]).sum();
    assert!(outside < 0.1 * inside);
    // Current around the loop follows Ohm's law (thin wires cut corners: allow 25%).
    let wire: f64 = m.kind.indexed_iter()
        .filter(|&(cell, &k)| k == Material::Conductor as u8 && m.resistance[cell] != 5.0)
        .map(|(cell, _)| m.resistance[cell])
        .sum();
    let total = wire + 5.0 * resistor.len() as f64 + app.settings.battery_resistance;
    let emf = app.world().stacks[0].emf.abs();
    assert_rel(mean(&current).abs(), emf / total, 0.25);
}

#[test]
fn series_scene_drives_three_times_the_current() {
    let mut app = app();
    app.load_scene(2);
    run(&mut app, 30.0);
    let (mut left, mut right) = (vec![], vec![]);
    for _ in 0..(30.0 / app.settings.dt) as usize {
        app.step();
        let current = current_density(app.world());
        left.push(mean_x(&current, 5..16, 8));
        right.push(mean_x(&current, 24..35, 8));
    }
    assert_rel(mean(&right) / mean(&left), 3.0, 0.2);
}

#[test]
fn magnetic_field_next_to_the_battery_is_steady() {
    // Regression for the jitter next to the battery: the displayed B holds steady.
    let mut app = app();
    app.load_scene(0);
    run(&mut app, 40.0);
    let mut samples: Vec<Vec<f64>> = vec![];
    for i in 0..(20.0 / app.settings.dt) as usize {
        app.step();
        if i % 4 == 0 {
            // One displayed frame at 8x speed.
            let (_, b) = app.display_fields();
            samples.push((11..13).flat_map(|c| (13..18).map(move |r| (c, r))).map(|cell| b[cell]).collect());
        }
    }
    let cells = samples[0].len();
    let (mut spread, mut level) = (0.0, 0.0);
    for k in 0..cells {
        let values: Vec<f64> = samples.iter().map(|s| s[k]).collect();
        let m = mean(&values);
        spread += (values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / values.len() as f64).sqrt();
        level += m.abs();
    }
    assert!(spread / level < 0.3, "flicker {}", spread / level);
}

#[test]
fn capacitor_plates_reach_the_emf_with_a_uniform_field_between() {
    let mut app = app();
    app.load_scene(3);
    run(&mut app, 60.0);
    let potential = app.potential_grid(None);
    let plate = |col: usize| mean(&(10..22).map(|r| potential[(col, r)]).collect::<Vec<_>>());
    let emf = app.world().stacks[0].emf.abs();
    assert_rel(plate(22) - plate(16), emf, 0.1);
    // Interior of the gap: away from the plates (charges are smeared ~1 cell) and edges.
    let (e, _) = app.sim.maxwell.node_fields();
    let gap: Vec<[f64; 2]> = (18..21).flat_map(|c| (12..20).map(move |r| (c, r))).map(|(c, r)| [e[(c, r, 0)], e[(c, r, 1)]]).collect();
    assert!(gap.iter().all(|v| v[0] < 0.0)); // From the + plate (right) to the - plate (left).
    let strength: Vec<f64> = gap.iter().map(|v| v[0].hypot(v[1])).collect();
    let m = mean(&strength);
    let std = (strength.iter().map(|s| (s - m).powi(2)).sum::<f64>() / strength.len() as f64).sqrt();
    assert!(std / m < 0.1); // Nearly uniform.
}

#[test]
fn parallel_paths_split_the_current_by_resistance() {
    let mut app = app();
    app.load_scene(4);
    run(&mut app, 50.0);
    // Average over every cross-section along each resistor (same current in steady
    // state) and over time: few discrete electrons pass any one line (shot noise).
    let (mut thin_a, mut thin_b, mut wide) = (vec![], vec![], vec![]);
    for _ in 0..(40.0 / app.settings.dt) as usize {
        app.step();
        for col in 13..27 {
            thin_a.push(app.wire_current(col, 10).0);
            thin_b.push(app.wire_current(col, 17).0);
            wide.push(app.wire_current(col, 22).0);
        }
    }
    let (thin_a, thin_b, wide) = (mean(&thin_a).abs(), mean(&thin_b).abs(), mean(&wide).abs());
    assert_rel(thin_a, thin_b, 0.15);
    assert_rel(wide / ((thin_a + thin_b) / 2.0), 2.0, 0.2);
}

/// Amplitude and phase (degrees) of the component of x at `frequency` (least squares
/// on sin, cos and a constant).
fn sine_fit(t: &[f64], x: &[f64], frequency: f64) -> (f64, f64) {
    let basis: Vec<[f64; 3]> = t.iter().map(|&t| {
        let w = 2.0 * PI * frequency * t;
        [w.sin(), w.cos(), 1.0]
    }).collect();
    // Normal equations (3 x 3), solved by Gaussian elimination.
    let mut a = [[0.0; 4]; 3];
    for (row, &value) in basis.iter().zip(x) {
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] += row[i] * row[j];
            }
            a[i][3] += row[i] * value;
        }
    }
    for i in 0..3 {
        for k in i + 1..3 {
            let f = a[k][i] / a[i][i];
            for j in i..4 {
                a[k][j] -= f * a[i][j];
            }
        }
    }
    let mut coef = [0.0; 3];
    for i in (0..3).rev() {
        coef[i] = (a[i][3] - (i + 1..3).map(|j| a[i][j] * coef[j]).sum::<f64>()) / a[i][i];
    }
    (coef[0].hypot(coef[1]), coef[1].atan2(coef[0]).to_degrees())
}

/// Step the app, recording time and probes of the current density each tick.
fn record(app: &mut App, seconds: f64, probes: &[&dyn Fn(&ndarray::Array3<f64>) -> f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let (mut t, mut data) = (vec![], vec![vec![]; probes.len()]);
    for _ in 0..(seconds / app.settings.dt) as usize {
        app.step();
        t.push(app.sim_time());
        let current = current_density(app.world());
        for (d, probe) in data.iter_mut().zip(probes) {
            d.push(probe(&current));
        }
    }
    (t, data)
}

#[test]
fn transformer_drives_the_inner_loop_by_induction_only() {
    let range_mean = |j: &ndarray::Array3<f64>, cells: Vec<(usize, usize)>, k: usize| {
        mean(&cells.iter().map(|&(c, r)| j[(c, r, k)]).collect::<Vec<_>>())
    };
    // Primary: mean tangential current over three sides (one loop current).
    let primary = |j: &ndarray::Array3<f64>| {
        (range_mean(j, (10..30).map(|c| (c, 4)).collect(), 0) + range_mean(j, (8..24).map(|r| (35, r)).collect(), 1)
            - range_mean(j, (10..30).map(|c| (c, 27)).collect(), 0)) / 3.0
    };
    let load = |j: &ndarray::Array3<f64>| range_mean(j, (13..19).map(|r| (27, r)).collect(), 1); // Inner resistor.
    let mut results = vec![];
    for light in [8.0, 40.0] {
        let mut app = app();
        app.load_scene(5);
        // Mutual inductance falls as 1/c^2: 25x weaker at c = 40. Electric coupling
        // (the loops' charges) stays the same, so a load current that drops with it
        // is induced.
        app.settings.speed_of_light = light;
        app.reset_scene();
        (SCENES[5].build)(app.world_mut());
        run(&mut app, 60.0);
        let (t, data) = record(&mut app, 100.0, &[&primary, &load]);
        let fits: Vec<(f64, f64)> = data.iter().map(|x| sine_fit(&t, x, 0.02)).collect();
        results.push((fits[1].0 / fits[0].0, fits[1].1 - fits[0].1));
    }
    assert!(results[0].0 > 5.0 * results[1].0, "{results:?}");
    // Faraday + Lenz with a resistive load: I_load ~ -dI_primary/dt, i.e. 90 degrees behind.
    let lag = (results[0].1 + 180.0).rem_euclid(360.0) - 180.0;
    assert!((lag + 90.0).abs() <= 40.0, "lag {lag}");
}

#[test]
fn antenna_radiates_waves_that_the_receiver_picks_up() {
    let (frequency, columns): (f64, Vec<usize>) = (0.1, (12..27).step_by(2).collect());
    let mut results = vec![];
    for driven in [true, false] {
        let mut app = app();
        app.load_scene(6);
        if !driven {
            // Replace the transmitter's source with wire: nothing radiates.
            app.world_mut().paint(&[(9, 16)], Material::Conductor, PaintProps { resistance: 0.1, ..PaintProps::default() });
        }
        run(&mut app, 30.0);
        let (mut t, mut fields, mut receiver) = (vec![], vec![vec![]; columns.len()], vec![]);
        for _ in 0..(30.0 / app.settings.dt) as usize {
            app.step();
            t.push(app.sim_time());
            let (e, _) = app.display_fields();
            for (f, &x) in fields.iter_mut().zip(&columns) {
                f.push(e[(x, 16, 1)]);
            }
            receiver.push(mean(&(15..18).map(|r| current_density(app.world())[(29, r, 1)]).collect::<Vec<_>>()));
        }
        // Phases along the row, unwrapped, then the slope of a straight-line fit.
        let mut phases: Vec<f64> = fields.iter().map(|f| sine_fit(&t, f, frequency).1.to_radians()).collect();
        for i in 1..phases.len() {
            while phases[i] - phases[i - 1] > PI {
                phases[i] -= 2.0 * PI;
            }
            while phases[i] - phases[i - 1] < -PI {
                phases[i] += 2.0 * PI;
            }
        }
        let xs: Vec<f64> = columns.iter().map(|&c| c as f64).collect();
        let (mx, mp) = (mean(&xs), mean(&phases));
        let slope = xs.iter().zip(&phases).map(|(x, p)| (x - mx) * (p - mp)).sum::<f64>()
            / xs.iter().map(|x| (x - mx).powi(2)).sum::<f64>();
        results.push((slope, sine_fit(&t, &receiver, frequency).0));
    }
    let wave = 2.0 * PI * frequency / 2.0; // The antenna scene's speed of light is 2.
    // A travelling wave's phase falls with distance at the speed of light.
    assert_rel(results[0].0, -wave, 0.35);
    assert!(results[0].1 > 5.0 * results[1].1);
}
