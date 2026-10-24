//! Built-in example scenes, loaded with F1-F7.
//!
//! Each scene is built from code: it resets physics settings and display layers to
//! their defaults, paints materials and test charges, then sets the layers that
//! show its point best. Scenes start from rest, so you see them settle.

use crate::layers::{LayerId, Layers};
use crate::materials::{Cell, Material, Orientation, PaintProps};
use crate::settings::Settings;
use crate::world::World;

pub struct Scene {
    pub title: &'static str,
    /// One line, shown in the info panel on load.
    pub hint: &'static str,
    /// Grid shape: scenes are laid out on a fixed grid, whatever the canvas.
    pub cols: usize,
    pub rows: usize,
    pub build: fn(&mut World),
    /// Adjusts the layers after resetting them all to defaults.
    pub layers: fn(&mut Layers),
    pub sim_speed: f64,
    /// Physics settings that differ from the defaults for this scene.
    pub settings: fn(&mut Settings),
}

/// A rectangle's border cells, corners included, sorted.
pub fn outline(c0: usize, c1: usize, r0: usize, r1: usize) -> Vec<Cell> {
    let mut cells: Vec<Cell> = (c0..=c1).flat_map(|c| [(c, r0), (c, r1)]).collect();
    cells.extend((r0..=r1).flat_map(|r| [(c0, r), (c1, r)]));
    cells.sort_unstable();
    cells.dedup();
    cells
}

/// Cells of a horizontal (r0 == r1) or vertical (c0 == c1) run, ends included.
pub fn line(c0: usize, c1: usize, r0: usize, r1: usize) -> Vec<Cell> {
    block(c0, c1 + 1, r0, r1 + 1)
}

/// Cells of the block [c0, c1) x [r0, r1).
pub fn block(c0: usize, c1: usize, r0: usize, r1: usize) -> Vec<Cell> {
    (c0..c1).flat_map(|c| (r0..r1).map(move |r| (c, r))).collect()
}

fn wire(resistance: f64) -> PaintProps {
    PaintProps { resistance, ..PaintProps::default() }
}

fn battery(emf: f64, orientation: Orientation, frequency: f64) -> PaintProps {
    PaintProps { emf, orientation, frequency, ..PaintProps::default() }
}

/// The textbook circuit: battery on the left (+ up), resistor on the right.
pub fn battery_and_resistor(world: &mut World) {
    world.paint(&outline(10, 29, 7, 24), Material::Conductor, wire(0.2));
    world.paint(&line(29, 29, 12, 19), Material::Conductor, wire(5.0));
    world.paint(&line(10, 10, 15, 15), Material::Battery, battery(6.0, Orientation::Up, 0.0));
}

/// A neutral conductor between a positive and a negative test charge.
pub fn screening(world: &mut World) {
    world.paint(&block(15, 25, 11, 21), Material::Conductor, wire(1.0));
    world.place_charges(&[(8, 16)], 5.0);
    world.place_charges(&[(31, 16)], -5.0);
}

/// Two identical loops: one cell of EMF 2 on the left, three in series on the right.
pub fn series_batteries(world: &mut World) {
    for (c0, cells) in [(3, 1), (22, 3)] {
        world.paint(&outline(c0, c0 + 14, 8, 23), Material::Conductor, wire(0.5));
        let middle = c0 + 7;
        world.paint(&line(middle - cells / 2, middle + cells / 2, 23, 23), Material::Battery,
                    battery(2.0, Orientation::Right, 0.0));
    }
}

/// Two large parallel plates, close together, each wired to one battery terminal.
///
/// The gap (5 cells) is several smoothing lengths wide so the uniform field between
/// the plates is resolved; low-resistance wires keep the charging (RC) time short.
pub fn capacitor(world: &mut World) {
    let w = wire(0.1);
    world.paint(&line(16, 16, 8, 23), Material::Conductor, w);
    world.paint(&line(22, 22, 8, 23), Material::Conductor, w);
    world.paint(&line(19, 19, 28, 28), Material::Battery, battery(8.0, Orientation::Right, 0.0));
    for cells in [
        line(8, 18, 28, 28), line(8, 8, 15, 27), line(9, 15, 15, 15),     // - side, to left plate
        line(20, 30, 28, 28), line(30, 30, 15, 27), line(23, 29, 15, 15), // + side, to right plate
    ] {
        world.paint(&cells, Material::Conductor, w);
    }
}

/// Three resistors between two bus bars: two identical, one twice as wide.
///
/// Branches are 2 and 4 cells wide rather than 1 and 2: one-cell wires resolve the
/// field worst at their junctions, which would blur the 2x comparison.
pub fn parallel_paths(world: &mut World) {
    let bus = wire(0.05);
    for (cols, rows) in [((4, 7), (9, 30)), ((33, 36), (9, 30)), ((4, 36), (28, 30))] {
        world.paint(&block(cols.0, cols.1, rows.0, rows.1), Material::Conductor, bus);
    }
    world.paint(&line(19, 19, 28, 29), Material::Battery, battery(6.0, Orientation::Left, 0.0));
    for (r0, r1) in [(10, 11), (16, 17), (21, 24)] {
        world.paint(&line(7, 32, r0, r1), Material::Conductor, bus);
        world.paint(&line(12, 27, r0, r1), Material::Conductor, wire(2.0));
    }
}

/// A big AC-driven loop with a smaller loop inside it, closed through a resistor.
///
/// In 2D a current loop is a long solenoid: its field is uniform inside and zero
/// outside, so a transformer's coils must be nested. The inner loop catches the
/// fraction of the outer loop's changing flux equal to its share of the area.
pub fn transformer(world: &mut World) {
    world.paint(&outline(4, 35, 4, 27), Material::Conductor, wire(0.1));
    // Two sources on opposite sides, both driving the same way round: the outer loop's
    // charge then swings as a quadrupole, whose electric field is weak in the middle,
    // so the inner loop is driven by induction rather than by being charged up.
    world.paint(&line(4, 4, 15, 15), Material::Battery, battery(5.0, Orientation::Up, 0.02));
    world.paint(&line(35, 35, 16, 16), Material::Battery, battery(5.0, Orientation::Down, 0.02));
    world.paint(&outline(12, 27, 10, 21), Material::Conductor, wire(0.1));
    world.paint(&line(27, 27, 13, 18), Material::Conductor, wire(2.0));
}

/// A half-wave dipole driven at its centre, and an identical receiver one wavelength away.
pub fn antenna(world: &mut World) {
    // c = 2 cells per time unit and f = 0.1 per time unit: wavelength 20 cells.
    world.paint(&line(9, 9, 11, 21), Material::Conductor, wire(0.1));
    world.paint(&line(9, 9, 16, 16), Material::Battery, battery(10.0, Orientation::Up, 0.1));
    world.paint(&line(29, 29, 11, 21), Material::Conductor, wire(0.1));
    world.paint(&line(29, 29, 16, 16), Material::Conductor, wire(2.0));
}

fn defaults(_: &mut Settings) {}

pub const SCENES: [Scene; 7] = [
    Scene {
        title: "Battery and resistor",
        hint: "energy (green S arrows) flows from the battery through space into the resistor",
        cols: 40,
        rows: 32,
        build: battery_and_resistor,
        // Linear S: on a log scale the weak flow outside the loop looks as strong as inside.
        layers: |l| {
            l[LayerId::E].visible = false;
            l[LayerId::S].visible = true;
            l[LayerId::S].log_scale = false;
        },
        sim_speed: 8.0,
        settings: defaults,
    },
    Scene {
        title: "Screening",
        hint: "electrons shift until E inside the conductor is (nearly) zero",
        cols: 40,
        rows: 32,
        build: screening,
        layers: |l| l[LayerId::B].visible = false,
        sim_speed: 8.0,
        settings: defaults,
    },
    Scene {
        title: "Batteries in series",
        hint: "3 cells in series drive 3x the current: hover the wires to compare",
        cols: 40,
        rows: 32,
        build: series_batteries,
        layers: |l| l[LayerId::S].visible = false,
        sim_speed: 8.0,
        settings: defaults,
    },
    Scene {
        title: "Capacitor",
        hint: "the plates charge until they are one EMF apart; between them E is nearly uniform",
        cols: 40,
        rows: 32,
        build: capacitor,
        layers: |l| l[LayerId::B].visible = false,
        sim_speed: 8.0,
        settings: defaults,
    },
    Scene {
        title: "Parallel paths",
        hint: "hover the resistors and read 'Wire I': the two thin ones carry equal currents, \
               the double-wide one twice as much",
        cols: 40,
        rows: 32,
        build: parallel_paths,
        layers: |l| l[LayerId::S].visible = false,
        sim_speed: 8.0,
        settings: defaults,
    },
    Scene {
        title: "Transformer",
        hint: "the inner loop has no source: the outer loop's changing magnetic flux drives the \
               current through its resistor",
        cols: 40,
        rows: 32,
        build: transformer,
        layers: |l| l[LayerId::E].visible = false,
        // One AC period takes about 3 seconds at 16x.
        sim_speed: 16.0,
        settings: defaults,
    },
    Scene {
        title: "Antenna",
        hint: "the left wire radiates waves at light speed; the right one, a wavelength away, \
               picks them up (hover its middle: Wire I)",
        cols: 40,
        rows: 32,
        build: antenna,
        layers: |l| {
            l[LayerId::S].visible = true;
            l[LayerId::E].visible = false;
        },
        sim_speed: 8.0,
        // Slow light, so the waves are short (20 cells) and easy to watch.
        settings: |s| s.speed_of_light = 2.0,
    },
];
