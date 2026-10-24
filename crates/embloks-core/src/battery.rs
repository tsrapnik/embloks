//! Batteries as voltage sources with a small internal resistance.
//!
//! A battery cell conducts like a wire, with its own ion and electrons, but it also
//! pushes its electrons towards its - side with an *impressed field*: a
//! non-electrostatic force of EMF per cell along its axis (Griffiths' source force
//! f_s, the "resistive voltage source" of FDTD). Electrons drift through the
//! battery with drag, so they keep moving until the charge they pile up on the
//! terminals cancels that force, which is when V(+) - V(-) = EMF. Under load the
//! battery cells' own drag drops part of it: V = EMF - I * R_internal, a textbook
//! voltage source. Charge only ever moves continuously, so the current through a
//! battery is carried by real electron velocities, like in any wire.
//!
//! Battery cells lined up along their orientation axis form one *stack*; the wire
//! cells just beyond its two ends are its terminals. With so few electrons per cell
//! the impressed field alone is not a steady source: the voltage rests on the
//! handful of electrons at the battery's faces, it comes up short, and it jitters
//! as they shift. So each stack measures what it delivers, V + I * R_internal, and
//! keeps scaling its impressed field until that equals the EMF (see trim): a
//! feedback loop on the force, not on the charge.

use std::f64::consts::PI;
use std::hash::{Hash, Hasher};

use ndarray::{Array2, Array3};

use crate::materials::{Cell, Material, Materials, ORIENTATION_VECTORS, Orientation};
use crate::settings::Settings;
use crate::world::World;

/// A terminal cell, possibly just off the grid (then it is never wired).
pub type Terminal = (i64, i64);

#[derive(Clone, Debug)]
pub struct Stack {
    pub cells: Vec<Cell>,
    // Terminal cells before the first and after the last battery cell along the axis.
    pub low: Terminal,
    pub high: Terminal,
    // Voltage the stack maintains: V(high) - V(low). Negative if it points towards low.
    // For AC cells this is the amplitude.
    pub emf: f64,
    pub internal_resistance: f64,
    // Per cell: (signed EMF, frequency). Frequency 0 is DC.
    pub sources: Vec<(f64, f64)>,
}

// Stacks key the battery gains, like the frozen dataclass in the Python original:
// equal when every field is, floats compared bitwise.
impl PartialEq for Stack {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl Eq for Stack {}

impl Hash for Stack {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key().hash(state);
    }
}

impl Stack {
    #[allow(clippy::type_complexity)]
    fn key(&self) -> (&[Cell], Terminal, Terminal, u64, u64, Vec<(u64, u64)>) {
        (
            &self.cells,
            self.low,
            self.high,
            self.emf.to_bits(),
            self.internal_resistance.to_bits(),
            self.sources.iter().map(|(e, f)| (e.to_bits(), f.to_bits())).collect(),
        )
    }

    pub fn is_ac(&self) -> bool {
        self.sources.iter().any(|&(_, frequency)| frequency > 0.0)
    }

    /// Sum of the cells' EMF magnitudes: the most the stack can ever drive.
    pub fn amplitude(&self) -> f64 {
        self.sources.iter().map(|(emf, _)| emf.abs()).sum()
    }

    /// V(high) - V(low) the stack drives at `time`: AC cells follow EMF * sin(2 pi f t).
    pub fn emf_at(&self, time: f64) -> f64 {
        if self.sources.is_empty() {
            return self.emf;
        }
        self.sources
            .iter()
            .map(|&(emf, frequency)| {
                if frequency == 0.0 { emf } else { emf * (2.0 * PI * frequency * time).sin() }
            })
            .sum()
    }
}

/// Group battery cells into stacks: runs along their own (shared) axis.
pub fn find_stacks(materials: &Materials) -> Vec<Stack> {
    let horizontal = |cell: Cell| {
        let o = Orientation::from_u8(materials.orientation[cell]);
        o == Orientation::Right || o == Orientation::Left
    };
    let in_run = |(c, r): (i64, i64), axis_horizontal: bool| {
        materials.in_bounds(c, r) && {
            let cell = (c as usize, r as usize);
            materials.material(cell) == Material::Battery && horizontal(cell) == axis_horizontal
        }
    };

    let mut stacks = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for ((c, r), &kind) in materials.kind.indexed_iter() {
        if kind != Material::Battery as u8 || seen.contains(&(c, r)) {
            continue;
        }
        let axis_horizontal = horizontal((c, r));
        let step: (i64, i64) = if axis_horizontal { (1, 0) } else { (0, 1) };
        // Walk back to the first cell of the run, then collect it forwards.
        let mut first = (c as i64, r as i64);
        while in_run((first.0 - step.0, first.1 - step.1), axis_horizontal) {
            first = (first.0 - step.0, first.1 - step.1);
        }
        let mut cells = Vec::new();
        let mut cell = first;
        while in_run(cell, axis_horizontal) {
            cells.push((cell.0 as usize, cell.1 as usize));
            cell = (cell.0 + step.0, cell.1 + step.1);
        }
        seen.extend(cells.iter().copied());
        // RIGHT and DOWN point along +step; LEFT and UP against it.
        let sources: Vec<(f64, f64)> = cells
            .iter()
            .map(|&x| {
                let o = Orientation::from_u8(materials.orientation[x]);
                let sign = if matches!(o, Orientation::Right | Orientation::Down) { 1.0 } else { -1.0 };
                (sign * materials.emf[x], materials.frequency[x])
            })
            .collect();
        let total = sources.iter().map(|(signed, _)| signed).sum();
        let low = (first.0 - step.0, first.1 - step.1);
        let internal_resistance = cells.iter().map(|&x| materials.resistance[x]).sum();
        stacks.push(Stack { cells, low, high: cell, emf: total, internal_resistance, sources });
    }
    stacks
}

/// Whether a terminal cell is a wire (a battery only works with both wired).
pub fn wired(materials: &Materials, (c, r): Terminal) -> bool {
    materials.in_bounds(c, r) && materials.material((c as usize, r as usize)) == Material::Conductor
}

/// The cells around a terminal (3x3, clipped to the grid) whose potential it reads.
pub fn terminal_block(materials: &Materials, (c, r): Terminal) -> impl Iterator<Item = Cell> + '_ {
    let cols = (c - 1).max(0)..(c + 2).min(materials.cols as i64);
    cols.flat_map(move |x| ((r - 1).max(0)..(r + 2).min(materials.rows as i64)).map(move |y| (x as usize, y as usize)))
}

/// Mean potential over a terminal cell and its 3x3 neighbours of wire in the same island.
///
/// None if the terminal isn't a wire. Averaging damps the noise from individual
/// electrons passing right by the cell centre. Battery cells are left out: the
/// potential climbs through them.
pub fn terminal_potential(world: &World, potential: &Array2<f64>, cell: Terminal) -> Option<f64> {
    let m = &world.materials;
    if !wired(m, cell) {
        return None;
    }
    let labels = &world.connectivity.labels;
    let island = labels[(cell.0 as usize, cell.1 as usize)];
    let (sum, count) = terminal_block(m, cell)
        .filter(|&x| labels[x] == island && m.material(x) == Material::Conductor)
        .fold((0.0, 0), |(sum, count), x| (sum + potential[x], count + 1));
    Some(sum / count as f64)
}

/// Mask (cols, rows) of the cells whose potential terminal_potential reads.
pub fn probe_cells(world: &World) -> Array2<bool> {
    let m = &world.materials;
    let mut mask = Array2::from_elem((m.cols, m.rows), false);
    for stack in &world.stacks {
        for terminal in [stack.low, stack.high] {
            for cell in terminal_block(m, terminal) {
                mask[cell] = true;
            }
        }
    }
    mask
}

/// Per-cell gain of the impressed field: each stack's gain on its cells, 1 elsewhere.
fn gain_grid(world: &World) -> Array2<f64> {
    let m = &world.materials;
    let mut gain = Array2::ones((m.cols, m.rows));
    for (stack, &value) in &world.battery_gain {
        for &cell in &stack.cells {
            gain[cell] = value;
        }
    }
    gain
}

/// The batteries' impressed field per cell at time `time`, (cols, rows, 2).
///
/// EMF per cell towards each battery cell's + side, so the force on an electron
/// (charge < 0) points to its - side; zero outside batteries. AC cells follow
/// EMF * sin(2 pi f t). Over a cell of length 1 it adds EMF to the integral of
/// the field, times the stack's gain (see trim).
pub fn impressed_field_grid(world: &World, time: f64) -> Array3<f64> {
    let m = &world.materials;
    let gain = gain_grid(world);
    let mut field = Array3::zeros((m.cols, m.rows, 2));
    for ((c, r), &emf) in m.emf.indexed_iter() {
        if emf == 0.0 {
            continue;
        }
        let frequency = m.frequency[(c, r)];
        let mut value = emf * gain[(c, r)];
        if frequency > 0.0 {
            value *= (2.0 * PI * frequency * time).sin();
        }
        let (ox, oy) = ORIENTATION_VECTORS[m.orientation[(c, r)] as usize];
        field[(c, r, 0)] = value * ox;
        field[(c, r, 1)] = value * oy;
    }
    field
}

/// The impressed field for electrons in the given cells (see impressed_field_grid).
pub fn impressed_field(world: &World, cells: &[Cell], time: f64) -> Vec<[f64; 2]> {
    let field = impressed_field_grid(world, time);
    cells.iter().map(|&(c, r)| [field[(c, r, 0)], field[(c, r, 1)]]).collect()
}

/// Current through a stack, from low terminal to high, given currents per cell (cols, rows, 2).
///
/// Averaged over the battery cells and its wired terminals: electrons bunch up
/// where they leave the battery, so a single cell over- or under-reads.
pub fn stack_current(world: &World, stack: &Stack, current: &Array3<f64>) -> f64 {
    let direction = ((stack.high.0 - stack.low.0).signum() as f64, (stack.high.1 - stack.low.1).signum() as f64);
    let terminals = [stack.low, stack.high]
        .into_iter()
        .filter(|&t| wired(&world.materials, t))
        .map(|(c, r)| (c as usize, r as usize));
    let cells: Vec<Cell> = stack.cells.iter().copied().chain(terminals).collect();
    let total: f64 = cells
        .iter()
        .map(|&(c, r)| current[(c, r, 0)] * direction.0 + current[(c, r, 1)] * direction.1)
        .sum();
    total / cells.len() as f64
}

/// Nudge every wired stack's gain so it delivers its EMF: V + I * R_internal = EMF.
///
/// `potential` and `current` (per cell, (cols, rows, 2)) are the present ones. The
/// gain integrates the shortfall over battery_trim_time, weighted by the EMF
/// itself (a least-mean-squares fit), so it works for AC sources as well: near
/// a zero crossing, where the shortfall says little, it hardly moves.
pub fn trim(world: &mut World, potential: &Array2<f64>, current: &Array3<f64>, dt: f64, settings: &Settings, time: f64) {
    let stacks = world.stacks.clone();
    for stack in stacks {
        let v_low = terminal_potential(world, potential, stack.low);
        let v_high = terminal_potential(world, potential, stack.high);
        let amplitude = stack.amplitude();
        let (Some(v_low), Some(v_high)) = (v_low, v_high) else {
            world.battery_gain.remove(&stack);
            continue;
        };
        if amplitude == 0.0 {
            world.battery_gain.remove(&stack);
            continue;
        }
        let target = stack.emf_at(time);
        let delivered = v_high - v_low + stack_current(world, &stack, current) * stack.internal_resistance;
        let mut gain = world.battery_gain.get(&stack).copied().unwrap_or(1.0);
        gain += dt / settings.battery_trim_time * (target - delivered) * target / (amplitude * amplitude);
        let (low, high) = settings.battery_gain_range;
        world.battery_gain.insert(stack, gain.clamp(low, high));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::PaintProps;
    use crate::physics::current_density;
    use crate::scenes;
    use crate::simulation::Simulation;
    use crate::test_util::{assert_rel, mean, rectangle, row};

    fn paint_battery(m: &mut Materials, cells: &[Cell], orientation: Orientation, emf: f64) {
        m.paint(cells, Material::Battery, PaintProps { resistance: 0.1, emf, orientation, ..PaintProps::default() });
    }

    fn battery(emf: f64, orientation: Orientation, frequency: f64) -> PaintProps {
        PaintProps { emf, orientation, frequency, ..PaintProps::default() }
    }

    fn wire(resistance: f64) -> PaintProps {
        PaintProps { resistance, ..PaintProps::default() }
    }

    #[test]
    fn single_battery_stack_and_terminals() {
        let mut m = Materials::new(8, 6);
        paint_battery(&mut m, &[(3, 2)], Orientation::Right, 5.0);
        let stacks = find_stacks(&m);
        assert_eq!(stacks.len(), 1);
        let stack = &stacks[0];
        assert_eq!((stack.low, stack.high), ((2, 2), (4, 2)));
        assert_eq!(stack.emf, 5.0);
        assert_rel(stack.internal_resistance, 0.1, 1e-9);
    }

    #[test]
    fn pointing_left_or_up_gives_negative_emf_along_axis() {
        let mut m = Materials::new(8, 6);
        paint_battery(&mut m, &[(3, 2)], Orientation::Left, 5.0);
        paint_battery(&mut m, &[(6, 3)], Orientation::Up, 2.0);
        let stacks = find_stacks(&m);
        let by_cell = |cell: Cell| stacks.iter().find(|s| s.cells == vec![cell]).unwrap();
        assert_eq!(by_cell((3, 2)).emf, -5.0);
        let vertical = by_cell((6, 3));
        assert_eq!((vertical.low, vertical.high, vertical.emf), ((6, 2), (6, 4), -2.0));
    }

    #[test]
    fn batteries_in_series_add_and_opposing_ones_subtract() {
        let mut m = Materials::new(10, 6);
        paint_battery(&mut m, &[(2, 1), (3, 1), (4, 1)], Orientation::Right, 2.0);
        paint_battery(&mut m, &[(2, 3), (3, 3)], Orientation::Right, 5.0);
        paint_battery(&mut m, &[(4, 3)], Orientation::Left, 1.0);
        let stacks = find_stacks(&m);
        let by_first = |cell: Cell| stacks.iter().find(|s| s.cells[0] == cell).unwrap();
        assert_eq!(by_first((2, 1)).emf, 6.0);
        assert_rel(by_first((2, 1)).internal_resistance, 0.3, 1e-9);
        assert_eq!(by_first((2, 3)).emf, 9.0);
        assert_eq!(by_first((2, 3)).high, (5, 3));
    }

    #[test]
    fn batteries_on_different_axes_are_separate_stacks() {
        let mut m = Materials::new(8, 6);
        paint_battery(&mut m, &[(3, 2)], Orientation::Right, 1.0);
        paint_battery(&mut m, &[(4, 2)], Orientation::Down, 1.0);
        assert_eq!(find_stacks(&m).len(), 2);
    }

    fn terminal_voltage(sim: &mut Simulation, stack: &Stack) -> f64 {
        let potential = sim.potential(None);
        let high = terminal_potential(&sim.world, &potential, stack.high).unwrap();
        let low = terminal_potential(&sim.world, &potential, stack.low).unwrap();
        high - low
    }

    #[test]
    fn open_circuit_terminal_voltage_reaches_emf() {
        let mut w = World::new(40, 20, Settings::default(), 0);
        w.paint(&row(8, 20, 10), Material::Conductor, wire(0.5));
        w.paint(&row(21, 33, 10), Material::Conductor, wire(0.5));
        w.paint(&[(20, 10)], Material::Battery, battery(3.0, Orientation::Right, 0.0));
        let mut sim = Simulation::new(w);
        sim.run(15.0);
        let stack = sim.world.stacks[0].clone();
        let mut voltages = vec![];
        for _ in 0..(5.0 / sim.world.settings.dt) as usize {
            sim.step();
            voltages.push(terminal_voltage(&mut sim, &stack));
        }
        assert_rel(mean(&voltages), 3.0, 0.03);
        // Electrons move smoothly, so the voltage holds steady instead of dithering.
        let m = mean(&voltages);
        let std = (voltages.iter().map(|v| (v - m).powi(2)).sum::<f64>() / voltages.len() as f64).sqrt();
        assert!(std < 0.1, "std {std}");
        // Electrons shifted from the + side to the - side: their centre of charge moved
        // left of the battery's centre. (Counting whole electrons per wire, as the Python
        // test did, hinges on whether a single one has crossed out of the battery cell.)
        let p = &sim.world.particles;
        let centre = p.pos.iter().map(|q| q[0]).sum::<f64>() / p.len() as f64;
        assert!(centre < 20.5 - 0.01, "centre {centre}");
    }

    /// Mean x current along the top edge of the test loop, sampled 20 times.
    fn top_current(sim: &mut Simulation) -> f64 {
        let mut samples = vec![];
        for _ in 0..20 {
            sim.run(0.5);
            let current = current_density(&sim.world);
            samples.push(mean(&(6..22).map(|c| current[(c, 3, 0)]).collect::<Vec<_>>()));
        }
        mean(&samples)
    }

    fn loop_world(settings: Settings, emf: f64, orientation: Orientation, frequency: f64) -> (World, usize) {
        let mut w = World::new(30, 20, settings, 0);
        let lp = rectangle(4, 23, 3, 12);
        w.paint(&lp, Material::Conductor, wire(0.5));
        w.paint(&[(14, 12)], Material::Battery, battery(emf, orientation, frequency));
        (w, lp.len())
    }

    #[test]
    fn loop_current_follows_ohms_law_and_polarity() {
        for (orientation, sign) in [(Orientation::Right, 1.0), (Orientation::Left, -1.0)] {
            let s = Settings::default();
            let (w, cells) = loop_world(s.clone(), 2.0, orientation, 0.0);
            let expected = 2.0 / ((cells - 1) as f64 * 0.5 + s.battery_resistance);
            let mut sim = Simulation::new(w);
            sim.run(30.0);
            // Battery + on the right of the bottom edge pushes conventional current out of
            // its right terminal, up the right side and leftwards (-x) along the top.
            assert_rel(-sign * top_current(&mut sim), expected, 0.25);
        }
    }

    #[test]
    fn unconnected_battery_does_nothing() {
        let mut w = World::new(10, 6, Settings::default(), 0);
        w.paint(&row(1, 4, 2), Material::Conductor, wire(1.0));
        w.paint(&[(4, 2)], Material::Battery, battery(5.0, Orientation::Right, 0.0));
        let before = w.particles.len();
        let mut sim = Simulation::new(w);
        sim.run(1.0);
        assert_eq!(sim.world.particles.len(), before);
        assert!(sim.world.battery_gain.is_empty());
    }

    #[test]
    fn ohms_law_does_not_depend_on_charge_to_mass() {
        // Drift current I = E / R must hold for any electron charge/mass (drag scales with it).
        for charge_to_mass in [0.5, 1.0, 4.0] {
            let s = Settings { carrier_charge_to_mass: charge_to_mass, ..Settings::default() };
            let (w, cells) = loop_world(s.clone(), 2.0, Orientation::Right, 0.0);
            let expected = 2.0 / ((cells - 1) as f64 * 0.5 + s.battery_resistance);
            let mut sim = Simulation::new(w);
            sim.run(30.0);
            assert_rel(-top_current(&mut sim), expected, 0.25);
        }
    }

    #[test]
    fn impressed_field_pushes_electrons_to_the_minus_side() {
        let mut w = World::new(10, 6, Settings::default(), 0);
        w.paint(&[(3, 2), (4, 2)], Material::Battery, battery(2.0, Orientation::Left, 0.25));
        w.paint(&[(6, 4)], Material::Battery, battery(5.0, Orientation::Down, 0.0));
        let cells = [(3, 2), (6, 4), (1, 1)];
        // Fields point to each cell's + side; vacuum gets none. sin(2 pi * 0.25 * 1) = 1.
        let field = impressed_field(&w, &cells, 1.0);
        let expected = [[-2.0, 0.0], [0.0, 5.0], [0.0, 0.0]];
        for (f, e) in field.iter().zip(expected) {
            assert!((f[0] - e[0]).abs() < 1e-12 && (f[1] - e[1]).abs() < 1e-12);
        }
        let later = impressed_field(&w, &cells[..1], 2.0)[0];
        assert!(later[0].abs() < 1e-12 && later[1].abs() < 1e-12);
        let stack = w.stacks.iter().find(|s| s.cells.contains(&(3, 2))).unwrap().clone();
        w.battery_gain.insert(stack, 1.5);
        let gained = impressed_field(&w, &cells[..1], 1.0)[0];
        assert!((gained[0] + 3.0).abs() < 1e-12 && gained[1].abs() < 1e-12);
    }

    #[test]
    fn battery_electrons_drift_instead_of_jumping() {
        let s = Settings::default();
        let (w, _) = loop_world(s.clone(), 10.0, Orientation::Right, 0.0);
        let mut sim = Simulation::new(w);
        for _ in 0..(5.0 / s.dt) as usize {
            let before = sim.world.particles.pos.clone();
            let substeps = sim.step();
            let moved = sim.world.particles.pos.iter().zip(&before)
                .map(|(a, b)| (a[0] - b[0]).abs().max((a[1] - b[1]).abs()))
                .fold(0.0, f64::max);
            assert!(moved <= substeps as f64 * s.max_step + 1e-9);
        }
    }

    #[test]
    fn trim_corrects_a_shortfall_and_follows_the_load() {
        let s = Settings::default();
        let mut w = World::new(10, 6, s.clone(), 0);
        w.paint(&row(1, 4, 2), Material::Conductor, wire(1.0));
        w.paint(&row(5, 8, 2), Material::Conductor, wire(1.0));
        w.paint(&[(4, 2)], Material::Battery, battery(4.0, Orientation::Right, 0.0));
        let stack = w.stacks[0].clone();
        let mut potential = Array2::zeros((10, 6));
        potential.slice_mut(ndarray::s![5.., ..]).fill(3.0); // Delivers 3 of 4: the gain goes up.
        let mut current = Array3::zeros((10, 6, 2));
        trim(&mut w, &potential, &current, 0.1, &s, 0.0);
        assert_rel(w.battery_gain[&stack], 1.0 + 0.1 / s.battery_trim_time / 4.0, 1e-9);
        // Under load, V = EMF - I R is exactly right: nothing to correct.
        w.battery_gain.insert(stack.clone(), 1.2);
        for c in 3..6 {
            current[(c, 2, 0)] = 10.0;
        }
        trim(&mut w, &potential, &current, 0.1, &s, 0.0);
        assert_rel(w.battery_gain[&stack], 1.2, 1e-9);
    }

    #[test]
    fn ac_stack_follows_a_sine_and_dc_stays_constant() {
        let mut m = Materials::new(10, 6);
        paint_battery(&mut m, &[(3, 2)], Orientation::Right, 4.0);
        m.paint(&[(6, 2)], Material::Battery,
                PaintProps { resistance: 0.1, emf: 2.0, orientation: Orientation::Left, frequency: 0.25, density: 1.0 });
        let stacks = find_stacks(&m);
        let dc = stacks.iter().find(|s| s.cells[0] == (3, 2)).unwrap();
        let ac = stacks.iter().find(|s| s.cells[0] == (6, 2)).unwrap();
        assert!(!dc.is_ac() && dc.emf_at(0.0) == 4.0 && dc.emf_at(7.3) == 4.0);
        assert!(ac.is_ac() && ac.emf == -2.0);
        assert_rel(ac.emf_at(1.0), -2.0, 1e-9); // sin(2 pi * 0.25 * 1) = 1.
        assert!(ac.emf_at(2.0).abs() < 1e-12);
    }

    #[test]
    fn parallel_batteries_hold_steady() {
        // Two stacks feeding the same terminals must not fight over them.
        let mut w = World::new(40, 32, Settings::default(), 0);
        scenes::parallel_paths(&mut w);
        let mut sim = Simulation::new(w);
        sim.run(5.0);
        let fastest = sim.world.particles.vel.iter().map(|v| v[0].hypot(v[1])).fold(0.0, f64::max);
        assert!(fastest < 2.0);
        assert!(sim.world.battery_gain.values().all(|&g| 0.5 < g && g < 2.0));
    }

    #[test]
    fn ac_battery_voltage_follows_its_sine() {
        let s = Settings::default();
        let (w, _) = loop_world(s.clone(), 5.0, Orientation::Right, 0.05);
        let mut sim = Simulation::new(w);
        sim.run(10.0);
        let stack = sim.world.stacks[0].clone();
        let (mut voltages, mut emfs) = (vec![], vec![]);
        for _ in 0..(20.0 / s.dt) as usize {
            sim.step();
            voltages.push(terminal_voltage(&mut sim, &stack));
            emfs.push(stack.emf_at(sim.time));
        }
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        assert!((dot(&voltages, &emfs) / dot(&emfs, &emfs) - 1.0).abs() < 0.05);
        let (mv, me) = (mean(&voltages), mean(&emfs));
        let dv: Vec<f64> = voltages.iter().map(|v| v - mv).collect();
        let de: Vec<f64> = emfs.iter().map(|e| e - me).collect();
        assert!(dot(&dv, &de) / (dot(&dv, &dv) * dot(&de, &de)).sqrt() > 0.99);
    }

    #[test]
    fn a_steady_circuit_leaves_the_field_around_it_calm() {
        // Regression: batteries that moved electrons by jumps shook E in the empty space
        // around them every tick (tick-to-tick rms ~0.015 in this scene).
        let s = Settings::default();
        let mut w = World::new(40, 30, s.clone(), 0);
        scenes::battery_and_resistor(&mut w);
        let mut sim = Simulation::new(w);
        sim.run(20.0);
        let vacuum: Vec<Cell> = sim.world.materials.kind.indexed_iter()
            .filter(|&(_, &k)| k == Material::Vacuum as u8).map(|(cell, _)| cell).collect();
        let sample = |sim: &Simulation| {
            let e = sim.e_field();
            vacuum.iter().flat_map(|&(c, r)| [e[(c, r, 0)], e[(c, r, 1)]]).collect::<Vec<_>>()
        };
        let mut previous = sample(&sim);
        let (mut total, mut count) = (0.0, 0);
        for _ in 0..(5.0 / s.dt) as usize {
            sim.step();
            let now = sample(&sim);
            total += now.iter().zip(&previous).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
            count += now.len();
            previous = now;
        }
        assert!((total / count as f64).sqrt() < 0.005);
    }
}
