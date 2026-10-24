use std::collections::HashMap;

use ndarray::Array2;
use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64;

use crate::battery::{Stack, find_stacks};
use crate::connectivity::{Connectivity, compute_connectivity};
use crate::materials::{Cell, Material, Materials, PaintProps};
use crate::particles::Particles;
use crate::settings::Settings;

/// Electrons spawn spread over their cell (see fill_pattern), offset by at most this
/// much (cells), so a fresh conductor is not a perfect lattice.
pub const SPAWN_JITTER: f64 = 0.02;

/// Materials plus the charges they carry, kept consistent while painting.
///
/// Every conductive cell (conductor or battery) holds one fixed ion of charge
/// +density * cell_charge, spread evenly over the cell (see fields::ion_charge),
/// and, when neutral, density * carriers_per_cell electrons of charge
/// -cell_charge/n each. Wires have density 1, batteries battery_density.
#[derive(Clone, Debug)]
pub struct World {
    pub settings: Settings,
    pub rng: Pcg64,
    pub materials: Materials,
    pub particles: Particles,
    /// Per battery stack: how much it scales its impressed field (battery::trim).
    pub battery_gain: HashMap<Stack, f64>,
    pub connectivity: Connectivity,
    pub stacks: Vec<Stack>,
    /// Fixed test charges, at most one per cell, at cell centres.
    pub fixed_pos: Vec<[f64; 2]>,
    pub fixed_charge: Vec<f64>,
    /// Bumped whenever test charges change, so cached fields know to rebuild.
    pub fixed_version: u64,
}

impl World {
    pub fn new(cols: usize, rows: usize, settings: Settings, seed: u64) -> Self {
        let materials = Materials::new(cols, rows);
        let mut world = Self {
            settings,
            rng: Pcg64::seed_from_u64(seed),
            connectivity: compute_connectivity(&materials),
            stacks: Vec::new(),
            materials,
            particles: Particles::new(),
            battery_gain: HashMap::new(),
            fixed_pos: Vec::new(),
            fixed_charge: Vec::new(),
            fixed_version: 0,
        };
        world.update_structure();
        world
    }

    pub fn electron_charge(&self) -> f64 {
        -self.settings.cell_charge / self.settings.carriers_per_cell as f64
    }

    pub fn electron_mass(&self) -> f64 {
        // Mass shrinks with charge, so charge/mass and thus the dynamics don't depend on n.
        -self.electron_charge() / self.settings.carrier_charge_to_mass
    }

    /// A key that changes whenever charges appear or vanish without moving: painted
    /// materials (ions, spawned electrons) or test charges.
    pub fn scene_key(&self) -> (u64, u64) {
        (self.materials.version, self.fixed_version)
    }

    pub fn paint(&mut self, cells: &[Cell], material: Material, mut props: PaintProps) {
        props.density = 1.0;
        if material == Material::Battery {
            props.resistance = self.settings.battery_resistance;
            props.density = self.settings.battery_density as f64;
        }
        let old_density = self.materials.density.clone();
        let old_labels = self.connectivity.labels.clone();
        self.materials.paint(cells, material, props);
        self.update_structure();

        let n = self.settings.carriers_per_cell as f64;
        let change = (&self.materials.density - &old_density).mapv(|d| (d * n).round() as i64);
        if change.iter().any(|&c| c < 0) {
            self.remove_carriers(&change.mapv(|c| (-c).max(0)), &old_labels);
        }
        if change.iter().any(|&c| c > 0) {
            self.spawn_carriers(&change.mapv(|c| c.max(0)));
        }
    }

    pub fn clear(&mut self) {
        self.materials.clear();
        self.particles = Particles::new();
        self.update_structure();
        self.fixed_pos.clear();
        self.fixed_charge.clear();
        self.fixed_version += 1;
    }

    /// Recompute islands and battery stacks after the materials changed.
    pub fn update_structure(&mut self) {
        self.connectivity = compute_connectivity(&self.materials);
        self.stacks = find_stacks(&self.materials);
        let stacks = &self.stacks;
        self.battery_gain.retain(|stack, _| stacks.contains(stack));
    }

    /// Put a fixed test charge at the centre of each given cell, replacing any there.
    pub fn place_charges(&mut self, cells: &[Cell], charge: f64) {
        self.remove_charges(cells);
        let mut unique = cells.to_vec();
        unique.sort_unstable();
        unique.dedup();
        for (c, r) in unique {
            self.fixed_pos.push([c as f64 + 0.5, r as f64 + 0.5]);
            self.fixed_charge.push(charge);
        }
        self.fixed_version += 1;
    }

    /// Mask over the fixed test charges: which sit in one of the given cells.
    pub fn charges_in(&self, cells: &[Cell]) -> Vec<bool> {
        let set: std::collections::HashSet<(i64, i64)> = cells.iter().map(|&(c, r)| (c as i64, r as i64)).collect();
        self.fixed_pos.iter().map(|p| set.contains(&crate::particles::cell_of(*p))).collect()
    }

    /// Remove the fixed test charges in the given cells.
    pub fn remove_charges(&mut self, cells: &[Cell]) {
        let hit = self.charges_in(cells);
        let mut i = 0;
        self.fixed_pos.retain(|_| {
            i += 1;
            !hit[i - 1]
        });
        let mut i = 0;
        self.fixed_charge.retain(|_| {
            i += 1;
            !hit[i - 1]
        });
        self.fixed_version += 1;
    }

    /// Add counts[c, r] electrons to each cell, spread evenly over it.
    fn spawn_carriers(&mut self, counts: &Array2<i64>) {
        let mut distinct: Vec<i64> = counts.iter().copied().filter(|&c| c > 0).collect();
        distinct.sort_unstable();
        distinct.dedup();
        for count in distinct {
            let pattern = fill_pattern(count as usize);
            let mut positions = Vec::new();
            for ((c, r), _) in counts.indexed_iter().filter(|&(_, &n)| n == count) {
                for offset in &pattern {
                    let jitter = self.jitter();
                    positions.push([c as f64 + offset[0] + jitter[0], r as f64 + offset[1] + jitter[1]]);
                }
            }
            let (charge, mass) = (self.electron_charge(), self.electron_mass());
            self.particles.add(&positions, charge, mass);
        }
    }

    fn jitter(&mut self) -> [f64; 2] {
        [
            self.rng.random_range(-SPAWN_JITTER..SPAWN_JITTER),
            self.rng.random_range(-SPAWN_JITTER..SPAWN_JITTER),
        ]
    }

    /// Take away counts[c, r] electrons for each cell that lost charge density.
    ///
    /// Per island, the electrons leave in this order: those sitting in erased
    /// cells, then the rest of those in the cells that lost density, then the
    /// ones nearest to those cells. Any electrons still in erased cells after
    /// that move to the nearest remaining cell of their old island, or vanish if
    /// nothing of it is left.
    fn remove_carriers(&mut self, counts: &Array2<i64>, old_labels: &Array2<i64>) {
        let conductive = self.materials.conductive();
        let n = self.particles.len();
        let cells: Vec<Cell> = (0..n).map(|i| self.clamped_cell(i)).collect();
        let particle_island: Vec<i64> = cells.iter().map(|&cell| old_labels[cell]).collect();
        let lost = counts.mapv(|c| c > 0);
        let in_lost: Vec<bool> = cells.iter().map(|&cell| lost[cell]).collect();
        let stranded: Vec<bool> = cells.iter().map(|&cell| !conductive[cell]).collect();
        let mut keep = vec![true; n];

        let mut islands: Vec<i64> =
            lost.indexed_iter().filter(|&(_, &l)| l).map(|(cell, _)| old_labels[cell]).collect();
        islands.sort_unstable();
        islands.dedup();
        for island in islands {
            let lost_here: Vec<Cell> = lost
                .indexed_iter()
                .filter(|&(cell, &l)| l && old_labels[cell] == island)
                .map(|(cell, _)| cell)
                .collect();
            let quota: usize = lost_here.iter().map(|&cell| counts[cell] as usize).sum();
            let members = |i: usize| particle_island[i] == island;
            let mut inside: Vec<usize> = (0..n).filter(|&i| members(i) && in_lost[i]).collect();
            inside.sort_by_key(|&i| !stranded[i]); // Erased cells first (stable).
            for &i in inside.iter().take(quota) {
                keep[i] = false;
            }
            if inside.len() < quota {
                let others: Vec<usize> = (0..n).filter(|&i| members(i) && !in_lost[i]).collect();
                let centres: Vec<[f64; 2]> =
                    lost_here.iter().map(|&(c, r)| [c as f64 + 0.5, r as f64 + 0.5]).collect();
                let mut by_distance: Vec<(f64, usize)> = others
                    .iter()
                    .map(|&i| (nearest_distance(self.particles.pos[i], &centres), i))
                    .collect();
                by_distance.sort_by(|a, b| a.0.total_cmp(&b.0));
                for &(_, i) in by_distance.iter().take(quota - inside.len()) {
                    keep[i] = false;
                }
            }

            let leftover: Vec<usize> = (0..n).filter(|&i| keep[i] && members(i) && stranded[i]).collect();
            if !leftover.is_empty() {
                let targets: Vec<Cell> = conductive
                    .indexed_iter()
                    .filter(|&(cell, &c)| c && old_labels[cell] == island)
                    .map(|(cell, _)| cell)
                    .collect();
                if targets.is_empty() {
                    for i in leftover {
                        keep[i] = false;
                    }
                } else {
                    self.move_to_nearest_cell(&leftover, &targets);
                }
            }
        }
        self.particles.keep(&keep);
    }

    fn clamped_cell(&self, i: usize) -> Cell {
        let (c, r) = self.particles.cell(i);
        (c.clamp(0, self.materials.cols as i64 - 1) as usize, r.clamp(0, self.materials.rows as i64 - 1) as usize)
    }

    fn move_to_nearest_cell(&mut self, indices: &[usize], cells: &[Cell]) {
        let centres: Vec<[f64; 2]> = cells.iter().map(|&(c, r)| [c as f64 + 0.5, r as f64 + 0.5]).collect();
        for &i in indices {
            let pos = self.particles.pos[i];
            let nearest = centres
                .iter()
                .min_by(|a, b| distance2(pos, **a).total_cmp(&distance2(pos, **b)))
                .copied()
                .unwrap();
            let jitter = self.jitter();
            self.particles.pos[i] = [nearest[0] + jitter[0], nearest[1] + jitter[1]];
            self.particles.vel[i] = [0.0; 2];
        }
    }
}

/// n points spread evenly over a unit cell: a rank-1 lattice.
///
/// Columns and rows are each n evenly spaced values, so per axis the points
/// deposit 3/4 on the cell centre and 1/8 on each neighbour, like the ion
/// (fields::spread_over_cells): a freshly painted conductor is almost field-free.
pub fn fill_pattern(n: usize) -> Vec<[f64; 2]> {
    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 { a } else { gcd(b, a % b) }
    }
    // Step through the rows by a generator coprime to n near n / golden ratio^2,
    // so successive points land far apart.
    let target = 0.382 * n as f64;
    let step = (1..=n)
        .filter(|&g| gcd(g, n) == 1)
        .min_by(|&a, &b| (a as f64 - target).abs().total_cmp(&(b as f64 - target).abs()))
        .unwrap_or(1);
    (0..n)
        .map(|i| [(i as f64 + 0.5) / n as f64, ((i * step) % n) as f64 / n as f64 + 0.5 / n as f64])
        .collect()
}

fn distance2(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

/// Distance from a point to the closest of `centres`.
fn nearest_distance(point: [f64; 2], centres: &[[f64; 2]]) -> f64 {
    centres.iter().map(|&c| distance2(point, c)).fold(f64::INFINITY, f64::min).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::row;

    fn make_world(n: u32, cols: usize, rows: usize) -> World {
        World::new(cols, rows, Settings { carriers_per_cell: n, ..Settings::default() }, 0)
    }

    fn wire() -> PaintProps {
        PaintProps { resistance: 1.0, ..PaintProps::default() }
    }

    fn battery(emf: f64) -> PaintProps {
        PaintProps { emf, ..PaintProps::default() }
    }

    fn electrons_per_cell(world: &World) -> Array2<usize> {
        let mut counts = Array2::zeros((world.materials.cols, world.materials.rows));
        for i in 0..world.particles.len() {
            let (c, r) = world.particles.cell(i);
            counts[(c as usize, r as usize)] += 1;
        }
        counts
    }

    fn assert_electrons_inside_conductors(world: &World) {
        for i in 0..world.particles.len() {
            let (c, r) = world.particles.cell(i);
            assert!(world.materials.is_conductive((c as usize, r as usize)));
        }
    }

    #[test]
    fn painting_spawns_n_electrons_per_cell_inside_it() {
        for n in [1, 4] {
            let mut world = make_world(n, 12, 8);
            world.paint(&row(2, 7, 3), Material::Conductor, wire());
            assert_eq!(world.particles.len(), 5 * n as usize);
            let counts = electrons_per_cell(&world);
            assert!((2..7).all(|c| counts[(c, 3)] == n as usize));
            let q = -world.settings.cell_charge / n as f64;
            assert!(world.particles.charge.iter().all(|&c| (c - q).abs() < 1e-12));
            assert!(world.particles.charge.iter().zip(&world.particles.mass).all(|(c, m)| (c / m + 1.0).abs() < 1e-12));
        }
    }

    #[test]
    fn repainting_conductors_keeps_their_electrons() {
        let mut world = make_world(1, 12, 8);
        world.paint(&row(2, 7, 3), Material::Conductor, wire());
        let before = world.particles.pos.clone();
        world.paint(&row(2, 7, 3), Material::Conductor, PaintProps { resistance: 5.0, ..PaintProps::default() });
        assert_eq!(world.particles.pos, before);
    }

    #[test]
    fn painting_a_battery_over_a_wire_keeps_its_charges_and_adds_more() {
        let mut world = make_world(1, 12, 8);
        world.paint(&row(2, 7, 3), Material::Conductor, wire());
        let before = world.particles.pos.clone();
        world.paint(&[(4, 3)], Material::Battery, battery(5.0));
        let s = world.settings.clone();
        assert_eq!(world.particles.pos[..before.len()], before[..]);
        assert_eq!(world.particles.len(), before.len() + ((s.battery_density - 1) * s.carriers_per_cell) as usize);
        assert_eq!(world.stacks.len(), 1);
        assert_eq!(world.materials.resistance[(4, 3)], s.battery_resistance);
        // And back to a wire: the extra electrons leave again.
        world.paint(&row(4, 5, 3), Material::Conductor, wire());
        assert_eq!(world.particles.len(), before.len());
    }

    #[test]
    fn a_painted_battery_gets_a_dense_ion_and_electrons() {
        let mut world = make_world(1, 12, 8);
        world.paint(&[(4, 3)], Material::Battery, battery(5.0));
        let s = world.settings.clone();
        assert_eq!(world.materials.density[(4, 3)], s.battery_density as f64);
        assert_eq!(world.particles.len(), (s.battery_density * s.carriers_per_cell) as usize);
        assert_electrons_inside_conductors(&world);
        world.paint(&[(4, 3)], Material::Vacuum, PaintProps::default());
        assert_eq!(world.particles.len(), 0);
    }

    #[test]
    fn erasing_removes_electrons_of_erased_cells() {
        let mut world = make_world(2, 12, 8);
        world.paint(&row(2, 7, 3), Material::Conductor, wire());
        world.paint(&[(4, 3)], Material::Vacuum, PaintProps::default());
        assert_eq!(world.particles.len(), 8);
        assert_electrons_inside_conductors(&world);
    }

    #[test]
    fn erasing_an_empty_cell_takes_nearest_electron_from_its_island() {
        let mut world = make_world(1, 12, 8);
        world.paint(&row(2, 7, 3), Material::Conductor, wire());
        // Pile all electrons into the leftmost cell, leaving cell 6 empty.
        world.particles.pos.iter_mut().for_each(|p| *p = [2.5, 3.5]);
        world.paint(&[(6, 3)], Material::Vacuum, PaintProps::default());
        assert_eq!(world.particles.len(), 4);
    }

    #[test]
    fn extra_electrons_in_erased_cell_move_to_nearest_cell_of_same_island() {
        let mut world = make_world(1, 12, 8);
        world.paint(&row(2, 7, 3), Material::Conductor, wire());
        world.paint(&[(10, 6)], Material::Conductor, wire()); // A separate island far away.
        world.particles.pos[..5].iter_mut().for_each(|p| *p = [6.5, 3.5]); // Whole row crowds the end cell.
        world.paint(&[(6, 3)], Material::Vacuum, PaintProps::default());
        assert_eq!(world.particles.len(), 5);
        let counts = electrons_per_cell(&world);
        assert_eq!(counts[(5, 3)], 4); // Leftovers landed in the nearest remaining cell...
        assert_eq!(counts[(10, 6)], 1); // ...not in the other island.
        assert_electrons_inside_conductors(&world);
    }

    #[test]
    fn erasing_whole_island_removes_all_its_electrons_even_if_charged() {
        let mut world = make_world(1, 12, 8);
        world.paint(&row(2, 4, 3), Material::Conductor, wire());
        world.paint(&[(9, 6)], Material::Conductor, wire());
        world.particles.pos[2] = [2.5, 3.5]; // Move the other island's electron in (net -1).
        world.paint(&[(2, 3), (3, 3)], Material::Vacuum, PaintProps::default());
        assert_eq!(world.particles.len(), 0);
    }

    #[test]
    fn clear_removes_everything() {
        let mut world = make_world(1, 12, 8);
        world.paint(&row(2, 7, 3), Material::Conductor, wire());
        world.clear();
        assert_eq!(world.particles.len(), 0);
        assert_eq!(world.connectivity.count, 0);
    }
}
