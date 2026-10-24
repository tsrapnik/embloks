//! Putting charges on the grid, and the Poynting vector.
//!
//! The fields themselves are computed in maxwell.rs; close-range corrections between
//! electrons in close_range.rs.

use std::f64::consts::PI;

use ndarray::{Array2, Array3, Axis};

use crate::materials::Materials;
use crate::settings::Settings;

/// Bilinear (cloud-in-cell) weights of particles onto the 4 nearest cell centres.
///
/// Computed once per particle configuration, then reused to deposit charge and
/// current. Corners outside the grid are clamped onto the edge cells, so weights
/// per particle still sum to 1.
pub struct CloudInCell {
    pub cols: usize,
    pub rows: usize,
    /// Per particle: flat cell index (col * rows + row) and weight of each corner.
    pub index: Vec<[usize; 4]>,
    pub weight: Vec<[f64; 4]>,
}

impl CloudInCell {
    pub fn new(positions: &[[f64; 2]], cols: usize, rows: usize) -> Self {
        let clip = |v: i64, n: usize| v.clamp(0, n as i64 - 1) as usize;
        let (mut index, mut weight) = (Vec::with_capacity(positions.len()), Vec::with_capacity(positions.len()));
        for p in positions {
            let (u, v) = (p[0] - 0.5, p[1] - 0.5);
            let (c0, r0) = (u.floor() as i64, v.floor() as i64);
            let (fu, fv) = (u - c0 as f64, v - r0 as f64);
            let (ca, cb) = (clip(c0, cols), clip(c0 + 1, cols));
            let (ra, rb) = (clip(r0, rows), clip(r0 + 1, rows));
            index.push([ca * rows + ra, ca * rows + rb, cb * rows + ra, cb * rows + rb]);
            weight.push([(1.0 - fu) * (1.0 - fv), (1.0 - fu) * fv, fu * (1.0 - fv), fu * fv]);
        }
        Self { cols, rows, index, weight }
    }

    /// Spread a value per particle over the grid (cols, rows). Conserves the sum.
    pub fn deposit(&self, values: &[f64]) -> Array2<f64> {
        let mut flat = vec![0.0; self.cols * self.rows];
        for ((index, weight), value) in self.index.iter().zip(&self.weight).zip(values) {
            for k in 0..4 {
                flat[index[k]] += weight[k] * value;
            }
        }
        Array2::from_shape_vec((self.cols, self.rows), flat).unwrap()
    }

    /// Spread a vector per particle over the grid (cols, rows, 2). Conserves the sum.
    pub fn deposit_vectors(&self, values: &[[f64; 2]]) -> Array3<f64> {
        let mut flat = vec![0.0; self.cols * self.rows * 2];
        for ((index, weight), value) in self.index.iter().zip(&self.weight).zip(values) {
            for k in 0..4 {
                flat[2 * index[k]] += weight[k] * value[0];
                flat[2 * index[k] + 1] += weight[k] * value[1];
            }
        }
        Array3::from_shape_vec((self.cols, self.rows, 2), flat).unwrap()
    }

    /// Read a (cols, rows, 2) grid at the particles.
    pub fn gather(&self, grid: &Array3<f64>) -> Vec<[f64; 2]> {
        let flat = grid.as_standard_layout();
        let flat = flat.as_slice().unwrap();
        self.index
            .iter()
            .zip(&self.weight)
            .map(|(index, weight)| {
                let mut out = [0.0; 2];
                for k in 0..4 {
                    out[0] += weight[k] * flat[2 * index[k]];
                    out[1] += weight[k] * flat[2 * index[k] + 1];
                }
                out
            })
            .collect()
    }
}

/// Deposit each cell's value (cols, rows) as if spread evenly over its cell.
///
/// This is the cloud-in-cell deposit of a uniform square: per axis 3/4 stays on
/// the cell's centre and 1/8 goes to each neighbour. Ions are deposited this way,
/// so electrons evenly filling their cells cancel them, rather than only electrons
/// stacked on the centre. Parts beyond the grid edge stay on the edge cell, as in
/// CloudInCell, so the sum is conserved.
pub fn spread_over_cells(values: &Array2<f64>) -> Array2<f64> {
    let mut values = values.clone();
    for axis in [Axis(0), Axis(1)] {
        let n = values.len_of(axis);
        let mut out = values.mapv(|v| 0.75 * v);
        for i in 0..n {
            let here = values.index_axis(axis, i).to_owned() / 8.0;
            // Each neighbour gets 1/8; beyond the edge it stays on the edge cell.
            let below = if i > 0 { i - 1 } else { i };
            let above = if i + 1 < n { i + 1 } else { i };
            out.index_axis_mut(axis, below).scaled_add(1.0, &here);
            out.index_axis_mut(axis, above).scaled_add(1.0, &here);
        }
        values = out;
    }
    values
}

/// Charge of the fixed ions on the cell centres (cols, rows): spread over their cells.
pub fn ion_charge(materials: &Materials, cell_charge: f64) -> Array2<f64> {
    spread_over_cells(&materials.density.mapv(|d| d * cell_charge))
}

/// Poynting vector S = E x B / mu0 per cell, shape (cols, rows, 2): energy flow.
///
/// With E in the plane and B = (0, 0, B_z), E x B = (E_y B_z, -E_x B_z) is back
/// in the plane. 1/mu0 = c^2 / (4 pi k) in our units.
pub fn poynting(e: &Array3<f64>, bz: &Array2<f64>, settings: &Settings) -> Array3<f64> {
    let inv_mu0 = settings.speed_of_light.powi(2) / (4.0 * PI * settings.coulomb_k);
    let (cols, rows) = bz.dim();
    Array3::from_shape_fn((cols, rows, 2), |(c, r, k)| {
        let b = bz[(c, r)];
        if k == 0 { inv_mu0 * e[(c, r, 1)] * b } else { -inv_mu0 * e[(c, r, 0)] * b }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::{Material, PaintProps};
    use crate::maxwell::Maxwell;
    use crate::test_util::{assert_rel, normal, uniform};
    use crate::world::World;

    fn world() -> World {
        World::new(20, 14, Settings::default(), 0)
    }

    #[test]
    fn deposit_conserves_charge_and_is_exact_on_centres() {
        let positions = uniform(3, 50, 5.0); // Includes points near the edges.
        let charges = normal(4, 50);
        let total: f64 = charges.iter().sum();
        assert_rel(CloudInCell::new(&positions, 5, 5).deposit(&charges).sum(), total, 1e-9);
        let on_centre = CloudInCell::new(&[[2.5, 1.5]], 5, 5).deposit(&[3.0]);
        assert_eq!(on_centre[(2, 1)], 3.0);
        assert_eq!(on_centre.sum(), 3.0);
    }

    #[test]
    fn deposit_of_vectors_matches_one_component_at_a_time() {
        let cic = CloudInCell::new(&uniform(5, 30, 4.0), 6, 4);
        let values: Vec<[f64; 2]> = normal(6, 30).iter().zip(normal(7, 30)).map(|(&a, b)| [a, b]).collect();
        let both = cic.deposit_vectors(&values);
        assert_eq!(both.dim(), (6, 4, 2));
        for k in 0..2 {
            let single = cic.deposit(&values.iter().map(|v| v[k]).collect::<Vec<_>>());
            for ((c, r), v) in single.indexed_iter() {
                assert!((both[(c, r, k)] - v).abs() < 1e-12);
            }
        }
        assert_eq!(CloudInCell::new(&[], 6, 4).deposit(&[]).dim(), (6, 4));
    }

    #[test]
    fn gather_reads_back_grid_values_and_is_bilinear() {
        let mut grid = Array3::zeros((4, 3, 2));
        grid[(1, 1, 0)] = 2.0;
        grid[(1, 1, 1)] = -4.0;
        grid[(2, 1, 0)] = 4.0;
        let gathered = CloudInCell::new(&[[1.5, 1.5], [2.0, 1.5]], 4, 3).gather(&grid);
        assert_eq!(gathered, vec![[2.0, -4.0], [3.0, -2.0]]);
    }

    #[test]
    fn neutral_cell_has_weak_far_field() {
        let mut world = world();
        world.paint(&[(10, 7)], Material::Conductor, PaintProps { resistance: 1.0, ..PaintProps::default() });
        let (e, _) = Maxwell::for_world(&world).node_fields();
        let far = e[(0, 0, 0)].hypot(e[(0, 0, 1)]);
        // A lone ion's charge, +cell_charge as a line charge: |E| = 2 k q / r.
        let lone_ion = 2.0 * world.settings.cell_charge / 10f64.hypot(7.0);
        // The electrons (within its cell) cancel almost all of the ion's field from afar.
        assert!(far < 0.05 * lone_ion);
    }

    #[test]
    fn place_charges_replaces_and_remove_clears() {
        let mut world = world();
        world.place_charges(&[(1, 1), (2, 1), (2, 1)], 1.0); // Duplicate cell in the stroke.
        assert_eq!(world.fixed_charge.len(), 2);
        world.place_charges(&[(2, 1)], -5.0);
        let mut charges = world.fixed_charge.clone();
        charges.sort_by(f64::total_cmp);
        assert_eq!(charges, vec![-5.0, 1.0]);
        world.remove_charges(&[(1, 1)]);
        assert_eq!(world.fixed_charge, vec![-5.0]);
        assert_eq!(world.fixed_pos, vec![[2.5, 1.5]]);
    }

    #[test]
    fn neutral_conductor_with_evenly_spread_electrons_has_no_grid_field() {
        // Ions are spread over their cells, so electrons evenly filling them cancel them.
        let mut world = world();
        world.settings.carriers_per_cell = 16;
        let cells: Vec<_> = (4..16).flat_map(|c| (4..10).map(move |r| (c, r))).collect();
        world.paint(&cells, Material::Conductor, PaintProps { resistance: 1.0, ..PaintProps::default() });
        let sub: Vec<f64> = (0..4).map(|i| (i as f64 + 0.5) / 4.0).collect();
        let mut positions = vec![];
        for ((c, r), _) in world.materials.conductive().indexed_iter().filter(|(_, v)| **v) {
            for &x in &sub {
                for &y in &sub {
                    positions.push([c as f64 + x, r as f64 + y]);
                }
            }
        }
        world.particles.pos = positions;
        let (e, _) = Maxwell::for_world(&world).node_fields();
        assert!(e.iter().all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn spread_over_cells_conserves_charge() {
        let mut values = Array2::zeros((5, 4));
        values[(2, 1)] = 64.0;
        values[(0, 3)] = 8.0; // Corner cell: the parts beyond the edge stay on it.
        let spread = spread_over_cells(&values);
        assert_rel(spread.sum(), 72.0, 1e-12);
        assert_rel(spread[(2, 1)], 36.0, 1e-12); // (3/4)^2
        assert_rel(spread[(1, 1)], 6.0, 1e-12); // 3/4 * 1/8
        assert_rel(spread[(1, 0)], 1.0, 1e-12); // (1/8)^2
        assert_rel(spread[(0, 3)], 8.0 * (7.0f64 / 8.0).powi(2), 1e-12);
    }

    #[test]
    fn poynting_is_perpendicular_to_e_with_magnitude_e_b_over_mu0() {
        let s = Settings::default();
        let e = Array3::from_shape_vec((6, 5, 2), normal(2, 60)).unwrap();
        let bz = Array2::from_shape_vec((6, 5), normal(3, 30)).unwrap();
        let flow = poynting(&e, &bz, &s);
        let inv_mu0 = s.speed_of_light.powi(2) / (4.0 * PI * s.coulomb_k);
        for ((c, r), b) in bz.indexed_iter() {
            let (fx, fy, ex, ey) = (flow[(c, r, 0)], flow[(c, r, 1)], e[(c, r, 0)], e[(c, r, 1)]);
            assert!((fx * ex + fy * ey).abs() < 1e-9);
            assert_rel(fx.hypot(fy), inv_mu0 * ex.hypot(ey) * b.abs(), 1e-9);
        }
    }

    #[test]
    fn energy_flows_into_a_current_carrying_wire_from_both_sides() {
        let s = Settings::default();
        let mut current = Array3::zeros((80, 60, 2));
        for c in 0..80 {
            current[(c, 30, 0)] = 1.0; // Wire along row 30, conventional current to the right...
        }
        let mut e = Array3::zeros((80, 60, 2));
        e.index_axis_mut(Axis(2), 0).fill(0.1); // ...driven by a field pointing the same way.
        let flow = poynting(&e, &crate::maxwell::tests::steady_b(&current, &s, 20.0), &s);
        assert!(flow[(40, 25, 1)] > 0.0); // Above the wire (smaller y): pointing down, into it.
        assert!(flow[(40, 35, 1)] < 0.0); // Below the wire: pointing up, into it.
        assert!(flow[(40, 25, 0)].abs() < 1e-9 && flow[(40, 35, 0)].abs() < 1e-9);
    }
}
