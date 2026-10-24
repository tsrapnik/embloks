//! Close-range forces between electrons, and between electrons and ions.
//!
//! The grid fields (maxwell.rs) smooth every charge over about a cell, so two
//! electrons barely repel within a cell and can sit on top of each other. Within
//! electron_cutoff this adds what the grid leaves out: electrons repel as compact
//! line charges (smeared over only electron_softening), and ions attract them as
//! charge spread evenly over their cells. Electrons evenly filling a conductor then
//! feel no net close-range force, as in a real metal; crowding or gaps get pushed
//! back. This is the particle-particle part of a P3M scheme, with the Maxwell grid as
//! its particle-mesh part.
//!
//! Everything is tabulated once: the force the grid really applies between two
//! charges (measured with the Maxwell code itself, averaged over where the charges
//! sit in their cells), the correction for a point charge, and that correction
//! integrated over a uniform square (an ion). The same tables drive the GPU kernels.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex, OnceLock};

use ndarray::{Array2, Array3, Axis, s};

use crate::fields::CloudInCell;
use crate::materials::Materials;
use crate::maxwell::Maxwell;
use crate::settings::Settings;
use crate::world::World;

/// Samples (uniform in r^2, cutoff included) of the radial point-charge tables.
pub const PAIR_SAMPLES: usize = 4096;
/// Samples per cell (per axis) of the table of a uniform square.
pub const SQUARE_SAMPLES: usize = 32;
// Measuring the grid's force: source positions per axis within a cell, directions,
// and radial spacing (cells).
const SOURCE_SAMPLES: usize = 8;
const DIRECTIONS: usize = 16;
const RADIAL_STEP: f64 = 0.01;
/// Below this r^2 / softening^2 the compact kernel uses its Taylor series.
const SERIES_BELOW: f64 = 1e-4;

/// g with field g d at offset d from a unit line charge smeared as a Gaussian.
///
/// In 2D (div E = 4 pi rho) a Gaussian of width eps has the field
/// 2 d / r^2 (1 - exp(-r^2 / eps^2)): plain 2 / r far away, zero at r = 0.
pub fn compact_g(r2: f64, softening: f64) -> f64 {
    let x = r2 / (softening * softening);
    let value = if x < SERIES_BELOW { 2.0 * (1.0 - x / 2.0 + x * x / 6.0) } else { -2.0 * (-x).exp_m1() / x };
    value / (softening * softening)
}

fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    (0..n).map(|i| start + (end - start) * i as f64 / (n - 1) as f64).collect()
}

/// numpy.interp: piecewise linear through (xs, ys), clamped at the ends.
fn interp(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[xs.len() - 1] {
        return ys[ys.len() - 1];
    }
    let i = xs.partition_point(|&v| v <= x) - 1;
    ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i])
}

/// Radii (0 .. reach) and the radial force between two unit charges through the grid.
///
/// Measured with the Maxwell code: one charge deposited, smoothed, its static
/// field solved and read back at the other charge. Averaged over where the
/// source sits in its cell and over directions, since the grid's force depends
/// a little on both.
pub fn grid_force(passes: usize, reach: f64) -> (Vec<f64>, Vec<f64>) {
    let n = 2 * reach.ceil() as usize + 24;
    let settings = Settings { smoothing_passes: passes, coulomb_k: 1.0, ..Settings::default() };
    let mut maxwell = Maxwell::new(n, n, &settings);
    let radii = linspace(0.0, reach, (reach / RADIAL_STEP).round() as usize + 1);
    let directions: Vec<[f64; 2]> = (0..DIRECTIONS)
        .map(|k| {
            let angle = 2.0 * PI * (k as f64 + 0.5) / DIRECTIONS as f64;
            [angle.cos(), angle.sin()]
        })
        .collect();
    let mut total = vec![0.0; radii.len()];
    for i in 0..SOURCE_SAMPLES {
        for j in 0..SOURCE_SAMPLES {
            let source = [
                (n / 2) as f64 + (i as f64 + 0.5) / SOURCE_SAMPLES as f64,
                (n / 2) as f64 + (j as f64 + 0.5) / SOURCE_SAMPLES as f64,
            ];
            maxwell.ex.fill(0.0);
            maxwell.ey.fill(0.0);
            let rho = maxwell.smoothed2(&CloudInCell::new(&[source], n, n).deposit(&[1.0]));
            maxwell.enforce_gauss(&rho);
            let points: Vec<[f64; 2]> = radii
                .iter()
                .flat_map(|&r| directions.iter().map(move |u| [source[0] + r * u[0], source[1] + r * u[1]]))
                .collect();
            let field = maxwell.gather(&points);
            for (k, radial) in field.chunks(DIRECTIONS).enumerate() {
                let along: f64 = radial.iter().zip(&directions).map(|(f, u)| f[0] * u[0] + f[1] * u[1]).sum();
                total[k] += along / DIRECTIONS as f64;
            }
        }
    }
    let samples = (SOURCE_SAMPLES * SOURCE_SAMPLES) as f64;
    (radii, total.into_iter().map(|t| t / samples).collect())
}

/// Everything the close-range forces need, for one set of parameters.
#[derive(Debug)]
pub struct Tables {
    /// The close-range correction of a unit charge: g (field g d) and its potential,
    /// sampled at r^2 = cutoff^2 i / PAIR_SAMPLES, plus a zero sample past the cutoff.
    pub g: Vec<f64>,
    pub potential: Vec<f64>,
    /// point_correction of a unit charge spread evenly over a cell, (n, n, 3): potential
    /// and field over offsets from the cell centre in [-half, half]^2.
    pub square: Array3<f64>,
    pub half: f64,
    /// Close-range potential of a uniform unit square, averaged over the cell at each
    /// offset (dc, dr) from it, shape (2 reach + 1,) * 2.
    pub ion_stencil: Array2<f64>,
}

type TableKey = (u64, usize, u64);

/// The tables for (softening, passes, cutoff), computed once per process.
pub fn tables(softening: f64, passes: usize, cutoff: f64) -> Arc<Tables> {
    static CACHE: OnceLock<Mutex<HashMap<TableKey, Arc<Tables>>>> = OnceLock::new();
    let key = (softening.to_bits(), passes, cutoff.to_bits());
    let cache = CACHE.get_or_init(Default::default);
    if let Some(tables) = cache.lock().unwrap().get(&key) {
        return tables.clone();
    }
    let tables = Arc::new(build_tables(softening, passes, cutoff));
    cache.lock().unwrap().insert(key, tables.clone());
    tables
}

fn build_tables(softening: f64, passes: usize, cutoff: f64) -> Tables {
    let (g, potential) = radial_tables(softening, passes, cutoff);
    let (square, half) = square_table(&g, &potential, cutoff);
    let ion_stencil = ion_stencil(&square, half, cutoff);
    Tables { g, potential, square, half, ion_stencil }
}

/// The correction is the compact force minus the grid's. Its potential is the
/// integral of that force inwards from the cutoff, so it is zero there and the
/// field is exactly its gradient.
pub fn radial_tables(softening: f64, passes: usize, cutoff: f64) -> (Vec<f64>, Vec<f64>) {
    let (r, f_grid) = grid_force(passes, cutoff);
    let f_near: Vec<f64> = r.iter().zip(&f_grid).map(|(&r, &f)| r * compact_g(r * r, softening) - f).collect();
    let mut potential = vec![0.0; r.len()];
    for i in (0..r.len() - 1).rev() {
        potential[i] = potential[i + 1] + 0.5 * (f_near[i + 1] + f_near[i]) * (r[i + 1] - r[i]);
    }
    let r2 = linspace(0.0, cutoff * cutoff, PAIR_SAMPLES + 1);
    let mut g = Vec::with_capacity(r2.len() + 1);
    let mut pot = Vec::with_capacity(r2.len() + 1);
    for (i, &r2) in r2.iter().enumerate() {
        let rs = r2.sqrt();
        // The grid's force grows linearly from r = 0, so f / r is finite there.
        let grid_g = if i == 0 { f_grid[1] / r[1] } else { interp(rs, &r, &f_grid) / rs };
        g.push(compact_g(r2, softening) - grid_g);
        pot.push(interp(rs, &r, &potential));
    }
    g.push(0.0);
    pot.push(0.0);
    (g, pot)
}

/// Linear interpolation in a radial table at squared distance r2; zero beyond the cutoff.
pub fn radial(table: &[f64], r2: f64, cutoff: f64) -> f64 {
    if r2 >= cutoff * cutoff {
        return 0.0;
    }
    let x = (r2 * (PAIR_SAMPLES as f64 / (cutoff * cutoff))).min(PAIR_SAMPLES as f64);
    let i = x as usize;
    table[i] + (table[i + 1] - table[i]) * (x - i as f64)
}

/// Close-range potential and field of a unit charge at offset d.
pub fn point_correction(d: [f64; 2], g: &[f64], potential: &[f64], cutoff: f64) -> [f64; 3] {
    let r2 = d[0] * d[0] + d[1] * d[1];
    let gr = radial(g, r2, cutoff);
    [radial(potential, r2, cutoff), gr * d[0], gr * d[1]]
}

/// point_correction of a unit charge spread evenly over a cell, tabulated.
///
/// Returns the table (n, n, 3) over offsets from the cell centre in [-half, half]^2
/// (spacing 1 / SQUARE_SAMPLES), and `half`. The square is integrated with the
/// midpoint rule: a sliding sum of the point-charge values along each axis.
fn square_table(g: &[f64], potential: &[f64], cutoff: f64) -> (Array3<f64>, f64) {
    let m = SQUARE_SAMPLES;
    let h = 1.0 / m as f64;
    let half = cutoff.ceil() + 1.0;
    // Point values at offsets z_k = -half - 0.5 + (k + 0.5) h; table entry i
    // (offset -half + i h) sums z_i .. z_(i + m - 1) along each axis.
    let count = ((2.0 * half + 1.0) * m as f64).round() as usize;
    let z: Vec<f64> = (0..count).map(|k| -half - 0.5 + (k as f64 + 0.5) * h).collect();
    // Summed-area table with a leading row and column of zeros.
    let mut total = Array3::<f64>::zeros((count + 1, count + 1, 3));
    for a in 0..count {
        for b in 0..count {
            let point = point_correction([z[a], z[b]], g, potential, cutoff);
            for k in 0..3 {
                total[(a + 1, b + 1, k)] =
                    point[k] + total[(a, b + 1, k)] + total[(a + 1, b, k)] - total[(a, b, k)];
            }
        }
    }
    let n = count - m + 1;
    let table = Array3::from_shape_fn((n, n, 3), |(i, j, k)| {
        (total[(i + m, j + m, k)] - total[(i, j + m, k)] - total[(i + m, j, k)] + total[(i, j, k)]) * h * h
    });
    (table, half)
}

/// The square's potential averaged over the cell at each offset (dc, dr) from it
/// (trapezoid rule on the table).
fn ion_stencil(square: &Array3<f64>, half: f64, cutoff: f64) -> Array2<f64> {
    let m = SQUARE_SAMPLES;
    let reach = cutoff.ceil() as i64;
    let mut weights = vec![1.0 / m as f64; m + 1];
    weights[0] *= 0.5;
    weights[m] *= 0.5;
    let size = (2 * reach + 1) as usize;
    Array2::from_shape_fn((size, size), |(i, j)| {
        let (dc, dr) = (i as i64 - reach, j as i64 - reach);
        let a = ((dc as f64 - 0.5 + half) * m as f64).round() as usize;
        let b = ((dr as f64 - 0.5 + half) * m as f64).round() as usize;
        let mut sum = 0.0;
        for (x, wx) in weights.iter().enumerate() {
            for (y, wy) in weights.iter().enumerate() {
                sum += wx * square[(a + x, b + y, 0)] * wy;
            }
        }
        sum
    })
}

/// Bilinear interpolation in channel `k` of a table (n, n, 3) at fractional indices.
fn table_bilinear(table: &Array3<f64>, k: usize, u: f64, v: f64) -> f64 {
    let n = table.dim().0;
    let a = (u.floor() as i64).clamp(0, n as i64 - 2) as usize;
    let b = (v.floor() as i64).clamp(0, n as i64 - 2) as usize;
    let (fu, fv) = (u - a as f64, v - b as f64);
    table[(a, b, k)] * (1.0 - fu) * (1.0 - fv)
        + table[(a + 1, b, k)] * fu * (1.0 - fv)
        + table[(a, b + 1, k)] * (1.0 - fu) * fv
        + table[(a + 1, b + 1, k)] * fu * fv
}

/// Electrons sorted by cell (a counting sort): `order` lists particle indices cell by
/// cell, and cell i's particles are order[start[i] .. start[i + 1]].
pub struct CellList {
    pub cell: Vec<usize>,
    pub start: Vec<usize>,
    pub order: Vec<usize>,
}

impl CellList {
    pub fn new(pos: &[[f64; 2]], cols: usize, rows: usize) -> Self {
        let n = pos.len();
        let mut cell = Vec::with_capacity(n);
        let mut start = vec![0; cols * rows + 1];
        for p in pos {
            let c = (p[0].floor() as i64).clamp(0, cols as i64 - 1) as usize;
            let r = (p[1].floor() as i64).clamp(0, rows as i64 - 1) as usize;
            cell.push(c * rows + r);
            start[c * rows + r + 1] += 1;
        }
        for i in 0..cols * rows {
            start[i + 1] += start[i];
        }
        let mut fill = start[..cols * rows].to_vec();
        let mut order = vec![0; n];
        for (i, &c) in cell.iter().enumerate() {
            order[fill[c]] = i;
            fill[c] += 1;
        }
        Self { cell, start, order }
    }
}

/// Close-range forces on electrons and their potential, on top of the grid fields.
#[derive(Clone)]
pub struct CloseRange {
    pub settings: Settings,
    ion_cache: Option<((u64, usize, usize), Array2<f64>)>,
}

impl CloseRange {
    pub fn new(settings: &Settings) -> Self {
        Self { settings: settings.clone(), ion_cache: None }
    }

    pub fn enabled(&self) -> bool {
        self.settings.electron_cutoff > 0.0
    }

    pub fn tables(&self) -> Arc<Tables> {
        let s = &self.settings;
        tables(s.electron_softening, s.smoothing_passes, s.electron_cutoff)
    }

    /// Close-range force on each electron. Opposite and equal between electrons, so
    /// momentum is conserved.
    ///
    /// Electrons are sorted by cell, so each one only checks the cells within reach.
    /// Every pair is handled once, from the electron in the lower-numbered cell.
    pub fn force(&self, world: &World) -> Vec<[f64; 2]> {
        let s = &self.settings;
        let t = self.tables();
        let (pos, charge) = (&world.particles.pos, &world.particles.charge);
        let ions = &world.materials.density;
        let (cols, rows) = ions.dim();
        let (k, k_ion) = (s.coulomb_k, s.coulomb_k * s.cell_charge);
        let cutoff = s.electron_cutoff;
        let reach = cutoff.ceil() as usize;
        let cutoff2 = cutoff * cutoff;
        let scale = (t.g.len() - 2) as f64 / cutoff2;
        let m = t.square.dim().0;
        let samples = (m - 1) as f64 / (2.0 * t.half);
        let list = CellList::new(pos, cols, rows);

        let mut force = vec![[0.0; 2]; pos.len()];
        for i in 0..pos.len() {
            let [x, y] = pos[i];
            let (ci, ri) = (list.cell[i] / rows, list.cell[i] % rows);
            for c in ci.saturating_sub(reach)..(ci + reach + 1).min(cols) {
                for r in ri.saturating_sub(reach)..(ri + reach + 1).min(rows) {
                    if ions[(c, r)] > 0.0 {
                        // The ion of cell (c, r): the square table at the offset from its centre.
                        let u = (x - c as f64 - 0.5 + t.half) * samples;
                        let v = (y - r as f64 - 0.5 + t.half) * samples;
                        let a = (u as i64).clamp(0, m as i64 - 2) as usize;
                        let b = (v as i64).clamp(0, m as i64 - 2) as usize;
                        let (fu, fv) = (u - a as f64, v - b as f64);
                        for axis in 0..2 {
                            let sq = &t.square;
                            let value = sq[(a, b, axis + 1)] * (1.0 - fu) * (1.0 - fv)
                                + sq[(a + 1, b, axis + 1)] * fu * (1.0 - fv)
                                + sq[(a, b + 1, axis + 1)] * (1.0 - fu) * fv
                                + sq[(a + 1, b + 1, axis + 1)] * fu * fv;
                            force[i][axis] += k_ion * ions[(c, r)] * charge[i] * value;
                        }
                    }
                    let other = c * rows + r;
                    if other < list.cell[i] {
                        continue;
                    }
                    for &j in &list.order[list.start[other]..list.start[other + 1]] {
                        if other == list.cell[i] && j <= i {
                            continue;
                        }
                        let (dx, dy) = (x - pos[j][0], y - pos[j][1]);
                        let d2 = dx * dx + dy * dy;
                        if d2 >= cutoff2 {
                            continue;
                        }
                        let tt = d2 * scale;
                        let si = tt as usize;
                        let f = k * charge[i] * charge[j] * (t.g[si] + (t.g[si + 1] - t.g[si]) * (tt - si as f64));
                        force[i][0] += f * dx;
                        force[i][1] += f * dy;
                        force[j][0] -= f * dx;
                        force[j][1] -= f * dy;
                    }
                }
            }
        }
        force
    }

    /// Close-range potential of all ions, averaged over each cell (cols, rows).
    pub fn ion_potential(&mut self, materials: &Materials) -> &Array2<f64> {
        let key = (materials.version, materials.cols, materials.rows);
        if self.ion_cache.as_ref().is_none_or(|(k, _)| *k != key) {
            let s = &self.settings;
            let stencil = &self.tables().ion_stencil;
            let reach = (stencil.dim().0 / 2) as i64;
            let density = &materials.density;
            let (cols, rows) = density.dim();
            let scale = s.coulomb_k * s.cell_charge;
            let ions = Array2::from_shape_fn((cols, rows), |(c, r)| {
                let mut sum = 0.0;
                for dc in -reach..=reach {
                    for dr in -reach..=reach {
                        let (x, y) = (c as i64 + dc, r as i64 + dr);
                        if materials.in_bounds(x, y) {
                            sum += density[(x as usize, y as usize)] * stencil[((dc + reach) as usize, (dr + reach) as usize)];
                        }
                    }
                }
                scale * sum
            });
            self.ion_cache = Some((key, ions));
        }
        &self.ion_cache.as_ref().unwrap().1
    }

    /// Close-range part of the potential, averaged over each cell (cols, rows).
    ///
    /// The potential whose gradient `force` adds. Batteries need it: they hold the
    /// potential difference between their terminals, and electrons crowded at the
    /// minus terminal push harder than the grid potential alone shows. Averaged over
    /// the cell rather than read at its centre, which is sharp: it would jump with
    /// every electron passing close by.
    ///
    /// `where` (cols, rows) limits the work to the cells that will be read; the
    /// result is zero elsewhere.
    pub fn potential(&mut self, world: &World, where_: Option<&Array2<bool>>) -> Array2<f64> {
        let ions = self.ion_potential(&world.materials).clone();
        let charges = self.charges_potential(&world.materials, &world.particles.pos, &world.particles.charge, where_);
        let mut potential = ions + charges;
        if let Some(mask) = where_ {
            potential.zip_mut_with(mask, |v, &keep| if !keep { *v = 0.0 });
        }
        potential
    }

    /// Close-range potential of point charges like electrons, averaged over each cell
    /// (cols, rows). With `where_`, only exact on those cells.
    pub fn charges_potential(
        &self,
        materials: &Materials,
        pos: &[[f64; 2]],
        charge: &[f64],
        where_: Option<&Array2<bool>>,
    ) -> Array2<f64> {
        let s = &self.settings;
        let t = self.tables();
        let (cols, rows) = (materials.cols, materials.rows);
        let reach = s.electron_cutoff.ceil() as i64;
        // Only charges within reach of a requested cell contribute to it.
        let near = where_.map(|mask| dilate(mask, reach));
        let mut total = Array2::zeros((cols, rows));
        for (p, &q) in pos.iter().zip(charge) {
            let (pc, pr) = (p[0].floor() as i64, p[1].floor() as i64);
            if let Some(near) = &near
                && !near[(pc.clamp(0, cols as i64 - 1) as usize, pr.clamp(0, rows as i64 - 1) as usize)]
            {
                continue;
            }
            for dc in -reach..=reach {
                for dr in -reach..=reach {
                    let (c, r) = (pc + dc, pr + dr);
                    if !materials.in_bounds(c, r) {
                        continue;
                    }
                    // A charge's potential averaged over a cell is that of the cell, spread
                    // evenly, at the charge: the square table.
                    let d = [p[0] - (c as f64 + 0.5), p[1] - (r as f64 + 0.5)];
                    let u = (d[0] + t.half) * SQUARE_SAMPLES as f64;
                    let v = (d[1] + t.half) * SQUARE_SAMPLES as f64;
                    total[(c as usize, r as usize)] += table_bilinear(&t.square, 0, u, v) * q;
                }
            }
        }
        total * s.coulomb_k
    }
}

/// Cells within `reach` (Chebyshev distance) of a true cell.
fn dilate(mask: &Array2<bool>, reach: i64) -> Array2<bool> {
    let (cols, rows) = mask.dim();
    let mut out = Array2::from_elem((cols, rows), false);
    for ((c, r), _) in mask.indexed_iter().filter(|(_, v)| **v) {
        let (c0, c1) = ((c as i64 - reach).max(0) as usize, ((c as i64 + reach + 1) as usize).min(cols));
        let (r0, r1) = ((r as i64 - reach).max(0) as usize, ((r as i64 + reach + 1) as usize).min(rows));
        out.slice_mut(s![c0..c1, r0..r1]).fill(true);
    }
    out
}

/// The square table's channel `k` as a contiguous (n, n) array.
pub fn square_channel(tables: &Tables, k: usize) -> Array2<f64> {
    tables.square.index_axis(Axis(2), k).to_owned()
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::materials::{Material, PaintProps};
    use crate::test_util::{assert_rel, mean};
    use rand::{RngExt, SeedableRng};
    use rand_pcg::Pcg64;

    /// CloseRange::force the slow way: all pairs, and a sum over every ion.
    fn reference_force(world: &World, s: &Settings) -> Vec<[f64; 2]> {
        let t = tables(s.electron_softening, s.smoothing_passes, s.electron_cutoff);
        let p = &world.particles;
        let mut force = vec![[0.0; 2]; p.len()];
        for i in 0..p.len() {
            for j in i + 1..p.len() {
                let d = [p.pos[i][0] - p.pos[j][0], p.pos[i][1] - p.pos[j][1]];
                if d[0].hypot(d[1]) >= s.electron_cutoff {
                    continue;
                }
                let pc = point_correction(d, &t.g, &t.potential, s.electron_cutoff);
                for axis in 0..2 {
                    let pair = s.coulomb_k * p.charge[i] * p.charge[j] * pc[axis + 1];
                    force[i][axis] += pair;
                    force[j][axis] -= pair;
                }
            }
        }
        for ((c, r), _) in world.materials.conductive().indexed_iter().filter(|(_, v)| **v) {
            for i in 0..p.len() {
                let d = [p.pos[i][0] - (c as f64 + 0.5), p.pos[i][1] - (r as f64 + 0.5)];
                if d[0].abs() >= t.half || d[1].abs() >= t.half {
                    continue;
                }
                let (u, v) = ((d[0] + t.half) * SQUARE_SAMPLES as f64, (d[1] + t.half) * SQUARE_SAMPLES as f64);
                for axis in 0..2 {
                    force[i][axis] += s.coulomb_k * s.cell_charge * p.charge[i] * table_bilinear(&t.square, axis + 1, u, v);
                }
            }
        }
        force
    }

    #[test]
    fn compiled_force_matches_the_reference() {
        let s = Settings::default();
        let mut w = World::new(30, 20, s.clone(), 1);
        let mut rng = Pcg64::seed_from_u64(2);
        let cells: Vec<_> = (0..30).flat_map(|c| (0..20).map(move |r| (c, r))).filter(|_| rng.random::<f64>() < 0.5).collect();
        w.paint(&cells, Material::Conductor, PaintProps { resistance: 1.0, ..PaintProps::default() });
        let mut rng = Pcg64::seed_from_u64(3);
        for p in w.particles.pos.iter_mut() {
            p[0] = (p[0] + rng.random_range(-0.3..0.3)).clamp(0.001, 29.999);
            p[1] = (p[1] + rng.random_range(-0.3..0.3)).clamp(0.001, 19.999);
        }
        let force = CloseRange::new(&s).force(&w);
        let reference = reference_force(&w, &s);
        let largest = reference.iter().flatten().fold(0.0f64, |a, b| a.max(b.abs()));
        let error = force.iter().flatten().zip(reference.iter().flatten()).fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
        assert!(error < 1e-12 * largest, "error {error}, largest {largest}");
    }

    #[test]
    fn grid_plus_close_range_gives_the_compact_force() {
        // Two electrons: the Maxwell grid's force plus the correction follows a compact
        // line charge within the cutoff, on average over where they sit in their cells.
        let s = Settings::default();
        let mut w = World::new(24, 24, s.clone(), 0);
        w.particles.add(&[[0.0; 2]; 2], -1.0, 1.0);
        let mut rng = Pcg64::seed_from_u64(0);
        let close = CloseRange::new(&s);
        for r in [0.25, 0.5, 1.0, 2.0, 2.8] {
            let mut totals = vec![];
            for _ in 0..40 {
                let a = [rng.random_range(10.0..13.0), rng.random_range(10.0..13.0)];
                let angle: f64 = rng.random_range(0.0..2.0 * PI);
                let u = [angle.cos(), angle.sin()];
                w.particles.pos = vec![a, [a[0] + r * u[0], a[1] + r * u[1]]];
                let grid = Maxwell::for_world(&w).gather(&w.particles.pos)[1];
                let close_force = close.force(&w)[1];
                totals.push((-grid[0] + close_force[0]) * u[0] + (-grid[1] + close_force[1]) * u[1]);
            }
            assert_rel(mean(&totals), r * compact_g(r * r, s.electron_softening), 0.05);
        }
    }

    #[test]
    fn close_range_force_pushes_stacked_electrons_apart() {
        let s = Settings::default();
        let mut w = World::new(10, 10, s.clone(), 0);
        w.paint(&[(4, 4), (5, 4)], Material::Conductor, PaintProps { resistance: 1.0, ..PaintProps::default() });
        w.particles.pos.iter_mut().for_each(|p| *p = [5.0, 4.5]);
        w.particles.pos[0][0] -= 0.05;
        w.particles.pos[1][0] += 0.05;
        let force = CloseRange::new(&s).force(&w);
        // The pair repels (and momentum is conserved between electrons)...
        assert!(force[0][0] < -0.5 && force[1][0] > 0.5);
        // ...while the ions of the two cells draw the rest of them back symmetrically.
        let sum = force.iter().fold([0.0; 2], |a, f| [a[0] + f[0], a[1] + f[1]]);
        assert!(sum[0].abs() < 1e-9 && sum[1].abs() < 1e-9);
    }

    #[test]
    fn evenly_filled_conductor_feels_no_net_close_range_force() {
        // Deep inside a conductor (over a cutoff from its surface), electrons on an even
        // lattice and the evenly spread ions balance: no net close-range force.
        let s = Settings { carriers_per_cell: 16, ..Settings::default() };
        let mut w = World::new(22, 18, s.clone(), 0);
        let cells: Vec<_> = (2..20).flat_map(|c| (2..16).map(move |r| (c, r))).collect();
        w.paint(&cells, Material::Conductor, PaintProps { resistance: 1.0, ..PaintProps::default() });
        let sub: Vec<f64> = (0..4).map(|i| (i as f64 + 0.5) / 4.0).collect();
        let mut positions = vec![];
        for &(c, r) in &cells {
            for &x in &sub {
                for &y in &sub {
                    positions.push([c as f64 + x, r as f64 + y]);
                }
            }
        }
        w.particles.pos = positions;
        let force = CloseRange::new(&s).force(&w);
        let pair = (s.cell_charge / 16.0).powi(2) * 0.25 * compact_g(0.25f64.powi(2), s.electron_softening);
        for (p, f) in w.particles.pos.iter().zip(&force) {
            if p[0] > 6.0 && p[1] > 6.0 && p[0] < 16.0 && p[1] < 12.0 {
                assert!(f[0].abs() < 1e-3 * pair && f[1].abs() < 1e-3 * pair);
            }
        }
    }
}
