//! The fields: Maxwell's equations stepped in time on a staggered grid (FDTD).
//!
//! This is genuine 2D electromagnetism (TE mode: Ex, Ey in the plane, Bz out of it),
//! as in particle-in-cell plasma codes. Charges are infinitely long lines in z, so
//! static fields fall off as 1/r instead of 1/r^2.
//!
//! ```text
//!     dBz/dt = -(dEy/dx - dEx/dy)
//!     dE/dt  = c^2 curl(Bz) - J / eps0          (eps0 = 1 / (4 pi k))
//! ```
//!
//! Layout (Yee grid; nodes are cell centres, as everywhere else in the app):
//! - Ex on vertical cell edges, between nodes along x:   shape (nx + 1, ny)
//! - Ey on horizontal cell edges, between nodes along y:  shape (nx, ny + 1)
//! - Bz on cell corners:                                   shape (nx + 1, ny + 1)
//!
//! Every tick E is corrected so that Gauss's law div E = rho / eps0 holds exactly
//! (divergence cleaning). The electrostatic part of E is then instantaneous, as in
//! the Coulomb gauge, and only the part made by changing currents travels at c.
//! That also keeps the field right when charge appears or jumps without a current
//! (painting, test charges). A margin around the visible grid with gradually
//! increasing damping absorbs outgoing waves.

use std::f64::consts::PI;
use std::sync::Arc;

use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Axis, s};
use rustdct::{Dst1, DctPlanner};

use crate::fields::{CloudInCell, ion_charge};
use crate::settings::Settings;
use crate::world::World;

#[derive(Clone)]
pub struct Maxwell {
    pub cols: usize,
    pub rows: usize,
    pub settings: Settings,
    pub margin: usize,
    pub nx: usize,
    pub ny: usize,
    pub ex: Array2<f64>,
    pub ey: Array2<f64>,
    pub bz: Array2<f64>,
    pub potential: Array2<f64>,
    // Eigenvalues of the 5-point Laplacian with zero boundary values, diagonal in
    // the type-I sine transform: Poisson's equation is solved exactly in one go.
    // Inverted, and with the transforms' normalisation folded in.
    inv_laplacian: Array2<f64>,
    dst_x: Arc<dyn Dst1<f64>>,
    dst_y: Arc<dyn Dst1<f64>>,
    damping: Option<(f64, [Array2<f64>; 3])>,
}

impl Maxwell {
    pub fn new(cols: usize, rows: usize, settings: &Settings) -> Self {
        let m = settings.absorbing_margin;
        assert!(m >= settings.smoothing_passes, "absorbing_margin must be at least smoothing_passes");
        let (nx, ny) = (cols + 2 * m, rows + 2 * m);
        let eigen = |n: usize| -> Vec<f64> {
            (1..=n).map(|k| 2.0 * (PI * k as f64 / (n + 1) as f64).cos() - 2.0).collect()
        };
        let (kx, ky) = (eigen(nx), eigen(ny));
        // rustdct's DST-I is unnormalised (no factor 2): applying it twice multiplies
        // by (n + 1) / 2 per axis.
        let norm = 4.0 / ((nx + 1) * (ny + 1)) as f64;
        let inv_laplacian = Array2::from_shape_fn((nx, ny), |(i, j)| norm / (kx[i] + ky[j]));
        let mut planner = DctPlanner::new();
        Self {
            cols,
            rows,
            settings: settings.clone(),
            margin: m,
            nx,
            ny,
            ex: Array2::zeros((nx + 1, ny)),
            ey: Array2::zeros((nx, ny + 1)),
            bz: Array2::zeros((nx + 1, ny + 1)),
            potential: Array2::zeros((nx, ny)),
            inv_laplacian,
            dst_x: planner.plan_dst1(nx),
            dst_y: planner.plan_dst1(ny),
            damping: None,
        }
    }

    /// Fields for `world`, starting from rest: only the static E of its charges.
    pub fn for_world(world: &World) -> Self {
        let m = &world.materials;
        let mut maxwell = Self::new(m.cols, m.rows, &world.settings);
        let (rho, _) = maxwell.sources(world);
        maxwell.enforce_gauss(&rho);
        maxwell
    }

    // --- Geometry -------------------------------------------------------------------

    /// The part of a node array (nx, ny) covering the visible grid.
    pub fn visible<'a>(&self, array: &'a Array2<f64>) -> ArrayView2<'a, f64> {
        let m = self.margin;
        array.slice(s![m..m + self.cols, m..m + self.rows])
    }

    /// The part of a node array (nx, ny, d) covering the visible grid.
    pub fn visible3<'a>(&self, array: &'a Array3<f64>) -> ArrayView3<'a, f64> {
        let m = self.margin;
        array.slice(s![m..m + self.cols, m..m + self.rows, ..])
    }

    /// Per-step damping factors for Ex, Ey, Bz: 1 inside, falling in the margin.
    pub fn damping_factors(&mut self, dt: f64) -> &[Array2<f64>; 3] {
        if self.damping.as_ref().is_none_or(|(key, _)| *key != dt) {
            let (m, nx, ny) = (self.margin as f64, self.nx, self.ny);
            // 0 inside the visible nodes (margin .. n - 1 - margin), 1 at the outer edge.
            let depth = |coord: f64, n: usize| {
                let outside = (m - coord).max(coord - (n as f64 - 1.0 - m)).max(0.0);
                (outside / m).min(1.0)
            };
            let strength = self.settings.absorbing_strength;
            let factor = |shape: (usize, usize), x_offset: f64, y_offset: f64| {
                Array2::from_shape_fn(shape, |(i, j)| {
                    let d = depth(i as f64 + x_offset, nx).max(depth(j as f64 + y_offset, ny));
                    (-strength * d.powi(3) * dt).exp()
                })
            };
            let factors = [
                factor((nx + 1, ny), -0.5, 0.0),
                factor((nx, ny + 1), 0.0, -0.5),
                factor((nx + 1, ny + 1), -0.5, -0.5),
            ];
            self.damping = Some((dt, factors));
        }
        &self.damping.as_ref().unwrap().1
    }

    // --- Field operations -----------------------------------------------------------

    /// DST-I along both axes, in place.
    fn dst2(&self, array: &mut Array2<f64>) {
        let mut scratch = vec![0.0; self.dst_x.get_scratch_len().max(self.dst_y.get_scratch_len())];
        let mut buffer = vec![0.0; self.nx.max(self.ny)];
        for (axis, dst) in [(Axis(0), &self.dst_x), (Axis(1), &self.dst_y)] {
            for mut lane in array.lanes_mut(axis) {
                let n = lane.len();
                let buffer = &mut buffer[..n];
                for (b, v) in buffer.iter_mut().zip(lane.iter()) {
                    *b = *v;
                }
                // rustdct's FFT-based DST-I never writes the two zero entries of its FFT
                // input, so reused scratch must start out zeroed.
                let scratch = &mut scratch[..dst.get_scratch_len()];
                scratch.fill(0.0);
                dst.process_dst1_with_scratch(buffer, scratch);
                for (v, b) in lane.iter_mut().zip(buffer.iter()) {
                    *v = *b;
                }
            }
        }
    }

    /// phi with (discrete) laplacian(phi) = rhs and phi = 0 just outside the grid.
    pub fn poisson(&self, rhs: &Array2<f64>) -> Array2<f64> {
        let mut phi = rhs.clone();
        self.dst2(&mut phi);
        phi *= &self.inv_laplacian;
        self.dst2(&mut phi);
        phi
    }

    pub fn divergence(&self) -> Array2<f64> {
        let (ex, ey) = (&self.ex, &self.ey);
        &(&ex.slice(s![1.., ..]) - &ex.slice(s![..-1, ..])) + &(&ey.slice(s![.., 1..]) - &ey.slice(s![.., ..-1]))
    }

    /// E -= grad(phi), with phi = 0 outside the grid.
    fn gradient_subtract(&mut self, phi: &Array2<f64>) {
        let (nx, ny) = (self.nx, self.ny);
        for i in 0..=nx {
            let high = if i < nx { Some(i) } else { None };
            let low = if i > 0 { Some(i - 1) } else { None };
            for j in 0..ny {
                let grad = high.map_or(0.0, |h| phi[(h, j)]) - low.map_or(0.0, |l| phi[(l, j)]);
                self.ex[(i, j)] -= grad;
            }
        }
        for i in 0..nx {
            for j in 0..=ny {
                let high = if j < ny { phi[(i, j)] } else { 0.0 };
                let low = if j > 0 { phi[(i, j - 1)] } else { 0.0 };
                self.ey[(i, j)] -= high - low;
            }
        }
    }

    /// Correct E so div E = rho / eps0 exactly, keeping its wave (curl) part.
    ///
    /// Also stores the electric potential of rho (Coulomb gauge), which the
    /// batteries measure. Poisson's equation is linear, so the correction
    /// poisson(div E - 4 pi k rho) is poisson(div E) plus that potential.
    pub fn enforce_gauss(&mut self, rho: &Array2<f64>) {
        self.potential = self.poisson(&rho.mapv(|q| -4.0 * PI * self.settings.coulomb_k * q));
        let correction = self.poisson(&self.divergence()) + &self.potential;
        self.gradient_subtract(&correction);
    }

    /// dEy/dx - dEx/dy at the corners (E is zero beyond the grid).
    fn curl_e(&self, curl: &mut Array2<f64>) {
        let (nx, ny) = (self.nx, self.ny);
        for i in 0..=nx {
            for j in 0..=ny {
                let mut v = 0.0;
                if i < nx {
                    v += self.ey[(i, j)];
                }
                if i > 0 {
                    v -= self.ey[(i - 1, j)];
                }
                if j < ny {
                    v -= self.ex[(i, j)];
                }
                if j > 0 {
                    v += self.ex[(i, j - 1)];
                }
                curl[(i, j)] = v;
            }
        }
    }

    /// Advance E and B by dt. `rho` (nx, ny) and `current` (nx, ny, 2) live on nodes.
    ///
    /// The scheme is stable while light crosses under 1/sqrt(2) cell per step (CFL), so
    /// a fast speed of light gets split into smaller steps. Gauss's law is enforced
    /// once, at the end: the correction is a gradient, which has no curl, so it
    /// doesn't change B or the waves in between.
    pub fn step(&mut self, rho: &Array2<f64>, current: &Array3<f64>, dt: f64) {
        let s = &self.settings;
        let substeps = substeps(s, dt);
        let h = dt / substeps as f64;
        let c2 = s.speed_of_light.powi(2);
        let scale = 4.0 * PI * s.coulomb_k * h;
        let (nx, ny) = (self.nx, self.ny);
        // Currents from nodes onto the edges where E lives (boundary edges get none).
        let jx = Array2::from_shape_fn((nx + 1, ny), |(i, j)| {
            if i == 0 || i == nx { 0.0 } else { 0.5 * scale * (current[(i - 1, j, 0)] + current[(i, j, 0)]) }
        });
        let jy = Array2::from_shape_fn((nx, ny + 1), |(i, j)| {
            if j == 0 || j == ny { 0.0 } else { 0.5 * scale * (current[(i, j - 1, 1)] + current[(i, j, 1)]) }
        });
        let [damp_ex, damp_ey, damp_bz] = self.damping_factors(h).clone();
        let mut curl = Array2::zeros((nx + 1, ny + 1));
        for _ in 0..substeps {
            // Leapfrog: half a step of B, a full step of E, another half step of B.
            self.curl_e(&mut curl);
            self.bz.scaled_add(-0.5 * h, &curl);
            for i in 0..=nx {
                for j in 0..ny {
                    self.ex[(i, j)] += h * c2 * (self.bz[(i, j + 1)] - self.bz[(i, j)]) - jx[(i, j)];
                }
            }
            for i in 0..nx {
                for j in 0..=ny {
                    self.ey[(i, j)] -= h * c2 * (self.bz[(i + 1, j)] - self.bz[(i, j)]) + jy[(i, j)];
                }
            }
            self.curl_e(&mut curl);
            self.bz.scaled_add(-0.5 * h, &curl);
            self.ex *= &damp_ex;
            self.ey *= &damp_ey;
            self.bz *= &damp_bz;
        }
        self.enforce_gauss(rho);
    }

    // --- Coupling to the world ------------------------------------------------------

    /// Visible-grid node values (cols, rows, d) smoothed and placed on the full grid.
    ///
    /// Smoothing runs on the visible grid plus a border as wide as it spreads,
    /// which gives the same result as smoothing the whole (otherwise zero) grid.
    pub fn smoothed(&self, values: &Array3<f64>) -> Array3<f64> {
        let k = self.settings.smoothing_passes;
        let (cols, rows, d) = values.dim();
        let mut padded = Array3::zeros((cols + 2 * k, rows + 2 * k, d));
        padded.slice_mut(s![k..k + cols, k..k + rows, ..]).assign(values);
        for _ in 0..k {
            padded = smooth(&padded);
        }
        let mut out = Array3::zeros((self.nx, self.ny, d));
        let m = self.margin - k;
        out.slice_mut(s![m..m + cols + 2 * k, m..m + rows + 2 * k, ..]).assign(&padded);
        out
    }

    /// smoothed() for a scalar field (cols, rows) -> (nx, ny).
    pub fn smoothed2(&self, values: &Array2<f64>) -> Array2<f64> {
        self.smoothed(&values.clone().insert_axis(Axis(2))).remove_axis(Axis(2))
    }

    /// Smoothed charge (nx, ny) and current (nx, ny, 2) of the world on the full grid.
    ///
    /// All charge counts: ions, test charges and electrons; the current is the
    /// electrons' q*v (batteries included: their electrons move like any other).
    pub fn sources(&self, world: &World) -> (Array2<f64>, Array3<f64>) {
        let (s, m, p) = (&world.settings, &world.materials, &world.particles);
        let cic = CloudInCell::new(&p.pos, m.cols, m.rows);
        let mut values = Array3::zeros((m.cols, m.rows, 3));
        let mut rho = ion_charge(m, s.cell_charge) + cic.deposit(&p.charge);
        if !world.fixed_charge.is_empty() {
            rho += &CloudInCell::new(&world.fixed_pos, m.cols, m.rows).deposit(&world.fixed_charge);
        }
        values.index_axis_mut(Axis(2), 0).assign(&rho);
        let qv: Vec<[f64; 2]> = p.vel.iter().zip(&p.charge).map(|(v, q)| [q * v[0], q * v[1]]).collect();
        values.slice_mut(s![.., .., 1..]).assign(&cic.deposit_vectors(&qv));
        let full = self.smoothed(&values);
        (full.index_axis(Axis(2), 0).to_owned(), full.slice(s![.., .., 1..]).to_owned())
    }

    /// Potential (cols, rows) of visible-grid node charges `rho`, smoothed like all
    /// charge, on its own: the fields themselves are left alone.
    pub fn static_potential(&self, rho: &Array2<f64>) -> Array2<f64> {
        let rhs = self.smoothed2(rho).mapv(|q| -4.0 * PI * self.settings.coulomb_k * q);
        self.visible(&self.poisson(&rhs)).to_owned()
    }

    /// One tick: deposit charge and current from the world, then step the fields.
    pub fn advance(&mut self, world: &World, dt: f64) {
        let (rho, current) = self.sources(world);
        self.step(&rho, &current, dt);
    }

    /// Ex, Ey, Bz at particle positions (in visible-grid cell coordinates).
    ///
    /// Each component is interpolated bilinearly from its own staggered grid. (First
    /// averaging onto cell centres would hide an odd/even "checkerboard" of charge from
    /// the particles: it would feel no force, nothing would push back, and it grows.)
    pub fn gather(&self, positions: &[[f64; 2]]) -> Vec<[f64; 3]> {
        let m = self.margin as f64;
        // Fractional array indices: Ex sits on vertical cell edges (x integer, y centred),
        // Ey on horizontal edges, Bz on corners; node i is cell centre i - m + 0.5.
        positions
            .iter()
            .map(|&[x, y]| {
                [
                    bilinear(&self.ex, x + m, y - 0.5 + m),
                    bilinear(&self.ey, x - 0.5 + m, y + m),
                    bilinear(&self.bz, x + m, y + m),
                ]
            })
            .collect()
    }

    /// E (cols, rows, 2) and Bz (cols, rows) at the visible cell centres.
    pub fn node_fields(&self) -> (Array3<f64>, Array2<f64>) {
        let m = self.margin;
        let e = Array3::from_shape_fn((self.cols, self.rows, 2), |(c, r, k)| {
            let (i, j) = (c + m, r + m);
            if k == 0 {
                0.5 * (self.ex[(i, j)] + self.ex[(i + 1, j)])
            } else {
                0.5 * (self.ey[(i, j)] + self.ey[(i, j + 1)])
            }
        });
        let b = Array2::from_shape_fn((self.cols, self.rows), |(c, r)| {
            let (i, j) = (c + m, r + m);
            0.25 * (self.bz[(i, j)] + self.bz[(i + 1, j)] + self.bz[(i, j + 1)] + self.bz[(i + 1, j + 1)])
        });
        (e, b)
    }

    pub fn visible_potential(&self) -> Array2<f64> {
        self.visible(&self.potential).to_owned()
    }
}

/// FDTD substeps per tick of dt: light must cross under 1/sqrt(2) cell per step.
pub fn substeps(settings: &Settings, dt: f64) -> usize {
    ((settings.speed_of_light * dt * 2f64.sqrt() / 0.9).ceil() as usize).max(1)
}

/// Bilinear interpolation of a 2D array at fractional indices (gx, gy).
pub fn bilinear(array: &Array2<f64>, gx: f64, gy: f64) -> f64 {
    let (w, h) = array.dim();
    let i = (gx.floor() as i64).clamp(0, w as i64 - 2) as usize;
    let j = (gy.floor() as i64).clamp(0, h as i64 - 2) as usize;
    let (fx, fy) = (gx - i as f64, gy - j as f64);
    array[(i, j)] * (1.0 - fx) * (1.0 - fy)
        + array[(i + 1, j)] * fx * (1.0 - fy)
        + array[(i, j + 1)] * (1.0 - fx) * fy
        + array[(i + 1, j + 1)] * fx * fy
}

/// One pass of the 1-2-1 binomial filter along x and y, zero beyond the edges.
///
/// Smears charge and current over neighbouring cells, the grid's counterpart of a
/// softening length: without it, single electrons in a one-cell-wide wire make the
/// field vary wildly within the cell. Conserves charge that stays clear of the edges.
pub fn smooth(array: &Array3<f64>) -> Array3<f64> {
    let mut array = array.clone();
    for axis in [Axis(0), Axis(1)] {
        let n = array.len_of(axis);
        let mut out = array.mapv(|v| 0.5 * v);
        for i in 0..n {
            let quarter = array.index_axis(axis, i).mapv(|v| 0.25 * v);
            if i > 0 {
                out.index_axis_mut(axis, i - 1).scaled_add(1.0, &quarter);
            }
            if i + 1 < n {
                out.index_axis_mut(axis, i + 1).scaled_add(1.0, &quarter);
            }
        }
        array = out;
    }
    array
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::test_util::{assert_rel, normal};

    fn point_charge(mx: &Maxwell, col: usize, row: usize, q: f64) -> Array2<f64> {
        let mut rho = Array2::zeros((mx.nx, mx.ny));
        rho[(col + mx.margin, row + mx.margin)] = q;
        rho
    }

    /// B_z at the cell centres once a steady current has been flowing for a while.
    pub(crate) fn steady_b(current: &Array3<f64>, settings: &Settings, seconds: f64) -> Array2<f64> {
        let (cols, rows, _) = current.dim();
        let mut mx = Maxwell::new(cols, rows, settings);
        let rho = Array2::zeros((mx.nx, mx.ny));
        let full = mx.smoothed(current);
        for _ in 0..(seconds / settings.dt) as usize {
            mx.step(&rho, &full, settings.dt);
        }
        mx.node_fields().1
    }

    #[test]
    fn gauss_law_holds_exactly_and_static_field_is_2d_coulomb() {
        let s = Settings::default();
        let mut mx = Maxwell::new(40, 32, &s);
        let rho = point_charge(&mx, 20, 16, 1.0);
        mx.enforce_gauss(&rho);
        let residual = mx.divergence() - rho.mapv(|q| 4.0 * PI * s.coulomb_k * q);
        let worst = residual.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        assert!(worst < 1e-12, "residual {worst}");
        let (e, _) = mx.node_fields();
        for r in [4, 8] {
            // A line charge: |E| = 2 k q / r.
            assert_rel(e[(20 + r, 16, 0)].hypot(e[(20 + r, 16, 1)]), 2.0 / r as f64, 0.05);
        }
    }

    fn gaussian_pulse(mx: &mut Maxwell, col: f64, row: f64, width2: f64) {
        let m = mx.margin as f64;
        mx.bz = Array2::from_shape_fn((mx.nx + 1, mx.ny + 1), |(i, j)| {
            let (x, y) = (i as f64 - 0.5 - col - m, j as f64 - 0.5 - row - m);
            (-(x * x + y * y) / width2).exp()
        });
    }

    fn quiet(mx: &Maxwell) -> (Array2<f64>, Array3<f64>) {
        (Array2::zeros((mx.nx, mx.ny)), Array3::zeros((mx.nx, mx.ny, 2)))
    }

    #[test]
    fn waves_travel_at_the_speed_of_light() {
        // Slow enough to stay on the grid.
        let s = Settings { absorbing_margin: 40, speed_of_light: 2.0, ..Settings::default() };
        let mut mx = Maxwell::new(80, 80, &s);
        gaussian_pulse(&mut mx, 40.0, 40.0, 0.5);
        let (rho, current) = quiet(&mx);
        let (mut fronts, mut t) = (vec![], 0.0);
        for until in [5.0, 10.0, 15.0] {
            while t < until - 1e-9 {
                mx.step(&rho, &current, s.dt);
                t += s.dt;
            }
            let start = mx.margin + 40;
            let profile = mx.bz.slice(s![start.., start]);
            fronts.push(profile.iter().rposition(|v| v.abs() > 1e-3).unwrap() as f64);
        }
        for pair in fronts.windows(2) {
            assert!(((pair[1] - pair[0]) / 5.0 - s.speed_of_light).abs() <= 0.25);
        }
    }

    #[test]
    fn absorbing_border_swallows_outgoing_waves() {
        let s = Settings::default();
        let mut mx = Maxwell::new(40, 32, &s);
        gaussian_pulse(&mut mx, 20.0, 16.0, 2.0);
        let (rho, current) = quiet(&mx);
        let energy = |mx: &Maxwell| {
            mx.ex.mapv(|v| v * v).sum() + mx.ey.mapv(|v| v * v).sum() + s.speed_of_light.powi(2) * mx.bz.mapv(|v| v * v).sum()
        };
        for _ in 0..(5.0 / s.dt) as usize {
            mx.step(&rho, &current, s.dt);
        }
        let before = energy(&mx);
        for _ in 0..(40.0 / s.dt) as usize {
            mx.step(&rho, &current, s.dt);
        }
        assert!(energy(&mx) < 1e-3 * before);
    }

    #[test]
    fn fast_light_is_substepped_and_stays_stable() {
        // c dt sqrt(2) ~ 1.9: unstable in a single step.
        let s = Settings { speed_of_light: 40.0, ..Settings::default() };
        let mut mx = Maxwell::new(30, 30, &s);
        gaussian_pulse(&mut mx, 15.0, 15.0, 2.0);
        let (rho, current) = quiet(&mx);
        for _ in 0..300 {
            mx.step(&rho, &current, s.dt);
        }
        assert!(mx.bz.iter().all(|v| v.is_finite() && v.abs() < 1.0));
    }

    #[test]
    fn staggered_gather_reads_uniform_fields_exactly() {
        let mut mx = Maxwell::new(10, 8, &Settings::default());
        mx.ex.fill(0.3);
        mx.ey.fill(-0.2);
        mx.bz.fill(0.7);
        for f in mx.gather(&[[2.3, 4.6], [7.9, 1.1]]) {
            assert!((f[0] - 0.3).abs() < 1e-12 && (f[1] + 0.2).abs() < 1e-12 && (f[2] - 0.7).abs() < 1e-12);
        }
    }

    #[test]
    fn smoothing_conserves_charge() {
        let mut values = Array3::from_shape_vec((12, 9, 3), normal(0, 12 * 9 * 3)).unwrap();
        // Charge on the outermost ring would leak out; in the app that ring lies deep inside
        // the absorbing margin, where there is never any charge.
        for i in [0, 11] {
            values.index_axis_mut(Axis(0), i).fill(0.0);
        }
        for j in [0, 8] {
            values.index_axis_mut(Axis(1), j).fill(0.0);
        }
        let smoothed = smooth(&values);
        for k in 0..3 {
            assert_rel(smoothed.index_axis(Axis(2), k).sum(), values.index_axis(Axis(2), k).sum(), 1e-9);
        }
    }

    #[test]
    fn smoothing_the_visible_grid_matches_smoothing_the_whole_grid() {
        let s = Settings::default();
        let mx = Maxwell::new(9, 7, &s);
        let values = Array3::from_shape_vec((9, 7, 3), normal(1, 9 * 7 * 3)).unwrap();
        let mut whole = Array3::zeros((mx.nx, mx.ny, 3));
        let m = mx.margin;
        whole.slice_mut(s![m..m + 9, m..m + 7, ..]).assign(&values);
        for _ in 0..s.smoothing_passes {
            whole = smooth(&whole);
        }
        let ours = mx.smoothed(&values);
        assert!(ours.iter().zip(whole.iter()).all(|(a, b)| (a - b).abs() < 1e-15));
    }

    #[test]
    fn current_loop_is_a_solenoid_following_the_right_hand_rule() {
        // In 2D a loop is a long solenoid: uniform B inside, none outside.
        let s = Settings::default();
        let mut current = Array3::zeros((30, 30, 2));
        // Clockwise on screen (y down): right along the top, down the right side, ...
        for i in 8..22 {
            current[(i, 8, 0)] = 1.0;
            current[(22, i, 1)] = 1.0;
            current[(i, 22, 0)] = -1.0;
            current[(8, i, 1)] = -1.0;
        }
        let bz = steady_b(&current, &s, 40.0);
        let inside = bz.slice(s![12..19, 12..19]);
        let outside = bz.slice(s![1..5, 1..5]);
        // Clockwise on screen is anticlockwise seen from behind it: B points into the screen.
        assert!(inside.iter().all(|&b| b > 0.0));
        let mean = inside.mean().unwrap();
        assert!(inside.std(0.0) < 0.1 * mean);
        assert!(outside.iter().all(|b| b.abs() < 0.1 * mean));
        // Ampere's law for a solenoid: B = 4 pi k I / c^2 (current per unit length 1).
        assert_rel(mean, 4.0 * PI * s.coulomb_k / s.speed_of_light.powi(2), 0.1);
    }
}

#[cfg(test)]
mod poisson_tests {
    use super::*;

    #[test]
    fn poisson_inverts_the_laplacian_at_many_sizes() {
        for (cols, rows, margin) in [(10, 8, 4), (10, 8, 16), (40, 32, 4), (40, 32, 16)] {
            let s = Settings { absorbing_margin: margin, ..Settings::default() };
            let mx = Maxwell::new(cols, rows, &s);
            let rhs = Array2::from_shape_fn((mx.nx, mx.ny), |(i, j)| ((i * 7 + j * 3) % 11) as f64 - 5.0);
            let phi = mx.poisson(&rhs);
            let at = |i: i64, j: i64| if i < 0 || j < 0 || i >= mx.nx as i64 || j >= mx.ny as i64 { 0.0 } else { phi[(i as usize, j as usize)] };
            let mut worst = 0.0f64;
            for i in 0..mx.nx as i64 {
                for j in 0..mx.ny as i64 {
                    let lap = at(i + 1, j) + at(i - 1, j) + at(i, j + 1) + at(i, j - 1) - 4.0 * at(i, j);
                    worst = worst.max((lap - rhs[(i as usize, j as usize)]).abs());
                }
            }
            assert!(worst < 1e-9, "{cols}x{rows} margin {margin}: residual {worst:e}");
        }
    }
}
