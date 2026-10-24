use ndarray::{Array2, Array3};

use crate::battery;
use crate::close_range::CloseRange;
use crate::connectivity::Connectivity;
use crate::maxwell::Maxwell;
use crate::world::World;

/// Particles stopped at a wall stay this far inside their cell.
pub const WALL_GAP: f64 = 1e-6;

/// Drag rate (1/time) per cell: proportional to resistance.
///
/// Drift speed is v = (q/m) E / rate and current per cell I = density * cell_charge * v,
/// so rate = R * density * cell_charge * (q/m) makes I = E / R for any charge density
/// or q/m.
pub fn damping_rates(world: &World) -> Array2<f64> {
    let (s, m) = (&world.settings, &world.materials);
    let per_ohm = s.drag_per_resistance * s.cell_charge * s.carrier_charge_to_mass;
    &m.resistance * &m.density * per_ohm
}

/// The grid cell of a position, clamped onto the grid.
pub fn grid_cell(pos: [f64; 2], cols: usize, rows: usize) -> (usize, usize) {
    (
        (pos[0].floor() as i64).clamp(0, cols as i64 - 1) as usize,
        (pos[1].floor() as i64).clamp(0, rows as i64 - 1) as usize,
    )
}

/// Conventional current per cell, sum of q * v over its electrons, shape (cols, rows, 2).
///
/// For a 1-cell-wide wire this is the current through the cell (charge per time).
pub fn current_density(world: &World) -> Array3<f64> {
    let (m, p) = (&world.materials, &world.particles);
    let mut current = Array3::zeros((m.cols, m.rows, 2));
    for i in 0..p.len() {
        let (c, r) = grid_cell(p.pos[i], m.cols, m.rows);
        current[(c, r, 0)] += p.charge[i] * p.vel[i][0];
        current[(c, r, 1)] += p.charge[i] * p.vel[i][1];
    }
    current
}

/// E + v x B for in-plane velocities and B = (0, 0, B_z); force per unit charge.
///
/// v x (0, 0, B_z) = (v_y B_z, -v_x B_z): in the plane and perpendicular to v,
/// so the magnetic part steers charges but never speeds them up.
pub fn lorentz_field(e: [f64; 2], vel: [f64; 2], bz: f64) -> [f64; 2] {
    [e[0] + vel[1] * bz, e[1] - vel[0] * bz]
}

/// Velocity after dt under acceleration `acc` and drag -damping * v.
///
/// Drag is treated implicitly, which is stable for any damping * dt; at steady
/// state v = acc / damping, i.e. drift velocity proportional to E (Ohm's law).
pub fn kick(vel: [f64; 2], acc: [f64; 2], damping: f64, dt: f64) -> [f64; 2] {
    let scale = 1.0 / (1.0 + damping * dt);
    [(vel[0] + acc[0] * dt) * scale, (vel[1] + acc[1] * dt) * scale]
}

/// Bounce a particle off closed cell faces, so it never leaves its island.
///
/// Assumes it moved less than one cell per axis. The x move is resolved first,
/// then the y move from the resulting column.
pub fn confine(
    old_pos: [f64; 2],
    new_pos: [f64; 2],
    vel: [f64; 2],
    connectivity: &Connectivity,
    restitution: f64,
) -> ([f64; 2], [f64; 2]) {
    let (mut pos, mut vel) = (new_pos, vel);
    let old_cell = [old_pos[0].floor() as i64, old_pos[1].floor() as i64];
    resolve(0, old_cell[0], old_cell[1], &mut pos, &mut vel, connectivity, restitution);
    let column = pos[0].floor() as i64;
    resolve(1, old_cell[1], column, &mut pos, &mut vel, connectivity, restitution);
    (pos, vel)
}

/// Resolve one axis of a move that started in cell `start` (along the axis) and row or
/// column `other` (across it): bounce back if the face crossed is closed.
fn resolve(
    axis: usize,
    start: i64,
    other: i64,
    pos: &mut [f64; 2],
    vel: &mut [f64; 2],
    connectivity: &Connectivity,
    restitution: f64,
) {
    let end = pos[axis].floor() as i64;
    if end == start {
        return;
    }
    let (cols, rows) = connectivity.labels.dim();
    let forward = end > start;
    // Faces are indexed by the cell before them; the grid edge counts as closed.
    let face = if forward { start } else { start - 1 };
    let (n, n_other) = if axis == 0 { (cols, rows) } else { (rows, cols) };
    let on_grid = face >= 0 && face < n as i64 - 1 && other >= 0 && other < n_other as i64;
    let open = on_grid
        && if axis == 0 {
            connectivity.open_x[(face as usize, other as usize)]
        } else {
            connectivity.open_y[(other as usize, face as usize)]
        };
    if open {
        return;
    }
    let wall = if forward { start + 1 } else { start } as f64;
    let bounced = wall - restitution * (pos[axis] - wall);
    pos[axis] = bounced.clamp(start as f64 + WALL_GAP, start as f64 + 1.0 - WALL_GAP);
    vel[axis] *= -restitution;
}

/// Advance the electrons and the fields by dt from simulation `time`.
///
/// Returns the number of substeps used. Substeps keep every move below max_step
/// cells, so confine() sees at most one face crossing per axis. Forces are the
/// Lorentz force q (E + v x B) of the fields at each electron, plus the batteries'
/// impressed field and the close-range forces. The fields then advance by dt
/// with the new charge and current.
pub fn step(world: &mut World, maxwell: &mut Maxwell, close_range: &CloseRange, dt: f64, time: f64) -> usize {
    let s = world.settings.clone();
    if world.particles.is_empty() {
        maxwell.advance(world, dt);
        return 0;
    }
    let speed = world.particles.vel.iter().map(|v| v[0].hypot(v[1])).fold(0.0, f64::max);
    let substeps = ((speed * dt / s.max_step).ceil() as usize).clamp(1, s.max_substeps as usize);
    let h = dt / substeps as f64;
    let damping = damping_rates(world);
    let (cols, rows) = (world.materials.cols, world.materials.rows);
    for k in 0..substeps {
        let impressed = battery::impressed_field_grid(world, time + k as f64 * h);
        let fields = maxwell.gather(&world.particles.pos);
        let close = if close_range.enabled() { Some(close_range.force(world)) } else { None };
        let p = &mut world.particles;
        for i in 0..p.len() {
            let (c, r) = grid_cell(p.pos[i], cols, rows);
            let e = [fields[i][0] + impressed[(c, r, 0)], fields[i][1] + impressed[(c, r, 1)]];
            let qm = p.charge[i] / p.mass[i];
            let lorentz = lorentz_field(e, p.vel[i], fields[i][2]);
            let mut acc = [qm * lorentz[0], qm * lorentz[1]];
            if let Some(close) = &close {
                acc[0] += close[i][0] / p.mass[i];
                acc[1] += close[i][1] / p.mass[i];
            }
            let vel = kick(p.vel[i], acc, damping[(c, r)], h);
            let mut moved = [vel[0] * h, vel[1] * h];
            // Safety net if velocity jumped within this substep: never move further than max_step.
            let length = moved[0].hypot(moved[1]);
            let limit = (s.max_step / length.max(1e-300)).min(1.0);
            moved = [moved[0] * limit, moved[1] * limit];
            let target = [p.pos[i][0] + moved[0], p.pos[i][1] + moved[1]];
            (p.pos[i], p.vel[i]) = confine(p.pos[i], target, vel, &world.connectivity, s.wall_restitution);
        }
    }
    maxwell.advance(world, dt);
    substeps
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectivity::compute_connectivity;
    use crate::materials::{Material, Materials, Orientation, PaintProps};
    use crate::scenes;
    use crate::settings::Settings;
    use crate::simulation::Simulation;
    use crate::test_util::outline;
    use rand::{RngExt, SeedableRng};
    use rand_pcg::Pcg64;

    fn wire() -> PaintProps {
        PaintProps { resistance: 1.0, ..PaintProps::default() }
    }

    #[test]
    fn kick_reaches_terminal_velocity_acc_over_damping() {
        let mut vel = [0.0; 2];
        for _ in 0..2000 {
            vel = kick(vel, [2.0, -1.0], 4.0, 0.01);
        }
        assert!((vel[0] - 0.5).abs() < 1e-8 && (vel[1] + 0.25).abs() < 1e-8);
    }

    #[test]
    fn kick_is_stable_for_huge_damping() {
        let new = kick([5.0, 0.0], [0.0; 2], 1e6, 0.1);
        // Implicit drag only shrinks the velocity, it never flips or blows it up.
        assert!(0.0 <= new[0] && new[0] < 1e-4);
    }

    fn make_connectivity(cells: &[(usize, usize)], battery: Option<((usize, usize), Orientation)>) -> Connectivity {
        let mut m = Materials::new(6, 5);
        m.paint(cells, Material::Conductor, wire());
        if let Some((cell, orientation)) = battery {
            m.paint(&[cell], Material::Battery, PaintProps { emf: 1.0, orientation, ..PaintProps::default() });
        }
        compute_connectivity(&m)
    }

    #[test]
    fn confine_lets_particles_cross_open_faces() {
        let conn = make_connectivity(&[(1, 1), (2, 1)], None);
        let (pos, vel) = confine([1.9, 1.5], [2.1, 1.5], [1.0, 0.0], &conn, 0.5);
        assert_eq!((pos, vel), ([2.1, 1.5], [1.0, 0.0]));
    }

    #[test]
    fn confine_bounces_off_vacuum() {
        let cases = [
            ([1.9, 1.5], [2.1, 1.5], [1.0, 0.0]), // Into vacuum on the right.
            ([1.5, 1.1], [1.5, 0.9], [0.0, -1.0]), // Into vacuum above.
            ([1.1, 1.9], [0.9, 2.1], [-1.0, 1.0]), // Diagonally out of a corner.
        ];
        for (old, new, vel) in cases {
            let conn = make_connectivity(&[(1, 1)], None);
            let (pos, v) = confine(old, new, vel, &conn, 0.5);
            assert_eq!((pos[0].floor(), pos[1].floor()), (1.0, 1.0));
            // Velocity components that hit a wall reverse and shrink.
            for axis in 0..2 {
                if vel[axis] != 0.0 {
                    assert!((v[axis] + 0.5 * vel[axis]).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn confine_treats_grid_edge_as_wall() {
        let conn = make_connectivity(&[(0, 0)], None);
        let (pos, _) = confine([0.1, 0.5], [-0.1, 0.5], [-1.0, 0.0], &conn, 0.0);
        assert!(0.0 <= pos[0] && pos[0] < 1.0);
    }

    #[test]
    fn electrons_pass_through_batteries() {
        // Battery at (2, 1) pointing right, wired on both terminals.
        let conn = make_connectivity(&[(1, 1), (3, 1)], Some(((2, 1), Orientation::Right)));
        let (pos, _) = confine([1.9, 1.5], [2.1, 1.5], [1.0, 0.0], &conn, 0.0);
        assert_eq!((pos[0].floor(), pos[1].floor()), (2.0, 1.0));
    }

    #[test]
    fn random_walk_never_leaves_island() {
        let mut rng = Pcg64::seed_from_u64(5);
        let mut m = Materials::new(12, 10);
        let cells: Vec<_> = (0..12).flat_map(|c| (0..10).map(move |r| (c, r))).filter(|_| rng.random::<f64>() < 0.55).collect();
        m.paint(&cells, Material::Conductor, wire());
        let conn = compute_connectivity(&m);
        let mut pos: Vec<[f64; 2]> = cells.iter().map(|&(c, r)| [c as f64 + 0.5, r as f64 + 0.5]).collect();
        let start: Vec<i64> = cells.iter().map(|&cell| conn.labels[cell]).collect();
        for _ in 0..3000 {
            let restitution = rng.random::<f64>();
            for (p, island) in pos.iter_mut().zip(&start) {
                let moved = [rng.random_range(-0.45..0.45), rng.random_range(-0.45..0.45)];
                (*p, _) = confine(*p, [p[0] + moved[0], p[1] + moved[1]], moved, &conn, restitution);
                assert_eq!(conn.labels[(p[0].floor() as usize, p[1].floor() as usize)], *island);
            }
        }
    }

    fn block_world() -> World {
        let mut w = World::new(40, 32, Settings::default(), 0);
        let cells: Vec<_> = (14..26).flat_map(|c| (13..19).map(move |r| (c, r))).collect();
        w.paint(&cells, Material::Conductor, wire());
        w
    }

    #[test]
    fn neutral_conductor_stays_quiet() {
        // Electrons spawn spread over their cells; the outermost ones settle in ~4 time units.
        let mut sim = Simulation::new(block_world());
        sim.run(5.0);
        let fastest = sim.world.particles.vel.iter().map(|v| v[0].hypot(v[1])).fold(0.0, f64::max);
        assert!(fastest < 0.02, "fastest {fastest}");
    }

    #[test]
    fn conductor_screens_external_charge() {
        let mut w = block_world();
        w.place_charges(&[(8, 16)], 5.0);
        // Interior cells, one cell in from the surface.
        let inner: Vec<_> = (15..25).flat_map(|c| (14..18).map(move |r| (c, r))).collect();
        let mean_e = |sim: &Simulation| {
            let e = sim.e_field();
            let n = inner.len() as f64;
            let sum = inner.iter().fold([0.0; 2], |a, &(c, r)| [a[0] + e[(c, r, 0)], a[1] + e[(c, r, 1)]]);
            (sum[0] / n).hypot(sum[1] / n)
        };
        let mut sim = Simulation::new(w);
        let unscreened = mean_e(&sim);
        sim.run(10.0);
        assert!(mean_e(&sim) < 0.15 * unscreened);
        // Electrons shift towards the positive charge on the left (block centre x = 20).
        // Screening a weak field needs only a little surface charge, so the shift is small.
        let p = &sim.world.particles;
        assert!(p.pos.iter().map(|p| p[0]).sum::<f64>() / (p.len() as f64) < 20.0 - 0.01);
        assert!((0..p.len()).all(|i| {
            let (c, r) = p.cell(i);
            sim.world.materials.is_conductive((c as usize, r as usize))
        }));
    }

    #[test]
    fn fast_particles_get_substeps() {
        let mut w = block_world();
        w.particles.vel[0] = [100.0, 0.0];
        let mut maxwell = Maxwell::for_world(&w);
        let close = CloseRange::new(&w.settings);
        assert!(step(&mut w, &mut maxwell, &close, 1.0 / 30.0, 0.0) > 1);
    }

    #[test]
    fn step_without_particles_does_nothing() {
        let mut w = World::new(5, 5, Settings::default(), 0);
        let mut maxwell = Maxwell::for_world(&w);
        let close = CloseRange::new(&w.settings);
        assert_eq!(step(&mut w, &mut maxwell, &close, 1.0 / 30.0, 0.0), 0);
    }

    #[test]
    fn electrons_spread_over_a_thin_wire_instead_of_stacking() {
        // Regression: electrons used to collapse onto the centre line, in exact stacks.
        let mut w = World::new(20, 14, Settings::default(), 0);
        w.paint(&outline(3, 16, 3, 10), Material::Conductor, wire());
        let mut rng = Pcg64::seed_from_u64(1);
        for p in w.particles.pos.iter_mut() {
            // All stacked.
            *p = [p[0].floor() + 0.5 + rng.random_range(-0.02..0.02), p[1].floor() + 0.5 + rng.random_range(-0.02..0.02)];
        }
        let mut sim = Simulation::new(w);
        sim.run(5.0);
        let p = &sim.world.particles;
        let mut across = vec![];
        for i in 0..p.len() {
            let (c, r) = p.cell(i);
            let vertical = c == 3 || c == 16;
            let corner = vertical && (r == 3 || r == 10);
            if !corner {
                let x = if vertical { p.pos[i][0] } else { p.pos[i][1] };
                across.push((x.rem_euclid(1.0) - 0.5).abs());
            }
        }
        // Filling the width evenly gives a mean distance of 0.25 from the centre line.
        assert!(across.iter().sum::<f64>() / across.len() as f64 > 0.2);
        let mut gap = f64::INFINITY;
        for i in 0..p.len() {
            for j in i + 1..p.len() {
                gap = gap.min((p.pos[i][0] - p.pos[j][0]).hypot(p.pos[i][1] - p.pos[j][1]));
            }
        }
        assert!(gap > 0.05, "gap {gap}");
    }

    #[test]
    fn lorentz_magnetic_part_is_perpendicular_to_velocity() {
        let mut rng = Pcg64::seed_from_u64(1);
        for _ in 0..20 {
            let vel = [rng.random_range(-2.0..2.0), rng.random_range(-2.0..2.0)];
            let force = lorentz_field([0.0; 2], vel, rng.random_range(-2.0..2.0));
            assert!((force[0] * vel[0] + force[1] * vel[1]).abs() < 1e-12);
        }
    }

    #[test]
    fn lorentz_force_steers_electrons_drifting_with_a_current_towards_it() {
        // Parallel currents attract: B below a current to the right points into the
        // screen, and an electron drifting left there is pushed up, towards the current.
        let force = lorentz_field([0.0; 2], [-1.0, 0.0], 1.0);
        assert!(-force[1] < 0.0);
    }

    #[test]
    fn battery_current_is_carried_by_its_electrons() {
        let mut w = World::new(10, 6, Settings::default(), 0);
        w.paint(&[(1, 2), (2, 2), (5, 2), (6, 2)], Material::Conductor, wire());
        w.paint(&[(3, 2), (4, 2)], Material::Battery,
                PaintProps { emf: 2.0, orientation: Orientation::Left, ..PaintProps::default() });
        let stack = w.stacks[0].clone();
        // Electrons drifting right: conventional current from high (right) to low.
        w.particles.vel.iter_mut().for_each(|v| *v = [1.0, 0.0]);
        let current = current_density(&w);
        let x = |c: usize| current[(c, 2, 0)];
        // The battery's denser electrons carry more current at the same speed.
        assert!(x(3) < x(2) && x(2) < 0.0 && x(4) < x(5) && x(5) < 0.0);
        // Averaged over the stack and its terminals.
        let expected = (2..6).map(x).sum::<f64>() / 4.0;
        assert!((battery::stack_current(&w, &stack, &current) - expected).abs() < 1e-9);
    }

    #[test]
    fn scenes_module_is_linked() {
        let _ = scenes::SCENES.len();
    }
}
