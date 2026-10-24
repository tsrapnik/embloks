//! The GPU simulation against the CPU reference (embloks-core), on whatever adapter
//! is available (lavapipe in the dev container). Skipped without an adapter.

use embloks_core::battery;
use embloks_core::materials::{Material, Orientation, PaintProps};
use embloks_core::physics::current_density;
use embloks_core::scenes::{self, SCENES};
use embloks_core::settings::Settings;
use embloks_core::simulation::Simulation;
use embloks_core::world::World;
use embloks_gpu::Gpu;
use embloks_gpu::sim::GpuSim;
use ndarray::{Array2, Array3, s};

fn gpu() -> Option<Gpu> {
    match pollster::block_on(Gpu::headless()) {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("skipping GPU test: {e}");
            None
        }
    }
}

fn scene(index: usize) -> World {
    let scene = &SCENES[index];
    let mut settings = Settings::default();
    (scene.settings)(&mut settings);
    let mut world = World::new(scene.cols, scene.rows, settings, 0);
    (scene.build)(&mut world);
    world
}

/// Largest absolute difference relative to the largest reference value.
fn relative_error<'a>(actual: impl IntoIterator<Item = &'a f64>, expected: impl IntoIterator<Item = &'a f64>) -> f64 {
    let (mut error, mut scale) = (0.0f64, 0.0f64);
    for (a, e) in actual.into_iter().zip(expected) {
        error = error.max((a - e).abs());
        scale = scale.max(e.abs());
    }
    error / scale.max(1e-300)
}

#[test]
fn static_fields_match_the_cpu() {
    let Some(gpu) = gpu() else { return };
    for index in [0, 1, 3] {
        let world = scene(index);
        let cpu = Simulation::new(world.clone());
        let sim = GpuSim::new(&gpu, &world);
        let (ex, ey, bz) = sim.read_yee(&gpu);
        let potential = sim.read_potential(&gpu);
        assert!(relative_error(&potential, &cpu.maxwell.potential) < 1e-4, "scene {index}: potential");
        assert!(relative_error(&ex, &cpu.maxwell.ex) < 1e-3, "scene {index}: Ex");
        assert!(relative_error(&ey, &cpu.maxwell.ey) < 1e-3, "scene {index}: Ey");
        assert!(bz.iter().all(|&b| b == 0.0));
        let (e, _) = sim.read_display(&gpu);
        assert!(relative_error(&e, &cpu.maxwell.node_fields().0) < 1e-3, "scene {index}: display E");
    }
}

#[test]
fn ticks_follow_the_cpu() {
    let Some(gpu) = gpu() else { return };
    for index in [0, 1, 5] {
        let world = scene(index);
        let mut cpu = Simulation::new(world.clone());
        let mut sim = GpuSim::new(&gpu, &world);
        let ticks = 10;
        sim.run_ticks(&gpu, ticks);
        for _ in 0..ticks {
            cpu.step();
        }
        let mut read = world.clone();
        sim.read_state(&gpu, &mut read);
        let worst = read.particles.pos.iter().zip(&cpu.world.particles.pos)
            .map(|(a, b)| (a[0] - b[0]).abs().max((a[1] - b[1]).abs()))
            .fold(0.0, f64::max);
        assert!(worst < 2e-3, "scene {index}: electrons {worst} cells apart");
        let (ex, ey, bz) = sim.read_yee(&gpu);
        assert!(relative_error(&ex, &cpu.maxwell.ex) < 2e-2, "scene {index}: Ex {}", relative_error(&ex, &cpu.maxwell.ex));
        assert!(relative_error(&ey, &cpu.maxwell.ey) < 2e-2, "scene {index}: Ey");
        assert!(relative_error(&bz, &cpu.maxwell.bz) < 5e-2, "scene {index}: Bz {}", relative_error(&bz, &cpu.maxwell.bz));
        assert!(relative_error(&sim.read_current(&gpu), &current_density(&cpu.world)) < 5e-2, "scene {index}: current");
        for stack in &cpu.world.stacks {
            let (g, c) = (read.battery_gain.get(stack), cpu.world.battery_gain.get(stack));
            assert!((g.unwrap_or(&1.0) - c.unwrap_or(&1.0)).abs() < 1e-3, "scene {index}: gains {g:?} vs {c:?}");
        }
    }
}

fn wire(resistance: f64) -> PaintProps {
    PaintProps { resistance, ..PaintProps::default() }
}

#[test]
fn loop_current_follows_ohms_law() {
    let Some(gpu) = gpu() else { return };
    let s = Settings::default();
    let mut w = World::new(30, 20, s.clone(), 0);
    let lp = scenes::outline(4, 23, 3, 12);
    w.paint(&lp, Material::Conductor, wire(0.5));
    w.paint(&[(14, 12)], Material::Battery, PaintProps { emf: 2.0, orientation: Orientation::Right, ..PaintProps::default() });
    let expected = 2.0 / ((lp.len() - 1) as f64 * 0.5 + s.battery_resistance);
    let mut sim = GpuSim::new(&gpu, &w);
    sim.run_ticks(&gpu, (30.0 / s.dt) as usize);
    let mut samples = vec![];
    for _ in 0..20 {
        sim.run_ticks(&gpu, (0.5 / s.dt) as usize);
        let current = sim.read_current(&gpu);
        samples.push(current.slice(s![6..22, 3, 0]).mean().unwrap());
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    assert!((-mean / expected - 1.0).abs() < 0.25, "current {mean} vs {expected}");
}

#[test]
fn conductor_screens_external_charge() {
    let Some(gpu) = gpu() else { return };
    let mut w = World::new(40, 32, Settings::default(), 0);
    let block: Vec<_> = (14..26).flat_map(|c| (13..19).map(move |r| (c, r))).collect();
    w.paint(&block, Material::Conductor, wire(1.0));
    w.place_charges(&[(8, 16)], 5.0);
    let mean_e = |e: &Array3<f64>| {
        let inner = e.slice(s![15..25, 14..18, ..]);
        let n = (inner.len() / 2) as f64;
        (inner.slice(s![.., .., 0]).sum() / n).hypot(inner.slice(s![.., .., 1]).sum() / n)
    };
    let mut sim = GpuSim::new(&gpu, &w);
    let unscreened = mean_e(&sim.read_display(&gpu).0);
    sim.run_ticks(&gpu, (10.0 / w.settings.dt) as usize);
    let screened = mean_e(&sim.read_display(&gpu).0);
    assert!(screened < 0.15 * unscreened, "{screened} vs {unscreened}");
}

#[test]
fn series_scene_drives_three_times_the_current() {
    let Some(gpu) = gpu() else { return };
    let world = scene(2);
    let dt = world.settings.dt;
    let mut sim = GpuSim::new(&gpu, &world);
    sim.run_ticks(&gpu, (30.0 / dt) as usize);
    let (mut left, mut right) = (0.0, 0.0);
    for _ in 0..60 {
        sim.run_ticks(&gpu, 15);
        let current = sim.read_current(&gpu);
        left += current.slice(s![5..16, 8, 0]).mean().unwrap();
        right += current.slice(s![24..35, 8, 0]).mean().unwrap();
    }
    assert!((right / left / 3.0 - 1.0).abs() < 0.2, "ratio {}", right / left);
}

#[test]
fn capacitor_reaches_the_emf_and_the_probe_reads_it() {
    let Some(gpu) = gpu() else { return };
    let world = scene(3);
    let mut sim = GpuSim::new(&gpu, &world);
    sim.run_ticks(&gpu, (60.0 / world.settings.dt) as usize);
    let plate = |col: usize| {
        let probes: Vec<f64> = (10..22).map(|r| sim.read_probe(&gpu, Some((col, r)), 1).potential).collect();
        probes.iter().sum::<f64>() / probes.len() as f64
    };
    let emf = world.stacks[0].emf.abs();
    let difference = plate(22) - plate(16);
    assert!((difference / emf - 1.0).abs() < 0.1, "plates {difference} apart, EMF {emf}");
    // The battery's own reading: connected, and its terminals the EMF apart.
    let battery = sim.read_probe(&gpu, Some((19, 28)), 1);
    assert_eq!(battery.stack_state, 2);
    let stack = &world.stacks[0];
    let sign = stack.emf.signum();
    assert!((sign * (battery.v_high - battery.v_low) / emf - 1.0).abs() < 0.1);
    let _ = battery::probe_cells(&world);
}

#[test]
fn probe_counts_electrons_and_reads_the_wire_current() {
    let Some(gpu) = gpu() else { return };
    let world = scene(4); // Parallel paths.
    let mut sim = GpuSim::new(&gpu, &world);
    let probe = sim.read_probe(&gpu, Some((5, 15)), 2);
    assert!(probe.cell.is_some() && probe.electrons > 0);
    assert!(probe.mean_e_in_wires.is_some());
    assert!(probe.histograms[0].iter().sum::<u32>() > 0 && probe.histograms[1].iter().all(|&c| c == 0));
    sim.run_ticks(&gpu, (50.0 / world.settings.dt) as usize);
    let (mut thin, mut wide) = (0.0, 0.0);
    for _ in 0..20 {
        sim.run_ticks(&gpu, 30);
        thin += sim.read_probe(&gpu, Some((20, 10)), 2).wire_current.0;
        wide += sim.read_probe(&gpu, Some((20, 22)), 2).wire_current.0;
    }
    assert_eq!(sim.read_probe(&gpu, Some((20, 22)), 2).wire_current.1, 4);
    assert!((wide / thin / 2.0 - 1.0).abs() < 0.3, "wide / thin = {}", wide / thin);
    assert!(sim.read_probe(&gpu, None, 2).cell.is_none());
}

#[test]
fn edits_keep_the_fields_going_and_restart_the_display() {
    let Some(gpu) = gpu() else { return };
    let mut world = scene(0);
    let mut sim = GpuSim::new(&gpu, &world);
    sim.run_ticks(&gpu, 60);
    sim.read_state(&gpu, &mut world);
    let gains_before = world.battery_gain.clone();
    // Paint a test charge: Gauss's law takes it in at once, without a tick.
    world.place_charges(&[(5, 5)], 5.0);
    sim.upload(&gpu, &world, true);
    let mut cpu = Simulation::new(world.clone());
    cpu.maxwell.ex = sim.read_yee(&gpu).0; // Same wave part...
    let (e, _) = sim.read_display(&gpu);
    assert!(e[(6, 5, 0)] > 0.5, "the charge's field shows at once");
    let mut after = world.clone();
    sim.read_state(&gpu, &mut after);
    assert_eq!(after.battery_gain, gains_before);
    let _: Array2<f64> = cpu.maxwell.potential;
}
