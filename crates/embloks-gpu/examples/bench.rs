//! Time physics ticks on the CPU reference and on the GPU, for grids like the
//! README's table (30% of the canvas wire, 8 electrons per cell).
//!
//!     cargo run --release -p embloks-gpu --example bench [cols rows ...]

use std::time::Instant;

use embloks_core::materials::{Material, PaintProps};
use embloks_core::settings::Settings;
use embloks_core::simulation::Simulation;
use embloks_core::world::World;
use embloks_gpu::Gpu;
use embloks_gpu::sim::GpuSim;
use rand::{RngExt, SeedableRng};

fn world(cols: usize, rows: usize) -> World {
    let mut world = World::new(cols, rows, Settings::default(), 0);
    let mut rng = rand_pcg::Pcg64::seed_from_u64(1);
    let cells: Vec<_> = (0..cols).flat_map(|c| (0..rows).map(move |r| (c, r))).filter(|_| rng.random::<f64>() < 0.3).collect();
    world.paint(&cells, Material::Conductor, PaintProps { resistance: 1.0, ..PaintProps::default() });
    world
}

fn main() {
    let args: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let sizes: Vec<(usize, usize)> = if args.len() >= 2 {
        args.chunks(2).map(|c| (c[0], c[1])).collect()
    } else {
        vec![(40, 32), (100, 80), (200, 160)]
    };
    let gpu = pollster::block_on(Gpu::headless()).ok();
    if let Some(gpu) = &gpu {
        println!("GPU: {} ({:?})", gpu.adapter_info.name, gpu.adapter_info.backend);
    }
    println!("{:>9}  {:>9}  {:>12}  {:>12}", "grid", "electrons", "CPU ms/tick", "GPU ms/tick");
    for (cols, rows) in sizes {
        let w = world(cols, rows);
        let electrons = w.particles.len();
        let mut cpu = Simulation::new(w.clone());
        cpu.step();
        let ticks = 5;
        let start = Instant::now();
        for _ in 0..ticks {
            cpu.step();
        }
        let cpu_ms = start.elapsed().as_secs_f64() * 1000.0 / ticks as f64;
        let gpu_ms = gpu.as_ref().map(|gpu| {
            let mut sim = GpuSim::new(gpu, &w);
            sim.run_ticks(gpu, 8); // Warm up (pipeline compilation, first dispatches).
            let ticks = 64;
            let start = Instant::now();
            sim.run_ticks(gpu, ticks);
            start.elapsed().as_secs_f64() * 1000.0 / ticks as f64
        });
        let gpu_text = gpu_ms.map_or("-".to_string(), |ms| format!("{ms:.2}"));
        println!("{:>9}  {electrons:>9}  {cpu_ms:>12.1}  {gpu_text:>12}", format!("{cols}x{rows}"));
    }
}
