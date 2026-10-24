//! The physics of embloks-core's `Simulation`, in WebGPU compute shaders.
//!
//! The CPU `World` stays authoritative for materials, batteries and test charges,
//! and for adding or removing electrons while painting. Everything that moves
//! (electrons, fields, battery gains, smoothed currents) lives on the GPU between
//! edits: a frame's ticks are encoded in one go with no read back, and the info
//! panel's numbers come back through a small probe buffer.
//!
//! Single precision throughout; simulated time stays f64 on the CPU.

use std::f64::consts::PI;

use embloks_core::battery::{Stack, terminal_block, wired};
use embloks_core::close_range::{self, CloseRange, PAIR_SAMPLES, SQUARE_SAMPLES};
use embloks_core::fields::{CloudInCell, ion_charge};
use embloks_core::materials::{Material, Materials};
use embloks_core::maxwell::{self, Maxwell};
use embloks_core::physics::damping_rates;
use embloks_core::settings::Settings;
use embloks_core::world::World;
use ndarray::{Array2, Array3};

use crate::Gpu;
use crate::layout::gpu_struct;

/// Most ticks encoded per frame (Settings::max_ticks_per_frame is 32).
pub const MAX_TICKS: usize = 64;
/// f32 positions near x = 200 resolve ~2e-5 cells: the CPU's 1e-6 wall gap would vanish.
pub const WALL_GAP: f32 = 1e-4;
/// The impressed field's sin(2 pi f t) uses t modulo this, so f32 keeps its phase
/// exact for frequencies that are multiples of 1 / PERIOD (all of the AC ladder).
pub const PERIOD: f64 = 1000.0;
/// Bins of the log-magnitude histograms behind the layers' auto gain: 10^-12 .. 10^6.
pub const HIST_BINS: usize = 256;
const HIST_LOG_MIN: f64 = -12.0;
const HIST_LOG_SPAN: f64 = 18.0;
const STACK_HEADER: usize = 12;
const PROBE_OUT: usize = 32;

gpu_struct!(Params {
    cols: u32, rows: u32, nx: u32, ny: u32, margin: u32, n: u32, cells: u32, n_stacks: u32,
    dt: f32, max_step: f32, max_substeps: u32, restitution: f32,
    charge: f32, mass: f32, k: f32, k_ion: f32,
    cutoff: f32, reach: u32, half: f32, square_samples: f32, square_n: u32, g_scale: f32,
    c2: f32, four_pi_k: f32, fdtd_h: f32, wall_gap: f32,
    alpha_current: f32, alpha_display: f32, inv_mu0: f32,
    trim_rate: f32, gain_low: f32, gain_high: f32,
    o_ex: u32, o_ey: u32, o_bz: u32, o_src: u32, o_tmp: u32, o_work: u32, o_work2: u32, o_phi: u32,
    o_damp_ex: u32, o_damp_ey: u32, o_damp_bz: u32, o_static_rho: u32, o_ion_potential: u32,
    o_current: u32, o_current_smooth: u32, o_display: u32,
    o_counts: u32, o_start: u32, o_order: u32, o_pcell: u32, o_prank: u32,
    o_g: u32, o_square: u32, o_sine_x: u32, o_sine_y: u32, o_inv_laplacian: u32,
    o_gain: u32, o_stack_header: u32, o_stack_cells: u32, o_tick_time: u32, o_targets: u32,
    o_stack_readings: u32, o_probe_request: u32, o_probe_out: u32,
});

/// Bump allocator for the arrays inside one buffer (offsets in elements).
#[derive(Default)]
struct Regions(u32);

impl Regions {
    fn take(&mut self, len: usize) -> u32 {
        let at = self.0;
        // Keep every array 16-byte aligned, so copies of any of them are allowed.
        self.0 += (len as u32).next_multiple_of(4);
        at
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct CellInfo {
    damping: f32,
    emf: f32,
    frequency: f32,
    density: f32,
    flags: u32,
    stack: u32,
    _pad: [u32; 2],
}

/// The info panel's numbers for one frame, as the probe shaders left them.
#[derive(Clone, Debug, Default)]
pub struct Probe {
    pub cell: Option<embloks_core::materials::Cell>,
    pub electrons: usize,
    pub potential: f64,
    pub e: [f64; 2],
    pub b: f64,
    pub s: [f64; 2],
    pub current: [f64; 2],
    pub wire_current: (f64, usize),
    /// 0: not a battery, 1: battery not connected, 2: connected.
    pub stack_state: u32,
    pub v_low: f64,
    pub v_high: f64,
    pub stack_current: f64,
    pub gain: f64,
    pub mean_e_in_wires: Option<f64>,
    /// Per layer (E, B, S), counts of magnitudes per log bin (see HIST_BINS).
    pub histograms: [Vec<u32>; 3],
}

impl Probe {
    /// The 99th percentile of a histogram's magnitudes (bin centres), if any.
    pub fn percentile(histogram: &[u32], q: f64) -> Option<f64> {
        let total: u64 = histogram.iter().map(|&c| c as u64).sum();
        if total == 0 {
            return None;
        }
        let threshold = q / 100.0 * total as f64;
        let mut seen = 0u64;
        for (bin, &count) in histogram.iter().enumerate() {
            seen += count as u64;
            if seen as f64 >= threshold {
                let log = HIST_LOG_MIN + (bin as f64 + 0.5) / HIST_BINS as f64 * HIST_LOG_SPAN;
                return Some(10f64.powf(log));
            }
        }
        None
    }
}

#[derive(Clone)]
struct Pipelines {
    begin_tick: wgpu::ComputePipeline,
    begin_substep: wgpu::ComputePipeline,
    end_tick: wgpu::ComputePipeline,
    clear_counts: wgpu::ComputePipeline,
    count_cells: wgpu::ComputePipeline,
    scan_counts: wgpu::ComputePipeline,
    scatter_cells: wgpu::ComputePipeline,
    max_speed: wgpu::ComputePipeline,
    plan: wgpu::ComputePipeline,
    push: wgpu::ComputePipeline,
    copy_back: wgpu::ComputePipeline,
    deposit: wgpu::ComputePipeline,
    smooth_x: wgpu::ComputePipeline,
    smooth_y: wgpu::ComputePipeline,
    current: wgpu::ComputePipeline,
    bz_half: wgpu::ComputePipeline,
    bz_half_damp: wgpu::ComputePipeline,
    e_step: wgpu::ComputePipeline,
    damp_e: wgpu::ComputePipeline,
    gauss_rhs: wgpu::ComputePipeline,
    matmul: [wgpu::ComputePipeline; 3],
    gauss_correct: wgpu::ComputePipeline,
    finish_tick: wgpu::ComputePipeline,
    finish_sync: wgpu::ComputePipeline,
    trim: wgpu::ComputePipeline,
    probe_cell: wgpu::ComputePipeline,
    probe_grid: wgpu::ComputePipeline,
}

/// Source of GpuSim::generation.
static BUFFER_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Thread groups of 64 for `items`, as (x, y) to stay under 65535 per dimension.
fn groups(items: u32) -> (u32, u32) {
    let g = items.div_ceil(64);
    (g.min(65535), g.div_ceil(65535).max(1))
}

/// The compiled shaders and their layout, shared by every simulation on a device
/// (compiling them is the slow part of starting one).
#[derive(Clone)]
pub struct Kernels {
    layout: wgpu::BindGroupLayout,
    pipelines: Pipelines,
}

impl Kernels {
    pub fn new(gpu: &Gpu) -> Self {
        let layout = main_layout(gpu);
        let pipelines = Pipelines::new(gpu, &layout);
        Self { layout, pipelines }
    }
}

pub struct GpuSim {
    pub params: Params,
    settings: Settings,
    pub fields: wgpu::Buffer,
    pub particles: wgpu::Buffer,
    particles_out: wgpu::Buffer,
    cells: wgpu::Buffer,
    info: wgpu::Buffer,
    tables: wgpu::Buffer,
    pub stacks_buffer: wgpu::Buffer,
    state: wgpu::Buffer,
    args: wgpu::Buffer,
    uniform: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    plan_bind_group: wgpu::BindGroup,
    pipelines: Pipelines,
    /// The stacks as found in the world at the last upload, in order.
    stacks: Vec<Stack>,
    fdtd_substeps: usize,
    /// Simulated time.
    pub time: f64,
    /// Changes whenever the electron buffers are replaced (renderers rebind): unique
    /// across every simulation, so a new one never looks like the last.
    pub generation: u64,
    /// Regions sized for these many electrons and stacks; more need new buffers.
    particle_capacity: usize,
    stack_capacity: (usize, usize),
    allocated: bool,
}

impl GpuSim {
    /// The simulation of `world`, with its fields from rest (only the static field of
    /// its charges), like Simulation::new.
    pub fn new(gpu: &Gpu, world: &World) -> Self {
        Self::with_kernels(gpu, &Kernels::new(gpu), world)
    }

    pub fn with_kernels(gpu: &Gpu, kernels: &Kernels, world: &World) -> Self {
        let s = world.settings.clone();
        let m = &world.materials;
        let (cols, rows) = (m.cols, m.rows);
        let margin = s.absorbing_margin;
        let (nx, ny) = (cols + 2 * margin, rows + 2 * margin);
        let (node_count, cell_count) = (nx * ny, cols * rows);
        let tables = close_range::tables(s.electron_softening, s.smoothing_passes, s.electron_cutoff);
        let square_n = tables.square.dim().0;
        let fdtd_substeps = maxwell::substeps(&s, s.dt);
        let h = s.dt / fdtd_substeps as f64;

        let mut f = Regions::default();
        let mut p = Params {
            cols: cols as u32,
            rows: rows as u32,
            nx: nx as u32,
            ny: ny as u32,
            margin: margin as u32,
            cells: cell_count as u32,
            dt: s.dt as f32,
            max_step: s.max_step as f32,
            max_substeps: s.max_substeps,
            restitution: s.wall_restitution as f32,
            charge: world.electron_charge() as f32,
            mass: world.electron_mass() as f32,
            k: s.coulomb_k as f32,
            k_ion: (s.coulomb_k * s.cell_charge) as f32,
            cutoff: s.electron_cutoff as f32,
            reach: s.electron_cutoff.ceil() as u32,
            half: tables.half as f32,
            square_samples: SQUARE_SAMPLES as f32,
            square_n: square_n as u32,
            g_scale: (PAIR_SAMPLES as f64 / s.electron_cutoff.powi(2)) as f32,
            c2: s.speed_of_light.powi(2) as f32,
            four_pi_k: (4.0 * PI * s.coulomb_k) as f32,
            fdtd_h: h as f32,
            wall_gap: WALL_GAP,
            alpha_current: (s.dt / s.current_smoothing_time).min(1.0) as f32,
            alpha_display: if s.display_smoothing_time > 0.0 {
                1.0 - (-s.dt / s.display_smoothing_time).exp() as f32
            } else {
                1.0
            },
            inv_mu0: (s.speed_of_light.powi(2) / (4.0 * PI * s.coulomb_k)) as f32,
            trim_rate: (s.dt / s.battery_trim_time) as f32,
            gain_low: s.battery_gain_range.0 as f32,
            gain_high: s.battery_gain_range.1 as f32,
            o_ex: f.take((nx + 1) * ny),
            o_ey: f.take(nx * (ny + 1)),
            o_bz: f.take((nx + 1) * (ny + 1)),
            o_src: f.take(3 * node_count),
            o_tmp: f.take(3 * node_count),
            o_work: f.take(2 * node_count),
            o_work2: f.take(2 * node_count),
            o_phi: f.take(2 * node_count),
            o_damp_ex: f.take((nx + 1) * ny),
            o_damp_ey: f.take(nx * (ny + 1)),
            o_damp_bz: f.take((nx + 1) * (ny + 1)),
            o_static_rho: f.take(cell_count),
            o_ion_potential: f.take(cell_count),
            o_current: f.take(2 * cell_count),
            o_current_smooth: f.take(2 * cell_count),
            o_display: f.take(3 * cell_count),
            ..Params::default()
        };
        let mut t = Regions::default();
        p.o_g = t.take(tables.g.len());
        p.o_square = t.take(square_n * square_n * 3);
        p.o_sine_x = t.take(nx * nx);
        p.o_sine_y = t.take(ny * ny);
        p.o_inv_laplacian = t.take(node_count);

        let field_count = f.0 as u64;
        let fields = gpu.storage_buffer("fields", field_count * 4, wgpu::BufferUsages::empty());
        let table_buffer = gpu.storage_buffer("tables", t.0 as u64 * 4, wgpu::BufferUsages::empty());
        let info = gpu.storage_buffer("cell info", (cell_count * size_of::<CellInfo>()) as u64, wgpu::BufferUsages::empty());
        let state = gpu.storage_buffer("state", ((8 + 3 * HIST_BINS) * 4) as u64, wgpu::BufferUsages::empty());
        let args = gpu.storage_buffer("indirect args", (s.max_substeps as usize * 9 * 4) as u64, wgpu::BufferUsages::INDIRECT);
        let uniform = gpu.uniform_buffer("params", size_of::<Params>() as u64);

        // Static tables: close-range kernels, sine matrices and the inverse Laplacian.
        let mut table = vec![0f32; t.0 as usize];
        let put = |table: &mut Vec<f32>, offset: u32, values: &mut dyn Iterator<Item = f64>| {
            for (k, v) in values.enumerate() {
                table[offset as usize + k] = v as f32;
            }
        };
        put(&mut table, p.o_g, &mut tables.g.iter().copied());
        put(&mut table, p.o_square, &mut tables.square.iter().copied());
        let sine = |n: usize| {
            (0..n * n).map(move |i| (PI * ((i / n + 1) * (i % n + 1)) as f64 / (n + 1) as f64).sin())
        };
        put(&mut table, p.o_sine_x, &mut sine(nx));
        put(&mut table, p.o_sine_y, &mut sine(ny));
        let eigen = |n: usize, k: usize| 2.0 * (PI * (k + 1) as f64 / (n + 1) as f64).cos() - 2.0;
        let norm = 4.0 / ((nx + 1) * (ny + 1)) as f64;
        put(&mut table, p.o_inv_laplacian, &mut (0..node_count).map(|i| norm / (eigen(nx, i / ny) + eigen(ny, i % ny))));
        gpu.queue.write_buffer(&table_buffer, 0, bytemuck::cast_slice(&table));

        // Absorbing margin.
        let mut mx = Maxwell::new(cols, rows, &s);
        let damping = mx.damping_factors(h).clone();
        for (array, offset) in damping.iter().zip([p.o_damp_ex, p.o_damp_ey, p.o_damp_bz]) {
            let values: Vec<f32> = array.iter().map(|&v| v as f32).collect();
            gpu.queue.write_buffer(&fields, offset as u64 * 4, bytemuck::cast_slice(&values));
        }

        let Kernels { layout, pipelines } = kernels.clone();
        let placeholder = gpu.storage_buffer("placeholder", 16, wgpu::BufferUsages::empty());
        let plan_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("plan"),
            layout: &pipelines.plan.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: state.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: args.as_entire_binding() },
            ],
        });
        // Placeholders until the first upload sizes the electron and stack buffers.
        let bind_group = bind_group(
            gpu,
            &layout,
            [&uniform, &fields, &placeholder, &placeholder, &placeholder, &info, &table_buffer, &placeholder, &state],
        );
        let mut sim = Self {
            params: p,
            settings: s,
            fields,
            particles: placeholder.clone(),
            particles_out: placeholder.clone(),
            cells: placeholder.clone(),
            info,
            tables: table_buffer,
            stacks_buffer: placeholder,
            state,
            args,
            uniform,
            layout,
            bind_group,
            plan_bind_group,
            pipelines,
            stacks: vec![],
            fdtd_substeps,
            time: 0.0,
            generation: 0,
            particle_capacity: 0,
            stack_capacity: (0, 0),
            allocated: false,
        };
        sim.upload(gpu, world, true);
        sim
    }

    /// The buffers a renderer draws from.
    pub fn sources(&self) -> crate::render::Sources<'_> {
        crate::render::Sources {
            fields: &self.fields,
            o_display: self.params.o_display,
            electrons: &self.particles,
            electron_count: self.params.n,
            generation: self.generation,
        }
    }

    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    pub fn stacks(&self) -> &[Stack] {
        &self.stacks
    }

    /// Bring the GPU up to date with an edited world: its electrons (positions and
    /// velocities), materials, batteries (with their gains) and static charges. The
    /// fields carry on, corrected for Gauss's law with the new charges (like
    /// Simulation::sync), and the display average restarts.
    pub fn upload(&mut self, gpu: &Gpu, world: &World, reset_display: bool) {
        let m = &world.materials;
        let s = &self.settings;
        let p = &mut self.params;
        p.n = world.particles.len() as u32;
        self.stacks = world.stacks.clone();
        p.n_stacks = self.stacks.len() as u32;

        // Battery stacks: headers and the cells their terminals and current read.
        let mut header = vec![0u32; self.stacks.len() * STACK_HEADER];
        let mut lists: Vec<u32> = vec![];
        let flat = |(c, r): (usize, usize)| (c * m.rows + r) as u32;
        let labels = &world.connectivity.labels;
        for (i, stack) in self.stacks.iter().enumerate() {
            let h = &mut header[i * STACK_HEADER..(i + 1) * STACK_HEADER];
            let both_wired = wired(m, stack.low) && wired(m, stack.high);
            for (k, terminal) in [stack.low, stack.high].into_iter().enumerate() {
                h[2 * k] = lists.len() as u32;
                if wired(m, terminal) {
                    let island = labels[(terminal.0 as usize, terminal.1 as usize)];
                    for cell in terminal_block(m, terminal) {
                        if labels[cell] == island && m.material(cell) == Material::Conductor {
                            lists.push(flat(cell));
                        }
                    }
                }
                h[2 * k + 1] = lists.len() as u32 - h[2 * k];
            }
            h[4] = lists.len() as u32;
            lists.extend(stack.cells.iter().map(|&c| flat(c)));
            for terminal in [stack.low, stack.high] {
                if wired(m, terminal) {
                    lists.push(flat((terminal.0 as usize, terminal.1 as usize)));
                }
            }
            h[5] = lists.len() as u32 - h[4];
            h[6] = ((stack.high.0 - stack.low.0).signum() as f32).to_bits();
            h[7] = ((stack.high.1 - stack.low.1).signum() as f32).to_bits();
            h[8] = (stack.amplitude() as f32).to_bits();
            h[9] = (stack.internal_resistance as f32).to_bits();
            h[10] = both_wired as u32;
        }

        let n = world.particles.len();
        let ns = self.stacks.len();
        let needs_particles = n > self.particle_capacity || !self.allocated;
        let needs_stacks = ns > self.stack_capacity.0 || lists.len() > self.stack_capacity.1 || !self.allocated;
        self.allocated = true;
        if needs_particles {
            self.particle_capacity = (n * 5 / 4).max(1024);
        }
        if needs_stacks {
            self.stack_capacity = ((ns * 2).max(8), (lists.len() * 2).max(64));
        }
        // Regions in the cell list and stacks buffers depend on those capacities.
        let cap = self.particle_capacity;
        let cells = m.cols * m.rows;
        let mut c = Regions::default();
        p.o_counts = c.take(cells);
        p.o_start = c.take(cells + 1);
        p.o_order = c.take(cap);
        p.o_pcell = c.take(cap);
        p.o_prank = c.take(cap);
        let (stack_cap, list_cap) = self.stack_capacity;
        let mut st = Regions::default();
        p.o_gain = st.take(stack_cap);
        p.o_stack_header = st.take(stack_cap * STACK_HEADER);
        p.o_stack_cells = st.take(list_cap);
        p.o_tick_time = st.take(MAX_TICKS);
        p.o_targets = st.take(MAX_TICKS * stack_cap);
        p.o_stack_readings = st.take(stack_cap * 4);
        p.o_probe_request = st.take(4);
        p.o_probe_out = st.take(PROBE_OUT);
        if needs_particles || needs_stacks {
            if needs_particles {
                let bytes = (cap * 16) as u64;
                self.particles = gpu.storage_buffer("electrons", bytes, wgpu::BufferUsages::empty());
                self.particles_out = gpu.storage_buffer("electrons out", bytes, wgpu::BufferUsages::empty());
                self.cells = gpu.storage_buffer("cell list", c.0 as u64 * 4, wgpu::BufferUsages::empty());
            }
            if needs_stacks {
                self.stacks_buffer = gpu.storage_buffer("stacks", st.0 as u64 * 4, wgpu::BufferUsages::empty());
            }
            self.bind_group = self.create_bind_group(gpu);
            self.generation = BUFFER_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        }

        let q = &gpu.queue;
        q.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&self.params));
        let electrons: Vec<[f32; 4]> = world
            .particles
            .pos
            .iter()
            .zip(&world.particles.vel)
            .map(|(p, v)| [p[0] as f32, p[1] as f32, v[0] as f32, v[1] as f32])
            .collect();
        if !electrons.is_empty() {
            q.write_buffer(&self.particles, 0, bytemuck::cast_slice(&electrons));
        }
        let gains: Vec<u32> = self
            .stacks
            .iter()
            .map(|st| (world.battery_gain.get(st).copied().unwrap_or(1.0) as f32).to_bits())
            .collect();
        let p = &self.params;
        let words = |o: u32| o as u64 * 4;
        if !gains.is_empty() {
            q.write_buffer(&self.stacks_buffer, words(p.o_gain), bytemuck::cast_slice(&gains));
            q.write_buffer(&self.stacks_buffer, words(p.o_stack_header), bytemuck::cast_slice(&header));
        }
        if !lists.is_empty() {
            q.write_buffer(&self.stacks_buffer, words(p.o_stack_cells), bytemuck::cast_slice(&lists));
        }
        q.write_buffer(&self.info, 0, bytemuck::cast_slice(&cell_info(world, &self.stacks)));
        let to_f32 = |a: &Array2<f64>| a.iter().map(|&v| v as f32).collect::<Vec<f32>>();
        q.write_buffer(&self.fields, words(p.o_static_rho), bytemuck::cast_slice(&to_f32(&static_charge(world))));
        let mut close = CloseRange::new(s);
        let ions = close.ion_potential(m).clone();
        q.write_buffer(&self.fields, words(p.o_ion_potential), bytemuck::cast_slice(&to_f32(&ions)));
        q.write_buffer(&self.state, 4 * 5, bytemuck::bytes_of(&(reset_display as u32)));

        // Gauss's law for the new charges, the instantaneous current the batteries
        // read on the next tick, and the display.
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &self.bind_group, &[]);
            self.encode_sort(&mut pass);
            self.encode_sources_and_gauss(&mut pass, false);
            self.dispatch(&mut pass, &self.pipelines.current, self.params.cells);
            self.dispatch(&mut pass, &self.pipelines.finish_sync, self.params.cells);
            self.dispatch(&mut pass, &self.pipelines.end_tick, 1);
        }
        gpu.queue.submit([encoder.finish()]);
    }

    /// Replace the smoothed current (e.g. from a save).
    pub fn set_current(&self, gpu: &Gpu, current: &Array3<f64>) {
        let (cols, rows, _) = current.dim();
        let mut values = vec![0f32; 2 * cols * rows];
        for ((c, r, k), &v) in current.indexed_iter() {
            values[k * cols * rows + c * rows + r] = v as f32;
        }
        gpu.queue.write_buffer(&self.fields, self.params.o_current_smooth as u64 * 4, bytemuck::cast_slice(&values));
    }

    fn create_bind_group(&self, gpu: &Gpu) -> wgpu::BindGroup {
        bind_group(
            gpu,
            &self.layout,
            [
                &self.uniform, &self.fields, &self.particles, &self.particles_out, &self.cells, &self.info,
                &self.tables, &self.stacks_buffer, &self.state,
            ],
        )
    }

    fn dispatch(&self, pass: &mut wgpu::ComputePass, pipeline: &wgpu::ComputePipeline, items: u32) {
        let (x, y) = groups(items);
        pass.set_pipeline(pipeline);
        pass.dispatch_workgroups(x, y, 1);
    }

    fn dispatch_one(&self, pass: &mut wgpu::ComputePass, pipeline: &wgpu::ComputePipeline) {
        pass.set_pipeline(pipeline);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Indirect dispatch from slot `slot`, `kind` 0: electrons, 1: cells, 2: one group.
    fn dispatch_indirect(&self, pass: &mut wgpu::ComputePass, pipeline: &wgpu::ComputePipeline, slot: usize, kind: usize) {
        pass.set_pipeline(pipeline);
        pass.dispatch_workgroups_indirect(&self.args, ((slot * 3 + kind) * 12) as u64);
    }

    fn encode_sort(&self, pass: &mut wgpu::ComputePass) {
        let p = &self.pipelines;
        self.dispatch(pass, &p.clear_counts, self.params.cells);
        self.dispatch(pass, &p.count_cells, self.params.n);
        self.dispatch_one(pass, &p.scan_counts);
        self.dispatch(pass, &p.scatter_cells, self.params.n);
    }

    /// Deposit and smooth the sources, step the fields if `step`, then enforce Gauss's
    /// law (Maxwell::advance, or just its Gauss part).
    fn encode_sources_and_gauss(&self, pass: &mut wgpu::ComputePass, step: bool) {
        let (p, params) = (&self.pipelines, &self.params);
        let nodes = params.nx * params.ny;
        self.dispatch(pass, &p.deposit, nodes);
        for _ in 0..self.settings.smoothing_passes {
            self.dispatch(pass, &p.smooth_x, nodes);
            self.dispatch(pass, &p.smooth_y, nodes);
        }
        let corners = (params.nx + 1) * (params.ny + 1);
        let edges = (params.nx + 1) * params.ny + params.nx * (params.ny + 1);
        if step {
            for _ in 0..self.fdtd_substeps {
                self.dispatch(pass, &p.bz_half, corners);
                self.dispatch(pass, &p.e_step, edges);
                self.dispatch(pass, &p.bz_half_damp, corners);
                self.dispatch(pass, &p.damp_e, edges);
            }
        }
        self.dispatch(pass, &p.gauss_rhs, nodes);
        let tiles = (params.nx.div_ceil(16), params.ny.div_ceil(16));
        for pipeline in [&p.matmul[0], &p.matmul[1], &p.matmul[0], &p.matmul[2]] {
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups(tiles.0, tiles.1, 2);
        }
        self.dispatch(pass, &p.gauss_correct, edges);
    }

    /// Encode `ticks` physics ticks from the current time (Simulation::step, repeated).
    pub fn encode_ticks(&mut self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, ticks: usize) {
        assert!(ticks <= MAX_TICKS);
        if ticks == 0 {
            return;
        }
        let dt = self.settings.dt;
        let ns = self.stacks.len();
        let mut times = vec![0f32; ticks];
        let mut targets = vec![0f32; ticks * ns];
        for (j, time) in times.iter_mut().enumerate() {
            let t = self.time + j as f64 * dt;
            *time = t.rem_euclid(PERIOD) as f32;
            for (s, stack) in self.stacks.iter().enumerate() {
                targets[j * ns + s] = stack.emf_at(t) as f32;
            }
        }
        self.time += ticks as f64 * dt;
        let q = &gpu.queue;
        let p = &self.params;
        q.write_buffer(&self.stacks_buffer, p.o_tick_time as u64 * 4, bytemuck::cast_slice(&times));
        if !targets.is_empty() {
            q.write_buffer(&self.stacks_buffer, p.o_targets as u64 * 4, bytemuck::cast_slice(&targets));
        }
        q.write_buffer(&self.state, 0, bytemuck::bytes_of(&u32::MAX));

        let pl = &self.pipelines;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ticks"), timestamp_writes: None });
        pass.set_bind_group(0, &self.bind_group, &[]);
        for _ in 0..ticks {
            self.dispatch_one(&mut pass, &pl.begin_tick);
            self.encode_sort(&mut pass);
            self.dispatch(&mut pass, &pl.trim, p.n_stacks);
            self.dispatch(&mut pass, &pl.max_speed, p.n);
            pass.set_bind_group(0, &self.plan_bind_group, &[]);
            self.dispatch_one(&mut pass, &pl.plan);
            pass.set_bind_group(0, &self.bind_group, &[]);
            for slot in 0..self.settings.max_substeps as usize {
                self.dispatch_one(&mut pass, &pl.begin_substep);
                if slot > 0 {
                    // The electrons moved: sort them again.
                    self.dispatch_indirect(&mut pass, &pl.clear_counts, slot, 1);
                    self.dispatch_indirect(&mut pass, &pl.count_cells, slot, 0);
                    self.dispatch_indirect(&mut pass, &pl.scan_counts, slot, 2);
                    self.dispatch_indirect(&mut pass, &pl.scatter_cells, slot, 0);
                }
                self.dispatch_indirect(&mut pass, &pl.push, slot, 0);
                self.dispatch_indirect(&mut pass, &pl.copy_back, slot, 0);
            }
            self.encode_sort(&mut pass);
            self.dispatch(&mut pass, &pl.current, p.cells);
            self.encode_sources_and_gauss(&mut pass, true);
            self.dispatch(&mut pass, &pl.finish_tick, p.cells);
            self.dispatch_one(&mut pass, &pl.end_tick);
        }
    }

    /// Encode the info panel's probe of `cell` and copy its results into `staging`
    /// (see probe_size and parse_probe). `arrow_stride` picks the cells whose arrows
    /// the histograms of E and S count.
    pub fn encode_probe(
        &self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        cell: Option<(usize, usize)>,
        arrow_stride: u32,
        staging: &wgpu::Buffer,
    ) {
        let p = &self.params;
        let index = cell.map_or(u32::MAX, |(c, r)| (c * p.rows as usize + r) as u32);
        gpu.queue.write_buffer(&self.stacks_buffer, p.o_probe_request as u64 * 4, bytemuck::cast_slice(&[index, arrow_stride]));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("probe"), timestamp_writes: None });
            pass.set_bind_group(0, &self.bind_group, &[]);
            // The display average may be newer than the cell list: count electrons afresh.
            self.encode_sort(&mut pass);
            self.dispatch_one(&mut pass, &self.pipelines.probe_cell);
            self.dispatch_one(&mut pass, &self.pipelines.probe_grid);
        }
        encoder.copy_buffer_to_buffer(&self.stacks_buffer, p.o_probe_out as u64 * 4, staging, 0, (PROBE_OUT * 4) as u64);
        encoder.copy_buffer_to_buffer(&self.state, 8 * 4, staging, (PROBE_OUT * 4) as u64, (3 * HIST_BINS * 4) as u64);
    }

    pub fn probe_size() -> u64 {
        ((PROBE_OUT + 3 * HIST_BINS) * 4) as u64
    }

    pub fn parse_probe(&self, bytes: &[u8], cell: Option<(usize, usize)>) -> Probe {
        let words: &[u32] = bytemuck::cast_slice(bytes);
        let f = |k: usize| f32::from_bits(words[k]) as f64;
        let hist = |layer: usize| words[PROBE_OUT + layer * HIST_BINS..PROBE_OUT + (layer + 1) * HIST_BINS].to_vec();
        let n_wires = f(18);
        Probe {
            cell: cell.filter(|_| f(0) != 0.0),
            electrons: f(1) as usize,
            potential: f(2),
            e: [f(3), f(4)],
            b: f(5),
            s: [f(6), f(7)],
            current: [f(8), f(9)],
            wire_current: (f(10), f(11) as usize),
            stack_state: f(12) as u32,
            v_low: f(13),
            v_high: f(14),
            stack_current: f(15),
            gain: f(16),
            mean_e_in_wires: (n_wires > 0.0).then(|| f(17)),
            histograms: [hist(0), hist(1), hist(2)],
        }
    }

    /// Byte ranges to read back for a save or an edit: electrons, then the gains, then
    /// the smoothed current. Copy them into one staging buffer with encode_state_copy.
    pub fn state_size(&self) -> u64 {
        let p = &self.params;
        (p.n as u64 * 16) + (self.stacks.len() as u64 * 4).next_multiple_of(16) + 2 * p.cells as u64 * 4
    }

    pub fn encode_state_copy(&self, encoder: &mut wgpu::CommandEncoder, staging: &wgpu::Buffer) {
        let p = &self.params;
        let electrons = p.n as u64 * 16;
        let gains = (self.stacks.len() as u64 * 4).next_multiple_of(16);
        if electrons > 0 {
            encoder.copy_buffer_to_buffer(&self.particles, 0, staging, 0, electrons);
        }
        if gains > 0 {
            encoder.copy_buffer_to_buffer(&self.stacks_buffer, p.o_gain as u64 * 4, staging, electrons, gains);
        }
        encoder.copy_buffer_to_buffer(&self.fields, p.o_current_smooth as u64 * 4, staging, electrons + gains, 2 * p.cells as u64 * 4);
    }

    /// Apply a state read back with encode_state_copy to the CPU world: electron
    /// positions and velocities, battery gains. Returns the smoothed current.
    pub fn apply_state(&self, bytes: &[u8], world: &mut World) -> Array3<f64> {
        let p = &self.params;
        let n = p.n as usize;
        let floats: &[f32] = bytemuck::cast_slice(bytes);
        let particles = &mut world.particles;
        if particles.len() == n {
            for i in 0..n {
                particles.pos[i] = [floats[4 * i] as f64, floats[4 * i + 1] as f64];
                particles.vel[i] = [floats[4 * i + 2] as f64, floats[4 * i + 3] as f64];
            }
        }
        let gains = 4 * n;
        world.battery_gain.clear();
        for (s, stack) in self.stacks.iter().enumerate() {
            if world.stacks.contains(stack) {
                world.battery_gain.insert(stack.clone(), floats[gains + s] as f64);
            }
        }
        let current = gains + self.stacks.len().next_multiple_of(4);
        let (cols, rows, cells) = (p.cols as usize, p.rows as usize, p.cells as usize);
        Array3::from_shape_fn((cols, rows, 2), |(c, r, k)| floats[current + k * cells + c * rows + r] as f64)
    }

    // --- Native read back (tests and tools) -------------------------------------------

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_state(&self, gpu: &Gpu, world: &mut World) -> Array3<f64> {
        let size = self.state_size();
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("state"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        self.encode_state_copy(&mut encoder, &staging);
        gpu.queue.submit([encoder.finish()]);
        staging.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let bytes = staging.get_mapped_range(..).expect("mapped").to_vec();
        self.apply_state(&bytes, world)
    }

    /// Run `ticks` ticks and wait for them (tests and tools).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run_ticks(&mut self, gpu: &Gpu, ticks: usize) {
        for chunk in (0..ticks).collect::<Vec<_>>().chunks(MAX_TICKS) {
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            self.encode_ticks(gpu, &mut encoder, chunk.len());
            gpu.queue.submit([encoder.finish()]);
        }
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_probe(&self, gpu: &Gpu, cell: Option<(usize, usize)>, arrow_stride: u32) -> Probe {
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe"),
            size: Self::probe_size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        self.encode_probe(gpu, &mut encoder, cell, arrow_stride, &staging);
        gpu.queue.submit([encoder.finish()]);
        staging.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let bytes = staging.get_mapped_range(..).expect("mapped").to_vec();
        self.parse_probe(&bytes, cell)
    }

    /// A (rows-major) visible-grid view of a full node array region, as f64 (nx, ny).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_region(&self, gpu: &Gpu, offset: u32, len: usize) -> Vec<f64> {
        gpu.read_f32(&self.fields, offset as u64, len as u64).into_iter().map(|v| v as f64).collect()
    }

    /// Ex, Ey, Bz as arrays shaped like Maxwell's.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_yee(&self, gpu: &Gpu) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let (nx, ny) = (self.params.nx as usize, self.params.ny as usize);
        let get = |o: u32, w: usize, h: usize| Array2::from_shape_vec((w, h), self.read_region(gpu, o, w * h)).unwrap();
        (get(self.params.o_ex, nx + 1, ny), get(self.params.o_ey, nx, ny + 1), get(self.params.o_bz, nx + 1, ny + 1))
    }

    /// The potential of the charges at the nodes (nx, ny), like Maxwell::potential.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_potential(&self, gpu: &Gpu) -> Array2<f64> {
        let (nx, ny) = (self.params.nx as usize, self.params.ny as usize);
        Array2::from_shape_vec((nx, ny), self.read_region(gpu, self.params.o_phi, nx * ny)).unwrap()
    }

    /// The displayed (time-averaged) E (cols, rows, 2) and B (cols, rows).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_display(&self, gpu: &Gpu) -> (Array3<f64>, Array2<f64>) {
        let (cols, rows) = (self.params.cols as usize, self.params.rows as usize);
        let cells = cols * rows;
        let v = self.read_region(gpu, self.params.o_display, 3 * cells);
        let e = Array3::from_shape_fn((cols, rows, 2), |(c, r, k)| v[k * cells + c * rows + r]);
        let b = Array2::from_shape_fn((cols, rows), |(c, r)| v[2 * cells + c * rows + r]);
        (e, b)
    }

    /// The instantaneous current per cell (cols, rows, 2), like physics::current_density.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_current(&self, gpu: &Gpu) -> Array3<f64> {
        let (cols, rows) = (self.params.cols as usize, self.params.rows as usize);
        let cells = cols * rows;
        let v = self.read_region(gpu, self.params.o_current, 2 * cells);
        Array3::from_shape_fn((cols, rows, 2), |(c, r, k)| v[k * cells + c * rows + r])
    }
}

/// Charge of the ions and test charges on the visible nodes (cols, rows), unsmoothed.
fn static_charge(world: &World) -> Array2<f64> {
    let m = &world.materials;
    let mut rho = ion_charge(m, world.settings.cell_charge);
    if !world.fixed_charge.is_empty() {
        rho += &CloudInCell::new(&world.fixed_pos, m.cols, m.rows).deposit(&world.fixed_charge);
    }
    rho
}

fn cell_info(world: &World, stacks: &[Stack]) -> Vec<CellInfo> {
    let m: &Materials = &world.materials;
    let damping = damping_rates(world);
    let mut stack_of = Array2::<u32>::zeros((m.cols, m.rows));
    for (i, stack) in stacks.iter().enumerate() {
        for &cell in &stack.cells {
            stack_of[cell] = i as u32 + 1;
        }
    }
    m.kind
        .indexed_iter()
        .map(|(cell, &kind)| {
            let conductive = kind != Material::Vacuum as u8;
            let wire = kind == Material::Conductor as u8;
            CellInfo {
                damping: damping[cell] as f32,
                emf: m.emf[cell] as f32,
                frequency: m.frequency[cell] as f32,
                density: m.density[cell] as f32,
                flags: conductive as u32 | (wire as u32) << 1 | (m.orientation[cell] as u32 & 3) << 2,
                stack: stack_of[cell],
                _pad: [0; 2],
            }
        })
        .collect()
}

fn bind_group(gpu: &Gpu, layout: &wgpu::BindGroupLayout, buffers: [&wgpu::Buffer; 9]) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
        .collect();
    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("simulation"), layout, entries: &entries })
}

fn main_layout(gpu: &Gpu) -> wgpu::BindGroupLayout {
    let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let mut entries = vec![wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }];
    for (binding, read_only) in [(1, false), (2, false), (3, false), (4, false), (5, true), (6, true), (7, false), (8, false)] {
        entries.push(storage(binding, read_only));
    }
    gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("simulation"), entries: &entries })
}

/// The simulation's shader sources: (label, WGSL).
pub fn shader_sources() -> [(&'static str, String); 3] {
    let common = [Params::wgsl(), include_str!("shaders/common.wgsl").to_string()].concat();
    [
        (
            "simulation",
            [common.clone(), include_str!("shaders/particles.wgsl").to_string(), include_str!("shaders/fields.wgsl").to_string()].concat(),
        ),
        (
            "poisson",
            [common, "fn nodes() -> u32 { return P.nx * P.ny; }\n".to_string(), include_str!("shaders/poisson.wgsl").to_string()].concat(),
        ),
        ("plan", [Params::wgsl(), include_str!("shaders/plan.wgsl").to_string()].concat()),
    ]
}

impl Pipelines {
    fn new(gpu: &Gpu, layout: &wgpu::BindGroupLayout) -> Self {
        let [(_, source), (_, poisson_source), (_, plan_source)] = shader_sources();
        let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("simulation"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let poisson = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("poisson"),
            source: wgpu::ShaderSource::Wgsl(poisson_source.into()),
        });
        let plan_module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("plan"),
            source: wgpu::ShaderSource::Wgsl(plan_source.into()),
        });
        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("simulation"),
            bind_group_layouts: &[Some(layout)],
            immediate_size: 0,
        });
        let make = |module: &wgpu::ShaderModule, entry: &str, constants: &[(&str, f64)]| {
            gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions { constants, ..Default::default() },
                cache: None,
            })
        };
        let main = |entry: &str| make(&module, entry, &[]);
        Self {
            begin_tick: main("begin_tick"),
            begin_substep: main("begin_substep"),
            end_tick: main("end_tick"),
            clear_counts: main("clear_counts"),
            count_cells: main("count_cells"),
            scan_counts: main("scan_counts"),
            scatter_cells: main("scatter_cells"),
            max_speed: main("max_speed"),
            plan: gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("plan"),
                layout: None,
                module: &plan_module,
                entry_point: Some("plan"),
                compilation_options: Default::default(),
                cache: None,
            }),
            push: main("push"),
            copy_back: main("copy_back"),
            deposit: main("deposit"),
            smooth_x: main("smooth_x"),
            smooth_y: main("smooth_y"),
            current: main("current"),
            bz_half: main("bz_half"),
            bz_half_damp: main("bz_half_damp"),
            e_step: main("e_step"),
            damp_e: main("damp_e"),
            gauss_rhs: main("gauss_rhs"),
            matmul: [
                make(&poisson, "matmul", &[("MODE", 0.0)]),
                make(&poisson, "matmul", &[("MODE", 1.0)]),
                make(&poisson, "matmul", &[("MODE", 3.0)]),
            ],
            gauss_correct: main("gauss_correct"),
            finish_tick: main("finish_tick"),
            finish_sync: main("finish_sync"),
            trim: main("trim"),
            probe_cell: main("probe_cell"),
            probe_grid: main("probe_grid"),
        }
    }
}
