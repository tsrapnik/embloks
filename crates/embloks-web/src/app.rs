//! The app in the browser: the platform-independent `App` (embloks-core) driven by
//! DOM events, its physics on the GPU (or the CPU reference, with ?backend=cpu),
//! drawn with WebGPU, and the HTML panels around it.
//!
//! On the GPU, electrons move without the CPU seeing them. Painting needs their
//! current state (it adds, removes and moves electrons), so a stroke first reads
//! them back: input that edits waits in a queue until that arrives (about a
//! frame), and the physics holds still while the stroke goes on.

use std::cell::{Cell as StdCell, RefCell};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use embloks_core::controller::{App, Button, CellReading, Effect, HudMode, Key, Mods, StackReading, StatusReading};
use embloks_core::fields::poynting;
use embloks_core::layers::{LayerControl, LayerId};
use embloks_core::materials::Cell;
use embloks_core::storage;
use embloks_gpu::Gpu;
use embloks_gpu::render::{Frame, Renderer, Sources, arrow_stride};
use embloks_gpu::sim::{GpuSim, Kernels, MAX_TICKS, Probe};
use wasm_bindgen::JsCast;

use crate::dom;
use crate::saves;

/// Something the user did, applied in order (see the module notes).
pub enum Input {
    Down(Button, f64, f64),
    Move(f64, f64),
    Up(Button),
    Leave,
    Wheel(i32),
    Key(Key, Mods),
    Layer(LayerId, LayerControl),
    Action(String, String),
}

/// Results of asynchronous work (browser storage, files), applied on the next frame.
pub enum Task {
    Loaded(u8, Result<Option<Vec<u8>>, String>),
    Imported(Result<Vec<u8>, String>),
    Message(String),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    Idle,
    /// Wanted: encode the copy with the next frame.
    Requested,
    Encoded,
    Mapping,
    Ready,
}

/// A GPU-to-CPU copy through a staging buffer, polled frame by frame.
struct Readback {
    buffer: Option<wgpu::Buffer>,
    status: Rc<StdCell<Status>>,
}

impl Readback {
    fn new() -> Self {
        Self { buffer: None, status: Rc::new(StdCell::new(Status::Idle)) }
    }

    fn status(&self) -> Status {
        self.status.get()
    }

    /// A staging buffer of at least `size` bytes.
    fn staging(&mut self, gpu: &Gpu, size: u64) -> wgpu::Buffer {
        if self.buffer.as_ref().is_none_or(|b| b.size() < size) {
            self.buffer = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("read back"),
                size: size.next_multiple_of(16),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        self.buffer.clone().unwrap()
    }

    /// After the encoder with the copy was submitted.
    fn start_mapping(&mut self) {
        if self.status() != Status::Encoded {
            return;
        }
        self.status.set(Status::Mapping);
        let status = self.status.clone();
        self.buffer.as_ref().unwrap().map_async(wgpu::MapMode::Read, .., move |result| {
            status.set(if result.is_ok() { Status::Ready } else { Status::Idle });
        });
    }

    /// The bytes, once they arrived.
    fn take(&mut self) -> Option<Vec<u8>> {
        if self.status() != Status::Ready {
            return None;
        }
        let buffer = self.buffer.as_ref().unwrap();
        let bytes = buffer.get_mapped_range(..).ok()?.to_vec();
        buffer.unmap();
        self.status.set(Status::Idle);
        Some(bytes)
    }
}

struct GpuBackend {
    kernels: Kernels,
    sim: GpuSim,
    /// The app's world generation and scene key the GPU last took.
    generation: u64,
    scene_key: (u64, u64),
    /// Whether the CPU world's electrons match the GPU's (no tick since they were read).
    fresh: bool,
    state: Readback,
    probe: Readback,
    probe_cell: Option<Cell>,
}

/// The CPU reference simulation, with buffers it fills for the renderer each frame.
struct CpuBackend {
    fields: wgpu::Buffer,
    electrons: wgpu::Buffer,
    generation: u64,
}

enum Backend {
    Gpu(Box<GpuBackend>),
    Cpu(CpuBackend),
}

pub struct Dom {
    pub canvas: web_sys::HtmlCanvasElement,
    pub info: web_sys::HtmlElement,
    pub layers: web_sys::HtmlElement,
    pub toolbar: web_sys::HtmlElement,
    pub import: web_sys::HtmlInputElement,
}

pub struct WebApp {
    pub app: App,
    /// Kept alive: with the instance gone, WebGPU rejects buffer mapping.
    _instance: (wgpu::Instance, wgpu::Adapter),
    gpu: Gpu,
    /// The canvas, or None when drawing offscreen (?offscreen: for automated tests on
    /// software GPUs that can't present to a canvas).
    surface: Option<wgpu::Surface<'static>>,
    offscreen: Option<wgpu::Texture>,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
    backend: Backend,
    pub dom: Dom,
    pub pixel_ratio: f64,
    last_time: Option<f64>,
    fps: f64,
    caption_time: f64,
    shown: [String; 3],
    pub erase_mode: bool,
    slot: u8,
    pub queue: Vec<Input>,
    pub tasks: Vec<Task>,
    /// Offscreen frames read back for tests: (padded row bytes, width, height).
    snapshot: Readback,
    snapshot_shape: (u32, u32, u32),
    last_snapshot: Option<Vec<u8>>,
    /// Frames submitted to the GPU and not finished yet. A slow GPU must not fall
    /// ever further behind: beyond MAX_FRAMES_IN_FLIGHT frames are skipped (the
    /// simulation then runs slower than chosen, as on the desktop).
    in_flight: Arc<AtomicU32>,
}

const MAX_FRAMES_IN_FLIGHT: u32 = 2;

impl WebApp {
    #[allow(clippy::too_many_arguments)]
    pub fn new(app: App, instance: (wgpu::Instance, wgpu::Adapter), gpu: Gpu, surface: Option<wgpu::Surface<'static>>, config: wgpu::SurfaceConfiguration, dom: Dom, cpu: bool, pixel_ratio: f64) -> Self {
        let renderer = Renderer::new(&gpu, config.format);
        let backend = if cpu {
            Backend::Cpu(CpuBackend {
                fields: gpu.storage_buffer("display", 16, wgpu::BufferUsages::empty()),
                electrons: gpu.storage_buffer("electrons", 16, wgpu::BufferUsages::empty()),
                generation: 0,
            })
        } else {
            let kernels = Kernels::new(&gpu);
            let sim = GpuSim::with_kernels(&gpu, &kernels, app.world());
            Backend::Gpu(Box::new(GpuBackend {
                kernels,
                sim,
                generation: app.world_generation,
                scene_key: app.world().scene_key(),
                fresh: true,
                state: Readback::new(),
                probe: Readback::new(),
                probe_cell: None,
            }))
        };
        let offscreen = surface.is_none().then(|| offscreen_texture(&gpu, &config));
        Self {
            app,
            _instance: instance,
            gpu,
            surface,
            offscreen,
            config,
            renderer,
            backend,
            dom,
            pixel_ratio,
            last_time: None,
            fps: 0.0,
            caption_time: 0.0,
            shown: Default::default(),
            erase_mode: false,
            slot: 1,
            queue: vec![],
            tasks: vec![],
            snapshot: Readback::new(),
            snapshot_shape: (0, 0, 0),
            last_snapshot: None,
            in_flight: Arc::new(AtomicU32::new(0)),
        }
    }

    pub fn backend_name(&self) -> &'static str {
        match self.backend {
            Backend::Gpu(_) => "GPU",
            Backend::Cpu(_) => "CPU",
        }
    }

    // --- Input ----------------------------------------------------------------------

    /// Whether `input` needs the electrons' current state (it paints or saves).
    fn needs_state(&self, input: &Input) -> bool {
        match input {
            Input::Down(..) => true,
            Input::Key(Key::Digit(_), mods) => mods.ctrl,
            Input::Action(name, _) => name == "save" || name == "export",
            _ => false,
        }
    }

    fn process_queue(&mut self, this: &Rc<RefCell<WebApp>>) {
        while !self.queue.is_empty() {
            let needs_state = self.needs_state_at_front();
            if let Backend::Gpu(g) = &mut self.backend
                && !g.fresh
                && needs_state
            {
                if g.state.status() == Status::Idle {
                    g.state.status.set(Status::Requested);
                }
                return;
            }
            let input = self.queue.remove(0);
            self.apply(input, this);
        }
    }

    fn needs_state_at_front(&self) -> bool {
        self.needs_state(&self.queue[0])
    }

    fn apply(&mut self, input: Input, this: &Rc<RefCell<WebApp>>) {
        match input {
            Input::Down(button, x, y) => {
                let button = if self.erase_mode { Button::Erase } else { button };
                self.app.pointer_down(button, x, y);
            }
            Input::Move(x, y) => self.app.pointer_move(x, y),
            Input::Up(button) => {
                // In erase mode a paint stroke was started as an erase stroke.
                self.app.pointer_up(button);
                self.app.pointer_up(Button::Erase);
            }
            Input::Leave => self.app.pointer_leave(),
            Input::Wheel(steps) => self.app.wheel(steps),
            Input::Key(key, mods) => {
                if let Some(effect) = self.app.handle_key(key, mods) {
                    self.effect(effect, this);
                }
            }
            Input::Layer(id, control) => self.app.layers[id].apply(control),
            Input::Action(name, value) => self.action(&name, &value, this),
        }
    }

    fn press(&mut self, key: Key, this: &Rc<RefCell<WebApp>>) {
        self.apply(Input::Key(key, Mods::default()), this);
    }

    fn action(&mut self, name: &str, value: &str, this: &Rc<RefCell<WebApp>>) {
        let key = match name {
            "tool-1" => Some(Key::Digit(1)),
            "tool-2" => Some(Key::Digit(2)),
            "tool-3" => Some(Key::Digit(3)),
            "value-" => Some(Key::Down),
            "value+" => Some(Key::Up),
            "rotate" => Some(Key::Char('r')),
            "frequency" => Some(Key::Char('f')),
            "brush-" => Some(Key::Char('[')),
            "brush+" => Some(Key::Char(']')),
            "pause" => Some(Key::Space),
            "speed-" => Some(Key::Char(',')),
            "speed+" => Some(Key::Char('.')),
            "clear" => Some(Key::Char('c')),
            "density-" => Some(Key::Char('-')),
            "density+" => Some(Key::Char('=')),
            "help" => Some(Key::Char('h')),
            "scene" => value.parse::<u8>().ok().map(|n| Key::Function(n + 1)),
            _ => None,
        };
        if let Some(key) = key {
            return self.press(key, this);
        }
        match name {
            "erase" => self.erase_mode = !self.erase_mode,
            "slot" => self.slot = value.parse().unwrap_or(self.slot),
            "save" => self.effect(Effect::SaveSlot(self.slot), this),
            "load" => self.effect(Effect::LoadSlot(self.slot), this),
            "export" => {
                self.sync_time();
                if let Err(e) = saves::download(&storage::save(&self.app), "scene.embloks") {
                    self.app.show_message(&format!("Could not export: {e:?}"));
                }
            }
            "import" => self.dom.import.click(),
            _ => {}
        }
    }

    fn effect(&mut self, effect: Effect, this: &Rc<RefCell<WebApp>>) {
        match effect {
            Effect::SaveSlot(slot) => {
                self.sync_time();
                let bytes = self.app.save_slot(slot);
                let this = this.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    if let Err(e) = saves::put(slot, &bytes).await {
                        this.borrow_mut().tasks.push(Task::Message(format!("Could not save slot {slot}: {e}")));
                    }
                });
            }
            Effect::LoadSlot(slot) => {
                let this = this.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    let result = saves::get(slot).await;
                    this.borrow_mut().tasks.push(Task::Loaded(slot, result));
                });
            }
        }
    }

    fn run_tasks(&mut self) {
        for task in std::mem::take(&mut self.tasks) {
            match task {
                Task::Loaded(slot, Ok(bytes)) => self.app.load_slot(slot, bytes.as_deref()),
                Task::Loaded(slot, Err(e)) => self.app.show_message(&format!("Could not load slot {slot}: {e}")),
                Task::Imported(Ok(bytes)) => match storage::load(&mut self.app, &bytes) {
                    Ok(()) => self.app.show_message("Opened the scene file"),
                    Err(e) => self.app.show_message(&format!("Could not open the file: {e}")),
                },
                Task::Imported(Err(e)) => self.app.show_message(&format!("Could not open the file: {e}")),
                Task::Message(text) => self.app.show_message(&text),
            }
        }
    }

    /// The app's clock follows the GPU's simulated time.
    fn sync_time(&mut self) {
        if let Backend::Gpu(g) = &self.backend {
            self.app.sim.time = g.sim.time;
        }
    }

    // --- Frame ----------------------------------------------------------------------

    pub fn frame(&mut self, this: &Rc<RefCell<WebApp>>, now_ms: f64) {
        let now = now_ms / 1000.0;
        let elapsed = self.last_time.map_or(0.0, |last| (now - last).clamp(0.0, 0.25));
        self.last_time = Some(now);
        if elapsed > 0.0 {
            self.fps += (1.0 / elapsed - self.fps) * 0.1;
        }
        self.resize();
        self.run_tasks();
        if let Some(bytes) = self.snapshot.take() {
            let (row, width, height) = self.snapshot_shape;
            let mut pixels = Vec::with_capacity((width * height * 4) as usize);
            for y in 0..height as usize {
                pixels.extend_from_slice(&bytes[y * row as usize..y * row as usize + width as usize * 4]);
            }
            self.last_snapshot = Some(pixels);
        }
        self.collect_readbacks();
        self.process_queue(this);
        self.sync_backend();

        if self.in_flight.load(Ordering::Relaxed) < MAX_FRAMES_IN_FLIGHT {
            let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
            self.advance(elapsed, &mut encoder);
            self.draw(&mut encoder);
        }
        self.update_panels(now);
    }

    fn resize(&mut self) {
        let canvas = &self.dom.canvas;
        let ratio = web_sys::window().unwrap().device_pixel_ratio();
        let width = ((canvas.client_width() as f64 * ratio).round() as u32).max(1);
        let height = ((canvas.client_height() as f64 * ratio).round() as u32).max(1);
        if (width, height) != (self.config.width, self.config.height) {
            canvas.set_width(width);
            canvas.set_height(height);
            self.config.width = width;
            self.config.height = height;
            match &self.surface {
                Some(surface) => surface.configure(&self.gpu.device, &self.config),
                None => self.offscreen = Some(offscreen_texture(&self.gpu, &self.config)),
            }
            self.pixel_ratio = ratio;
            self.app.resize(width as i32, height as i32);
        }
    }

    fn collect_readbacks(&mut self) {
        let Backend::Gpu(g) = &mut self.backend else { return };
        if let Some(bytes) = g.state.take() {
            let world = &mut self.app.sim.world;
            if world.particles.len() == g.sim.params.n as usize {
                self.app.current = g.sim.apply_state(&bytes, world);
                g.fresh = true;
            }
        }
        if let Some(bytes) = g.probe.take() {
            let probe = g.sim.parse_probe(&bytes, g.probe_cell);
            follow_gains(&mut self.app, &probe);
            if self.app.hud_mode != HudMode::Hidden && probe.cell == self.app.hovered_cell {
                let sections = gpu_hud(&mut self.app, &probe);
                self.show_info(dom::info_html(&sections));
            }
        }
    }

    /// Hand world changes to the GPU: a new world starts over, an edited one is uploaded.
    fn sync_backend(&mut self) {
        let Backend::Gpu(g) = &mut self.backend else { return };
        let world = self.app.world();
        if self.app.world_generation != g.generation {
            g.sim = GpuSim::with_kernels(&self.gpu, &g.kernels, world);
            g.sim.time = self.app.sim.time;
            g.sim.set_current(&self.gpu, &self.app.current);
        } else if world.scene_key() != g.scene_key {
            g.sim.upload(&self.gpu, world, true);
        } else {
            return;
        }
        g.generation = self.app.world_generation;
        g.scene_key = world.scene_key();
        g.fresh = true;
    }

    fn advance(&mut self, elapsed: f64, encoder: &mut wgpu::CommandEncoder) {
        match &mut self.backend {
            Backend::Gpu(g) => {
                // Hold still while the CPU needs (or edits) the electrons.
                let waiting = g.state.status() != Status::Idle || !self.queue.is_empty() || self.app.stroking();
                if waiting {
                    return;
                }
                let ticks = self.app.take_ticks(elapsed).min(MAX_TICKS);
                if ticks > 0 {
                    g.sim.encode_ticks(&self.gpu, encoder, ticks);
                    g.fresh = false;
                    self.app.tick_count += ticks as u64;
                    self.app.sim.time = g.sim.time;
                }
            }
            Backend::Cpu(_) => {
                self.app.advance(elapsed);
            }
        }
    }

    fn draw(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let texture = self.surface.as_ref().and_then(|surface| match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Some(t),
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                surface.configure(&self.gpu.device, &self.config);
                None
            }
            _ => None,
        });
        let target = match (&texture, &self.offscreen) {
            (Some(texture), _) => Some(texture.texture.create_view(&Default::default())),
            (None, Some(offscreen)) => Some(offscreen.create_view(&Default::default())),
            _ => None,
        };
        let brush = self
            .app
            .hovered_cell
            .map(|cell| self.app.brush.footprint(cell, self.app.grid.cols, self.app.grid.rows))
            .unwrap_or_default();
        self.renderer.update_world(&self.gpu, self.app.world(), &self.app.settings);
        let size = (self.config.width, self.config.height);
        let stride = arrow_stride(&self.app.grid, &self.app.settings);
        if let Backend::Cpu(c) = &mut self.backend {
            upload_cpu_frame(&self.gpu, c, &mut self.app);
        }
        if let Some(view) = &target {
            let frame = Frame {
                grid: self.app.grid,
                settings: &self.app.settings,
                layers: &self.app.layers,
                brush: &brush,
                pixel_ratio: self.pixel_ratio as f32,
            };
            let sources = match &self.backend {
                Backend::Gpu(g) => g.sim.sources(),
                Backend::Cpu(c) => Sources {
                    fields: &c.fields,
                    o_display: 0,
                    electrons: &c.electrons,
                    electron_count: self.app.world().particles.len() as u32,
                    generation: c.generation,
                },
            };
            self.renderer.render(&self.gpu, encoder, view, size, &sources, &frame);
        }
        if let Backend::Gpu(g) = &mut self.backend {
            if g.probe.status() == Status::Idle {
                let staging = g.probe.staging(&self.gpu, GpuSim::probe_size());
                g.sim.encode_probe(&self.gpu, encoder, self.app.hovered_cell, stride, &staging);
                g.probe_cell = self.app.hovered_cell;
                g.probe.status.set(Status::Encoded);
            }
            if g.state.status() == Status::Requested {
                let staging = g.state.staging(&self.gpu, g.sim.state_size());
                g.sim.encode_state_copy(encoder, &staging);
                g.state.status.set(Status::Encoded);
            }
        }
        if let (Status::Requested, Some(texture)) = (self.snapshot.status(), &self.offscreen) {
            let (width, height) = (texture.width(), texture.height());
            let row = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
            let staging = self.snapshot.staging(&self.gpu, (row * height) as u64);
            encoder.copy_texture_to_buffer(
                texture.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None },
                },
                texture.size(),
            );
            self.snapshot_shape = (row, width, height);
            self.snapshot.status.set(Status::Encoded);
        }
        let finished = std::mem::replace(encoder, self.gpu.device.create_command_encoder(&Default::default()));
        self.gpu.queue.submit([finished.finish()]);
        self.in_flight.fetch_add(1, Ordering::Relaxed);
        let in_flight = self.in_flight.clone();
        self.gpu.queue.on_submitted_work_done(move || {
            in_flight.fetch_sub(1, Ordering::Relaxed);
        });
        if let Backend::Gpu(g) = &mut self.backend {
            g.probe.start_mapping();
            g.state.start_mapping();
        }
        self.snapshot.start_mapping();
        if let Some(texture) = texture {
            self.gpu.queue.present(texture);
        }
    }

    fn show_info(&mut self, html: String) {
        if self.shown[0] != html {
            self.dom.info.set_inner_html(&html);
            self.shown[0] = html;
        }
    }

    fn update_panels(&mut self, now: f64) {
        self.dom.info.set_hidden(self.app.hud_mode == HudMode::Hidden);
        if matches!(self.backend, Backend::Cpu(_)) && self.app.hud_mode != HudMode::Hidden {
            let sections = self.app.hud();
            self.show_info(dom::info_html(&sections));
        }
        self.dom.layers.set_hidden(!self.app.layer_panel_visible);
        let layers = dom::layers_html(&self.app.layers);
        if self.shown[1] != layers {
            self.dom.layers.set_inner_html(&layers);
            self.shown[1] = layers;
        }
        let toolbar = dom::toolbar_html(&self.app, self.erase_mode, self.slot);
        if self.shown[2] != toolbar {
            self.dom.toolbar.set_inner_html(&toolbar);
            self.shown[2] = toolbar;
        }
        if now - self.caption_time > 0.5 {
            self.caption_time = now;
            let document = web_sys::window().unwrap().document().unwrap();
            document.set_title(&format!("{} | {}", self.app.caption(self.fps), self.backend_name()));
        }
    }
}

/// The layers' auto gain, from the GPU's histograms (Layer::follow on the CPU).
fn follow_gains(app: &mut App, probe: &Probe) {
    let s = &app.settings;
    let highs = [s.e_field_max, s.b_field_max, s.s_field_max];
    for (k, id) in [LayerId::E, LayerId::B, LayerId::S].into_iter().enumerate() {
        let layer = &mut app.layers[id];
        if layer.auto && layer.visible
            && let Some(p99) = Probe::percentile(&probe.histograms[k], 99.0)
        {
            layer.ease_towards(highs[k] / p99);
        }
    }
}

/// The info panel from the GPU's probe.
fn gpu_hud(app: &mut App, probe: &Probe) -> Vec<embloks_core::controller::HudSection> {
    let status = StatusReading { mean_e_in_wires: probe.mean_e_in_wires };
    let reading = probe.cell.map(|cell| {
        let world = app.world();
        let conductive = world.materials.is_conductive(cell);
        let stack = (probe.stack_state > 0)
            .then(|| world.stacks.iter().find(|st| st.cells.contains(&cell)))
            .flatten()
            .map(|stack| {
                let sign = if stack.emf >= 0.0 { 1.0 } else { -1.0 };
                StackReading {
                    emf: stack.emf.abs(),
                    is_ac: stack.is_ac(),
                    voltage: (probe.stack_state == 2).then_some(sign * (probe.v_high - probe.v_low)),
                    current: sign * probe.stack_current,
                    gain: probe.gain,
                }
            });
        let reading = CellReading {
            electrons: probe.electrons,
            potential: probe.potential,
            e: probe.e,
            b: probe.b,
            s: probe.s,
            current: conductive.then_some(probe.current),
            wire_current: conductive.then_some(probe.wire_current),
            stack,
        };
        (cell, reading)
    });
    app.hud_sections(&status, reading.as_ref().map(|(c, r)| (*c, r)))
}

/// CPU backend: the display fields and electrons into the renderer's buffers.
fn upload_cpu_frame(gpu: &Gpu, c: &mut CpuBackend, app: &mut App) {
    let (e, b) = app.display_fields();
    let (cols, rows) = (app.grid.cols, app.grid.rows);
    let cells = cols * rows;
    let mut display = vec![0f32; 3 * cells];
    for ((col, row), &bz) in b.indexed_iter() {
        let i = col * rows + row;
        display[i] = e[(col, row, 0)] as f32;
        display[cells + i] = e[(col, row, 1)] as f32;
        display[2 * cells + i] = bz as f32;
    }
    // Auto gain from the magnitudes the layers show (Layer::intensity).
    let stride = arrow_stride(&app.grid, &app.settings) as usize;
    let s_grid = poynting(&e, &b, &app.settings);
    let sampled = |grid: &ndarray::Array3<f64>| -> Vec<f64> {
        let mut out = vec![];
        for col in (stride / 2..cols).step_by(stride) {
            for row in (stride / 2..rows).step_by(stride) {
                out.push(grid[(col, row, 0)].hypot(grid[(col, row, 1)]));
            }
        }
        out
    };
    let c2 = app.settings.speed_of_light.powi(2);
    let highs = (app.settings.e_field_max, app.settings.b_field_max, app.settings.s_field_max);
    let b_magnitudes: Vec<f64> = b.iter().map(|v| v.abs() * c2).collect();
    for (id, magnitudes, high) in [
        (LayerId::E, sampled(&e), highs.0),
        (LayerId::B, b_magnitudes, highs.1),
        (LayerId::S, sampled(&s_grid), highs.2),
    ] {
        let layer = &mut app.layers[id];
        if layer.auto && layer.visible {
            layer.follow(&magnitudes, high);
        }
    }
    let electrons: Vec<[f32; 4]> = app
        .world()
        .particles
        .pos
        .iter()
        .map(|p| [p[0] as f32, p[1] as f32, 0.0, 0.0])
        .collect();
    let grow = |buffer: &mut wgpu::Buffer, bytes: &[u8], label: &str, generation: &mut u64| {
        if bytes.len() as u64 > buffer.size() {
            *buffer = gpu.storage_buffer(label, bytes.len() as u64 * 2, wgpu::BufferUsages::empty());
            *generation += 1;
        }
        if !bytes.is_empty() {
            gpu.queue.write_buffer(buffer, 0, bytes);
        }
    };
    grow(&mut c.fields, bytemuck::cast_slice(&display), "display", &mut c.generation);
    grow(&mut c.electrons, bytemuck::cast_slice(&electrons), "electrons", &mut c.generation);
}

/// Event helpers shared with lib.rs.
pub fn button_of(event: &web_sys::PointerEvent) -> Button {
    if event.button() == 2 { Button::Erase } else { Button::Paint }
}

pub fn target_element(event: &web_sys::Event) -> Option<web_sys::Element> {
    event.target()?.dyn_into::<web_sys::Element>().ok()
}

fn offscreen_texture(gpu: &Gpu, config: &wgpu::SurfaceConfiguration) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen"),
        size: wgpu::Extent3d { width: config.width, height: config.height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

impl WebApp {
    pub fn request_snapshot(&mut self) {
        if self.offscreen.is_some() && self.snapshot.status() == Status::Idle {
            self.last_snapshot = None;
            self.snapshot.status.set(Status::Requested);
        }
    }

    pub fn snapshot(&self) -> Option<&[u8]> {
        self.last_snapshot.as_deref()
    }
}
