//! Render example scenes offscreen; the images land in target/renders for a look.

use embloks_core::controller::App;
use embloks_core::layers::LayerId;
use embloks_core::settings::Settings;
use embloks_gpu::Gpu;
use embloks_gpu::render::{Frame, Renderer};
use embloks_gpu::sim::{GpuSim, Probe};

// Width a multiple of 64 pixels: texture copies need rows of 256-byte multiples.
const SIZE: (u32, u32) = (1024, 800);

fn render(gpu: &Gpu, app: &App, sim: &GpuSim) -> Vec<u8> {
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: SIZE.0, height: SIZE.1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let mut renderer = Renderer::new(gpu, format);
    renderer.update_world(gpu, app.world(), &app.settings);
    let brush = app.brush.footprint((20, 3), app.grid.cols, app.grid.rows);
    let frame = Frame { grid: app.grid, settings: &app.settings, layers: &app.layers, brush: &brush, pixel_ratio: 1.0 };
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    renderer.render(gpu, &mut encoder, &view, SIZE, &sim.sources(), &frame);
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (SIZE.0 * SIZE.1 * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(SIZE.0 * 4), rows_per_image: None },
        },
        wgpu::Extent3d { width: SIZE.0, height: SIZE.1, depth_or_array_layers: 1 },
    );
    gpu.queue.submit([encoder.finish()]);
    readback.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    readback.get_mapped_range(..).unwrap().to_vec()
}

fn save(name: &str, pixels: &[u8]) {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/renders");
    std::fs::create_dir_all(&dir).unwrap();
    let file = std::fs::File::create(dir.join(format!("{name}.png"))).unwrap();
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), SIZE.0, SIZE.1);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header().unwrap().write_image_data(pixels).unwrap();
}

#[test]
fn render_example_scenes() {
    let Ok(gpu) = pollster::block_on(Gpu::headless()) else { return };
    for (index, name, seconds) in [(0, "battery_and_resistor", 30.0), (1, "screening", 5.0), (6, "antenna", 20.0)] {
        let mut app = App::new(Settings::default(), SIZE.0 as i32, SIZE.1 as i32);
        app.load_scene(index);
        let mut sim = GpuSim::new(&gpu, app.world());
        sim.run_ticks(&gpu, (seconds / app.settings.dt) as usize);
        // Auto gain, as the app does every frame: towards the 99th percentile.
        let stride = embloks_gpu::render::arrow_stride(&app.grid, &app.settings);
        let probe = sim.read_probe(&gpu, None, stride);
        let s = app.settings.clone();
        for (id, histogram, high) in [(LayerId::E, 0, s.e_field_max), (LayerId::B, 1, s.b_field_max), (LayerId::S, 2, s.s_field_max)] {
            let layer = &mut app.layers[id];
            if let (true, Some(p99)) = (layer.auto, Probe::percentile(&probe.histograms[histogram], 99.0)) {
                for _ in 0..60 {
                    layer.ease_towards(high / p99);
                }
            }
        }
        let pixels = render(&gpu, &app, &sim);
        // Not blank: the scene's materials and fields cover plenty of pixels.
        let bg = app.settings.background_color;
        let painted = pixels.chunks(4).filter(|p| p[..3] != bg[..]).count();
        assert!(painted > 10_000, "{name}: only {painted} pixels drawn");
        save(name, &pixels);
    }
}

#[test]
fn a_new_simulation_never_reuses_the_last_ones_buffers() {
    let Ok(gpu) = pollster::block_on(Gpu::headless()) else { return };
    let mut app = App::new(Settings::default(), SIZE.0 as i32, SIZE.1 as i32);
    app.load_scene(0);
    let first = GpuSim::new(&gpu, app.world());
    app.load_scene(6);
    let second = GpuSim::new(&gpu, app.world());
    assert_ne!(first.sources().generation, second.sources().generation);
}
