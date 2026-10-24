//! Drawing a scene with WebGPU, from the same buffers the simulation steps (or,
//! for the CPU backend, buffers filled from it each frame).

use embloks_core::grid::Grid;
use embloks_core::layers::{LayerId, Layers};
use embloks_core::materials::{Cell, Material};
use embloks_core::settings::{Rgb, Settings};
use embloks_core::world::World;

use crate::Gpu;
use crate::layout::gpu_struct;

gpu_struct!(View {
    width: f32, height: f32, origin_x: f32, origin_y: f32, cell_px: f32,
    cols: u32, rows: u32, cells: u32, o_display: u32,
    background: u32, line_color: u32, min_line_px: f32, min_glyph_px: f32,
    show_ions: u32, ion_color: u32, plus_color: u32, minus_color: u32,
    b_visible: u32, b_gain: f32, b_log: u32, b_low: f32, b_high: f32, b_opacity: f32, b_max_alpha: f32,
    b_in_color: u32, b_out_color: u32, c2: f32,
    arrow_count: u32, arrow_stride: u32, line_width: f32, inv_mu0: f32,
    e_visible: u32, e_gain: f32, e_log: u32, e_low: f32, e_high: f32, e_opacity: f32, e_dim: u32, e_bright: u32,
    s_visible: u32, s_gain: f32, s_log: u32, s_low: f32, s_high: f32, s_opacity: f32, s_dim: u32, s_bright: u32,
    electron_radius: f32, electron_color: u32, dot_softness: f32,
    negative_color: u32, positive_color: u32, hover_color: u32,
});

fn pack(rgb: Rgb) -> u32 {
    rgb[0] as u32 | (rgb[1] as u32) << 8 | (rgb[2] as u32) << 16
}

/// Draw an arrow on every n-th cell, so arrows stay at least min_arrow_spacing_px apart.
pub fn arrow_stride(grid: &Grid, settings: &Settings) -> u32 {
    (settings.min_arrow_spacing_px / grid.cell_px as f64).ceil().max(1.0) as u32
}

/// Colour and flags per cell (render.MaterialLayer._build): rgb in the low 24 bits,
/// bit 24 conductive, 25 battery, 26-27 the battery's orientation.
pub fn cell_styles(world: &World, settings: &Settings) -> Vec<u32> {
    let m = &world.materials;
    let r = &settings.resistance_values;
    let (low, high) = (settings.conductor_color_low_r, settings.conductor_color_high_r);
    m.kind
        .indexed_iter()
        .map(|(cell, &kind)| match Material::from_u8(kind) {
            Material::Vacuum => pack(settings.background_color),
            Material::Conductor => {
                // Position of the resistance between the ladder's ends, on a log scale.
                let t = ((m.resistance[cell] / r[0]).ln() / (r[r.len() - 1] / r[0]).ln()).clamp(0.0, 1.0);
                let shade = |k: usize| (low[k] as f64 + (high[k] as f64 - low[k] as f64) * t) as u8;
                pack([shade(0), shade(1), shade(2)]) | 1 << 24
            }
            Material::Battery => pack(settings.battery_color) | 1 << 24 | 1 << 25 | (m.orientation[cell] as u32 & 3) << 26,
        })
        .collect()
}

/// Where the renderer reads the fields and electrons from.
pub struct Sources<'a> {
    pub fields: &'a wgpu::Buffer,
    /// Offset (in f32s) of the display average: Ex, Ey, Bz per cell, one after another.
    pub o_display: u32,
    pub electrons: &'a wgpu::Buffer,
    pub electron_count: u32,
    /// Changes whenever the buffers above are replaced.
    pub generation: u64,
}

/// Everything a frame shows besides the sources.
pub struct Frame<'a> {
    pub grid: Grid,
    pub settings: &'a Settings,
    pub layers: &'a Layers,
    pub brush: &'a [Cell],
    /// Device pixels per CSS pixel: lines are about one CSS pixel wide.
    pub pixel_ratio: f32,
}

pub struct Renderer {
    uniform: wgpu::Buffer,
    styles: wgpu::Buffer,
    charges: wgpu::Buffer,
    brush: wgpu::Buffer,
    charge_count: u32,
    layout: wgpu::BindGroupLayout,
    bind_group: Option<(u64, u64, wgpu::BindGroup)>,
    buffers_version: u64,
    world_key: Option<(u64, u64, usize, usize)>,
    cells_pipeline: wgpu::RenderPipeline,
    arrows_pipeline: wgpu::RenderPipeline,
    electrons_pipeline: wgpu::RenderPipeline,
    charges_pipeline: wgpu::RenderPipeline,
    brush_pipeline: wgpu::RenderPipeline,
}

impl Renderer {
    pub fn new(gpu: &Gpu, format: wgpu::TextureFormat) -> Self {
        let source = [View::wgsl(), include_str!("shaders/render.wgsl").to_string()].concat();
        let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("render"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        }];
        entries.extend((1..=5).map(storage));
        let layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("render"), entries: &entries });
        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |vertex: &str, fragment: &str, blend: Option<wgpu::BlendState>| {
            gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(vertex),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(vertex),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(fragment),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format, blend, write_mask: wgpu::ColorWrites::ALL })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let blend = Some(wgpu::BlendState::ALPHA_BLENDING);
        Self {
            uniform: gpu.uniform_buffer("view", size_of::<View>() as u64),
            styles: gpu.storage_buffer("cell styles", 16, wgpu::BufferUsages::empty()),
            charges: gpu.storage_buffer("test charges", 16, wgpu::BufferUsages::empty()),
            brush: gpu.storage_buffer("brush", 16, wgpu::BufferUsages::empty()),
            charge_count: 0,
            layout,
            bind_group: None,
            buffers_version: 0,
            world_key: None,
            cells_pipeline: pipeline("fullscreen", "cells", None),
            arrows_pipeline: pipeline("arrow", "flat_color", blend),
            electrons_pipeline: pipeline("electron", "round_dot", blend),
            charges_pipeline: pipeline("charge", "charge_disc", blend),
            brush_pipeline: pipeline("brush_cell", "outline", blend),
        }
    }

    /// Write `data` to `buffer`, growing it (a new buffer) if needed.
    fn fill(gpu: &Gpu, buffer: &mut wgpu::Buffer, data: &[u8], label: &str, version: &mut u64) {
        if data.len() as u64 > buffer.size() {
            *buffer = gpu.storage_buffer(label, (data.len() as u64 * 3 / 2).next_multiple_of(16), wgpu::BufferUsages::empty());
            *version += 1;
        }
        if !data.is_empty() {
            gpu.queue.write_buffer(buffer, 0, data);
        }
    }

    /// Refresh the materials and test charges, if they changed.
    pub fn update_world(&mut self, gpu: &Gpu, world: &World, settings: &Settings) {
        let m = &world.materials;
        let key = (m.version, world.fixed_version, m.cols, m.rows);
        if self.world_key == Some(key) {
            return;
        }
        self.world_key = Some(key);
        let styles = cell_styles(world, settings);
        Self::fill(gpu, &mut self.styles, bytemuck::cast_slice(&styles), "cell styles", &mut self.buffers_version);
        let charges: Vec<[f32; 4]> = world
            .fixed_pos
            .iter()
            .zip(&world.fixed_charge)
            .map(|(p, &q)| [p[0] as f32, p[1] as f32, q as f32, 0.0])
            .collect();
        self.charge_count = charges.len() as u32;
        Self::fill(gpu, &mut self.charges, bytemuck::cast_slice(&charges), "test charges", &mut self.buffers_version);
    }

    pub fn render(&mut self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView, size: (u32, u32), sources: &Sources, frame: &Frame) {
        let (s, layers, grid) = (frame.settings, frame.layers, &frame.grid);
        let brush: Vec<u32> = frame.brush.iter().map(|&(c, r)| (c * grid.rows + r) as u32).collect();
        Self::fill(gpu, &mut self.brush, bytemuck::cast_slice(&brush), "brush", &mut self.buffers_version);
        let key = (sources.generation, self.buffers_version);
        if self.bind_group.as_ref().is_none_or(|(g, v, _)| (*g, *v) != key) {
            let buffers = [&self.uniform, sources.fields, sources.electrons, &self.styles, &self.charges, &self.brush];
            let entries: Vec<wgpu::BindGroupEntry> = buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
                .collect();
            let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("render"), layout: &self.layout, entries: &entries });
            self.bind_group = Some((key.0, key.1, group));
        }

        let stride = arrow_stride(grid, s);
        let along = |n: usize| (n - (stride as usize / 2)).div_ceil(stride as usize) as u32;
        let arrow_count = along(grid.cols) * along(grid.rows);
        let (e, b, sl) = (&layers[LayerId::E], &layers[LayerId::B], &layers[LayerId::S]);
        let c2 = s.speed_of_light.powi(2);
        let view = View {
            width: size.0 as f32,
            height: size.1 as f32,
            origin_x: grid.origin_x as f32,
            origin_y: grid.origin_y as f32,
            cell_px: grid.cell_px as f32,
            cols: grid.cols as u32,
            rows: grid.rows as u32,
            cells: (grid.cols * grid.rows) as u32,
            o_display: sources.o_display,
            background: pack(s.background_color),
            line_color: pack(s.grid_line_color),
            min_line_px: s.min_cell_px_for_lines as f32,
            min_glyph_px: s.min_cell_px_for_glyphs as f32,
            show_ions: layers[LayerId::P].visible as u32,
            ion_color: pack(s.ion_color),
            plus_color: pack(s.battery_plus_color),
            minus_color: pack(s.battery_minus_color),
            b_visible: b.visible as u32,
            b_gain: b.gain as f32,
            b_log: b.log_scale as u32,
            b_low: s.b_field_min as f32,
            b_high: s.b_field_max as f32,
            b_opacity: b.opacity as f32,
            b_max_alpha: (s.b_max_alpha / 255.0) as f32,
            b_in_color: pack(s.b_in_color),
            b_out_color: pack(s.b_out_color),
            c2: c2 as f32,
            arrow_count,
            arrow_stride: stride,
            line_width: frame.pixel_ratio.max(1.0) * 1.2,
            inv_mu0: (c2 / (4.0 * std::f64::consts::PI * s.coulomb_k)) as f32,
            e_visible: e.visible as u32,
            e_gain: e.gain as f32,
            e_log: e.log_scale as u32,
            e_low: s.e_field_min as f32,
            e_high: s.e_field_max as f32,
            e_opacity: e.opacity as f32,
            e_dim: pack(s.e_arrow_dim_color),
            e_bright: pack(s.e_arrow_color),
            s_visible: sl.visible as u32,
            s_gain: sl.gain as f32,
            s_log: sl.log_scale as u32,
            s_low: s.s_field_min as f32,
            s_high: s.s_field_max as f32,
            s_opacity: sl.opacity as f32,
            s_dim: pack(s.s_arrow_dim_color),
            s_bright: pack(s.s_arrow_color),
            electron_radius: s.electron_radius as f32,
            electron_color: pack(s.electron_color),
            dot_softness: (1.0 / (grid.cell_px as f64 * s.electron_radius).max(1.0)) as f32,
            negative_color: pack(s.negative_charge_color),
            positive_color: pack(s.positive_charge_color),
            hover_color: pack(s.hover_color),
            ..View::default()
        };
        gpu.queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&view));

        let bg = s.background_color;
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("scene"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: bg[0] as f64 / 255.0,
                        g: bg[1] as f64 / 255.0,
                        b: bg[2] as f64 / 255.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_bind_group(0, &self.bind_group.as_ref().unwrap().2, &[]);
        pass.set_pipeline(&self.cells_pipeline);
        pass.draw(0..3, 0..1);
        if e.visible || sl.visible {
            pass.set_pipeline(&self.arrows_pipeline);
            pass.draw(0..18, 0..2 * arrow_count);
        }
        if layers[LayerId::P].visible && sources.electron_count > 0 {
            pass.set_pipeline(&self.electrons_pipeline);
            pass.draw(0..6, 0..sources.electron_count);
        }
        if self.charge_count > 0 {
            pass.set_pipeline(&self.charges_pipeline);
            pass.draw(0..6, 0..self.charge_count);
        }
        if !brush.is_empty() {
            pass.set_pipeline(&self.brush_pipeline);
            pass.draw(0..6, 0..brush.len() as u32);
        }
    }
}
