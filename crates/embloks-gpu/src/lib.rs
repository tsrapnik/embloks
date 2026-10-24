//! Embloks on the GPU (WebGPU via wgpu).
//!
//! - [`sim::GpuSim`]: the physics of embloks-core's `Simulation`, in compute shaders
//!   (single precision). The CPU keeps the authoritative `World` for painting; the
//!   GPU owns the moving state (electrons, fields, battery gains) between edits.
//! - [`render::Renderer`]: draws a scene from GPU buffers, whichever backend filled them.

pub mod render;
pub mod sim;
mod layout;

/// A device and queue, shared by the simulation and the renderer.
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
}

impl Gpu {
    /// Request a device from an adapter, with the limits the simulation needs.
    pub async fn from_adapter(adapter: &wgpu::Adapter) -> Result<Self, String> {
        let supported = adapter.limits();
        let limits = wgpu::Limits {
            // Large grids and many electrons need big buffers; never ask for more than
            // the adapter has.
            max_storage_buffer_binding_size: supported.max_storage_buffer_binding_size,
            max_buffer_size: supported.max_buffer_size,
            ..wgpu::Limits::default()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("embloks"),
                required_limits: limits,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no WebGPU device: {e}"))?;
        Ok(Self { device, queue, adapter_info: adapter.get_info() })
    }

    /// A device without a window (native tests and tools).
    pub async fn headless() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .map_err(|e| format!("no GPU adapter: {e}"))?;
        Self::from_adapter(&adapter).await
    }

    pub fn storage_buffer(&self, label: &str, size: u64, extra: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            // wgpu rejects zero-sized bindings; keep at least one element.
            size: size.max(16).next_multiple_of(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | extra,
            mapped_at_creation: false,
        })
    }

    pub fn uniform_buffer(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.next_multiple_of(16),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Copy `size` bytes of `buffer` (from `offset`) back to the CPU, blocking until
    /// done. Native only: for tests and tools.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_blocking(&self, buffer: &wgpu::Buffer, offset: u64, size: u64) -> Vec<u8> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read back"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, offset, &staging, 0, size);
        self.queue.submit([encoder.finish()]);
        staging.map_async(wgpu::MapMode::Read, .., |result| result.expect("map failed"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll failed");
        let data = staging.get_mapped_range(..).expect("mapped").to_vec();
        staging.unmap();
        data
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_f32(&self, buffer: &wgpu::Buffer, offset_elems: u64, count: u64) -> Vec<f32> {
        if count == 0 {
            return vec![];
        }
        bytemuck::cast_slice(&self.read_blocking(buffer, offset_elems * 4, count * 4)).to_vec()
    }
}
