// Shared declarations of the simulation's compute shaders. `Params` is generated
// from the Rust side (layout.rs) and prepended to this file.
//
// Eight storage buffers (the WebGPU guaranteed maximum per stage), each holding
// several arrays at offsets given in Params:
//   F      f32   fields (Yee grid, sources, Poisson work, potentials, currents, display)
//   parts  vec4  electrons at the start of a substep: (x, y, vx, vy) in cell units
//   parts_out    electrons after it (copied back once every electron has moved)
//   C      u32   cell list: counts, starts, sorted order, each electron's cell and rank
//   info   per-cell material properties
//   T      f32   static tables: close-range kernels, sine matrices, inverse Laplacian
//   S      u32   batteries and per-frame data (floats stored as bits)
//   state  u32   counters for the current tick and substep, histograms

struct CellInfo {
    damping: f32,
    emf: f32,
    frequency: f32,
    density: f32,
    // bit 0 conductive, bit 1 wire (conductor), bits 2-3 battery orientation
    flags: u32,
    // battery stack index + 1, or 0
    stack: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> F: array<f32>;
@group(0) @binding(2) var<storage, read_write> parts: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> parts_out: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> C: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> info: array<CellInfo>;
@group(0) @binding(6) var<storage, read> T: array<f32>;
@group(0) @binding(7) var<storage, read_write> S: array<u32>;
@group(0) @binding(8) var<storage, read_write> state: array<atomic<u32>>;

// state indices
const ST_TICK: u32 = 0u;
const ST_SUBSTEP: u32 = 1u;
const ST_SUBSTEPS: u32 = 2u;
const ST_H: u32 = 3u;
const ST_MAX_SPEED: u32 = 4u;
const ST_RESET_DISPLAY: u32 = 5u;
const ST_HIST: u32 = 8u;
const HIST_BINS: u32 = 256u;

const PI: f32 = 3.14159265358979;

// Threads are numbered across a 2D dispatch, since one dimension holds at most
// 65535 workgroups.
fn thread_index(g: vec3<u32>) -> u32 {
    return g.x + g.y * 65535u * 64u;
}

fn sf(i: u32) -> f32 {
    return bitcast<f32>(S[i]);
}

fn cell_of(pos: vec2<f32>) -> vec2<i32> {
    return clamp(vec2<i32>(floor(pos)), vec2<i32>(0, 0), vec2<i32>(i32(P.cols) - 1, i32(P.rows) - 1));
}

fn flat(c: vec2<i32>) -> u32 {
    return u32(c.x) * P.rows + u32(c.y);
}

fn cell_start(cell: u32) -> u32 {
    return atomicLoad(&C[P.o_start + cell]);
}

fn sorted(slot: u32) -> u32 {
    return atomicLoad(&C[P.o_order + slot]);
}

// Bilinear interpolation in a (w, h) array of F starting at `o`, at fractional
// indices (gx, gy), clamped like the CPU version (maxwell::bilinear).
fn bilinear(o: u32, w: u32, h: u32, gx: f32, gy: f32) -> f32 {
    let i = clamp(i32(floor(gx)), 0, i32(w) - 2);
    let j = clamp(i32(floor(gy)), 0, i32(h) - 2);
    let fx = gx - f32(i);
    let fy = gy - f32(j);
    let b = o + u32(i) * h + u32(j);
    return F[b] * (1.0 - fx) * (1.0 - fy) + F[b + h] * fx * (1.0 - fy)
         + F[b + 1u] * (1.0 - fx) * fy + F[b + h + 1u] * fx * fy;
}

// Channel k of the square table (close_range.rs) at fractional indices (u, v).
// Channel 0 is the potential, 1 and 2 the field.
fn square(k: u32, u: f32, v: f32) -> f32 {
    let n = P.square_n;
    let a = clamp(i32(floor(u)), 0, i32(n) - 2);
    let b = clamp(i32(floor(v)), 0, i32(n) - 2);
    let fu = u - f32(a);
    let fv = v - f32(b);
    let i00 = P.o_square + (u32(a) * n + u32(b)) * 3u + k;
    let i10 = i00 + n * 3u;
    return T[i00] * (1.0 - fu) * (1.0 - fv) + T[i10] * fu * (1.0 - fv)
         + T[i00 + 3u] * (1.0 - fu) * fv + T[i10 + 3u] * fu * fv;
}

// Close-range potential at cell (c, r) of the electrons near it, averaged over the
// cell (close_range::charges_potential).
fn charges_potential(c: i32, r: i32) -> f32 {
    let reach = i32(P.reach);
    var total = 0.0;
    for (var cc = max(c - reach, 0); cc <= min(c + reach, i32(P.cols) - 1); cc++) {
        for (var rr = max(r - reach, 0); rr <= min(r + reach, i32(P.rows) - 1); rr++) {
            let cell = flat(vec2<i32>(cc, rr));
            for (var slot = cell_start(cell); slot < cell_start(cell + 1u); slot++) {
                let p = parts[sorted(slot)].xy;
                let d = p - (vec2<f32>(f32(c), f32(r)) + 0.5);
                let uv = (d + P.half) * P.square_samples;
                total += square(0u, uv.x, uv.y);
            }
        }
    }
    return P.k * P.charge * total;
}

// Electric potential at a visible cell, close-range part included (Simulation::potential).
fn potential_at(c: i32, r: i32) -> f32 {
    let node = (u32(c) + P.margin) * P.ny + u32(r) + P.margin;
    return F[P.o_phi + node] + F[P.o_ion_potential + flat(vec2<i32>(c, r))] + charges_potential(c, r);
}
