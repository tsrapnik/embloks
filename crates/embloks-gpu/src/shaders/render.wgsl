// Drawing a scene (render.py in the Python original), straight from the simulation's
// buffers. `View` is generated on the Rust side and prepended. Pixels are canvas
// device pixels, y down.
//
//   cells      one full-screen pass: background, materials, ion rings, battery marks,
//              the B heatmap and grid lines, composited per pixel in that order
//   arrows     E or S arrows, one instance per sampled cell
//   electrons  one quad per electron
//   charges    test charges
//   brush      outline of the cells under the brush

@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var<storage, read> fields: array<f32>;
@group(0) @binding(2) var<storage, read> electrons: array<vec4<f32>>;
// Per cell: rgb in the low 24 bits; bit 24 conductive, 25 battery, 26-27 orientation.
@group(0) @binding(3) var<storage, read> styles: array<u32>;
// Per test charge: x, y, charge, unused.
@group(0) @binding(4) var<storage, read> charges: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> brush: array<u32>;

fn unpack_rgb(v: u32) -> vec3<f32> {
    return vec3(f32(v & 255u), f32((v >> 8u) & 255u), f32((v >> 16u) & 255u)) / 255.0;
}

fn to_clip(px: vec2<f32>) -> vec4<f32> {
    return vec4(px.x / V.width * 2.0 - 1.0, 1.0 - px.y / V.height * 2.0, 0.0, 1.0);
}

fn origin() -> vec2<f32> {
    return vec2(V.origin_x, V.origin_y);
}

// Display strength 0..1 (Layer::strength): log or linear between low and high.
fn strength(magnitude: f32, gain: f32, log_scale: u32, low: f32, high: f32) -> f32 {
    let scaled = magnitude * gain;
    if (scaled <= 0.0) {
        return 0.0;
    }
    if (log_scale != 0u) {
        return clamp(log(scaled / low) / log(high / low), 0.0, 1.0);
    }
    return clamp(scaled / high, 0.0, 1.0);
}

fn display_e(cell: u32) -> vec2<f32> {
    return vec2(fields[V.o_display + cell], fields[V.o_display + V.cells + cell]);
}

fn display_b(cell: u32) -> f32 {
    return fields[V.o_display + 2u * V.cells + cell];
}

// --- Cells --------------------------------------------------------------------------

@vertex
fn fullscreen(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}

// Inside a w x h rectangle with corner `corner`?
fn in_rect(p: vec2<f32>, corner: vec2<f32>, size: vec2<f32>) -> bool {
    return all(p >= corner) && all(p < corner + size);
}

const DIRECTIONS = array<vec2<f32>, 4>(vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(-1.0, 0.0), vec2(0.0, -1.0));

// The battery's stripes (+ red, - blue) along its cell edges and, on large cells,
// its + and - glyphs (render._draw_battery_marks).
fn battery_mark(local: vec2<f32>, size: f32, orientation: u32) -> vec4<f32> {
    let dir = DIRECTIONS[orientation];
    let stripe = max(1.0, floor(size / 5.0));
    let centre = vec2(size * 0.5);
    for (var k = 0; k < 2; k++) {
        let side = select(-1.0, 1.0, k == 0);
        let color = select(unpack_rgb(V.minus_color), unpack_rgb(V.plus_color), k == 0);
        let e = side * dir;
        var corner = vec2(0.0);
        var extent = vec2(size, size);
        if (e.x != 0.0) {
            corner.x = select(0.0, size - stripe, e.x > 0.0);
            extent.x = stripe;
        } else {
            corner.y = select(0.0, size - stripe, e.y > 0.0);
            extent.y = stripe;
        }
        if (in_rect(local, corner, extent)) {
            return vec4(color, 1.0);
        }
        if (size >= V.min_glyph_px) {
            let g = centre + e * size / 4.0;
            let half = size / 8.0;
            let width = max(1.0, floor(size / 12.0));
            let d = abs(local - g);
            let across = d.x <= half && d.y <= width * 0.5;
            let along = d.y <= half && d.x <= width * 0.5;
            if (across || (k == 0 && along)) {
                return vec4(color, 1.0);
            }
        }
    }
    return vec4(0.0);
}

// The B heatmap's colour and opacity at grid coordinates (cells), interpolated
// between cell centres like the smooth scaling of the original.
fn b_overlay(grid: vec2<f32>) -> vec4<f32> {
    let u = clamp(grid - 0.5, vec2(0.0), vec2(f32(V.cols) - 1.0, f32(V.rows) - 1.0));
    let a = vec2<u32>(floor(u));
    let b = min(a + 1u, vec2(V.cols - 1u, V.rows - 1u));
    let f = u - vec2<f32>(a);
    var acc = vec4(0.0);
    for (var k = 0u; k < 4u; k++) {
        let x = select(a.x, b.x, (k & 1u) != 0u);
        let y = select(a.y, b.y, (k & 2u) != 0u);
        let w = select(1.0 - f.x, f.x, (k & 1u) != 0u) * select(1.0 - f.y, f.y, (k & 2u) != 0u);
        let bz = display_b(x * V.rows + y);
        let t = strength(abs(bz) * V.c2, V.b_gain, V.b_log, V.b_low, V.b_high);
        let color = select(unpack_rgb(V.b_out_color), unpack_rgb(V.b_in_color), bz > 0.0);
        acc += w * vec4(color, t * V.b_max_alpha * V.b_opacity);
    }
    return acc;
}

@fragment
fn cells(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let p = frag.xy - origin();
    let size = V.cell_px;
    let extent = vec2(f32(V.cols), f32(V.rows)) * size;
    if (any(p < vec2(0.0)) || any(p >= extent)) {
        return vec4(unpack_rgb(V.background), 1.0);
    }
    let cell_xy = vec2<u32>(floor(p / size));
    let cell = cell_xy.x * V.rows + cell_xy.y;
    let style = styles[cell];
    var color = unpack_rgb(style);
    let local = p - vec2<f32>(cell_xy) * size;
    // Fixed ion: a ring at the centre of every conductive cell large enough to show it.
    if ((style & (1u << 24u)) != 0u && V.show_ions != 0u && size >= V.min_glyph_px) {
        let radius = max(2.0, round(size * 0.12));
        let width = max(1.0, floor(size / 16.0));
        let d = length(local - vec2(floor(size * 0.5)));
        if (d <= radius && d >= radius - width) {
            color = unpack_rgb(V.ion_color);
        }
    }
    if ((style & (1u << 25u)) != 0u) {
        let mark = battery_mark(local, size, (style >> 26u) & 3u);
        color = mix(color, mark.rgb, mark.a);
    }
    if (V.b_visible != 0u) {
        let b = b_overlay(p / size);
        if (b.a > 0.0) {
            color = mix(color, b.rgb, b.a);
        }
    }
    if (size >= V.min_line_px && (local.x < 1.0 || local.y < 1.0)) {
        color = unpack_rgb(V.line_color);
    }
    return vec4(color, 1.0);
}

// --- Arrows -------------------------------------------------------------------------

struct Colored {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) local: vec2<f32>,
}

// A quad along the segment a -> b, `width` pixels wide; corner k of 6.
fn segment_vertex(a: vec2<f32>, b: vec2<f32>, width: f32, k: u32) -> vec2<f32> {
    let d = b - a;
    let len = max(length(d), 1e-6);
    let n = vec2(-d.y, d.x) / len * width * 0.5;
    let corners = array<vec2<f32>, 6>(vec2(0.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
                                      vec2(0.0, -1.0), vec2(1.0, 1.0), vec2(0.0, 1.0));
    let c = corners[k];
    return a + d * c.x + n * c.y;
}

// Arrows centred on sampled cells; length and brightness follow the layer's strength
// (render.draw_field_arrows). Instances [0, arrow_count) are E arrows, the next
// arrow_count S arrows; each is three segments of six vertices.
struct ArrowLayer {
    gain: f32,
    log_scale: u32,
    low: f32,
    high: f32,
    opacity: f32,
    dim: vec3<f32>,
    bright: vec3<f32>,
}

fn arrow_layer(layer: u32) -> ArrowLayer {
    if (layer == 0u) {
        return ArrowLayer(V.e_gain, V.e_log, V.e_low, V.e_high, V.e_opacity, unpack_rgb(V.e_dim), unpack_rgb(V.e_bright));
    }
    return ArrowLayer(V.s_gain, V.s_log, V.s_low, V.s_high, V.s_opacity, unpack_rgb(V.s_dim), unpack_rgb(V.s_bright));
}

@vertex
fn arrow(@builtin(vertex_index) v: u32, @builtin(instance_index) instance: u32) -> Colored {
    var out: Colored;
    out.position = vec4(2.0, 2.0, 0.0, 1.0); // Off screen unless drawn.
    let index = instance % V.arrow_count;
    let layer_index = instance / V.arrow_count;
    if ((layer_index == 0u && V.e_visible == 0u) || (layer_index == 1u && V.s_visible == 0u)) {
        return out;
    }
    let layer = arrow_layer(layer_index);
    let stride = V.arrow_stride;
    let per_column = (V.rows - stride / 2u + stride - 1u) / stride;
    let col = stride / 2u + (index / per_column) * stride;
    let row = stride / 2u + (index % per_column) * stride;
    if (col >= V.cols || row >= V.rows) {
        return out;
    }
    let cell = col * V.rows + row;
    var field = display_e(cell);
    if (layer_index == 1u) {
        let b = display_b(cell);
        field = V.inv_mu0 * vec2(field.y * b, -field.x * b);
    }
    let magnitude = length(field);
    let spacing = V.cell_px * f32(stride);
    let length_px = spacing * 0.9 * strength(magnitude, layer.gain, layer.log_scale, layer.low, layer.high);
    if (length_px < 2.0) {
        return out;
    }
    let u = field / magnitude;
    let centre = origin() + (vec2(f32(col), f32(row)) + 0.5) * V.cell_px;
    let tip = centre + u * length_px * 0.5;
    let tail = centre - u * length_px * 0.5;
    let head = length_px * 0.3;
    let barb1 = tip - head * vec2(0.866 * u.x - 0.5 * u.y, 0.866 * u.y + 0.5 * u.x);
    let barb2 = tip - head * vec2(0.866 * u.x + 0.5 * u.y, 0.866 * u.y - 0.5 * u.x);
    let segment = v / 6u;
    var a = tail;
    if (segment == 1u) {
        a = barb1;
    } else if (segment == 2u) {
        a = barb2;
    }
    out.position = to_clip(segment_vertex(a, tip, V.line_width, v % 6u));
    let t = length_px / (spacing * 0.9);
    out.color = vec4(mix(layer.dim, layer.bright, t), layer.opacity);
    return out;
}

@fragment
fn flat_color(in: Colored) -> @location(0) vec4<f32> {
    return in.color;
}

// --- Electrons and test charges -------------------------------------------------------

const QUAD = array<vec2<f32>, 6>(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
                                 vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));

@vertex
fn electron(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Colored {
    var out: Colored;
    let centre = origin() + electrons[i].xy * V.cell_px;
    let radius = V.cell_px * V.electron_radius;
    let corner = QUAD[v];
    if (radius < 1.5) {
        // Tiny cells: a single pixel per electron.
        out.position = to_clip(floor(centre) + (corner + 1.0) * 0.5);
        out.local = vec2(0.0);
    } else {
        out.position = to_clip(centre + corner * (radius + 1.0));
        out.local = corner * (radius + 1.0) / radius;
    }
    out.color = vec4(unpack_rgb(V.electron_color), 1.0);
    return out;
}

@fragment
fn round_dot(in: Colored) -> @location(0) vec4<f32> {
    // Antialiased edge, about a pixel wide.
    let d = length(in.local);
    let edge = 1.0 - smoothstep(1.0 - V.dot_softness, 1.0, d);
    if (edge <= 0.0) {
        discard;
    }
    return vec4(in.color.rgb, in.color.a * edge);
}

@vertex
fn charge(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Colored {
    var out: Colored;
    let c = charges[i];
    let centre = origin() + c.xy * V.cell_px;
    let radius = max(3.0, V.cell_px * 0.35);
    out.position = to_clip(centre + QUAD[v] * radius);
    out.local = QUAD[v];
    out.color = vec4(select(unpack_rgb(V.negative_color), unpack_rgb(V.positive_color), c.z > 0.0), select(0.0, 1.0, c.z > 0.0));
    return out;
}

@fragment
fn charge_disc(in: Colored) -> @location(0) vec4<f32> {
    let radius = max(3.0, V.cell_px * 0.35);
    let d = length(in.local);
    if (d > 1.0) {
        discard;
    }
    // A white - (and | for positive charges) on discs big enough for it.
    if (radius >= 6.0) {
        let half = 0.55;
        let width = max(1.0, round(radius / 5.0)) / radius * 0.5;
        let a = abs(in.local);
        if ((a.x <= half && a.y <= width) || (in.color.a > 0.5 && a.y <= half && a.x <= width)) {
            return vec4(1.0);
        }
    }
    return vec4(in.color.rgb, 1.0);
}

// --- Brush outline ----------------------------------------------------------------

@vertex
fn brush_cell(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Colored {
    var out: Colored;
    let cell = brush[i];
    let corner = origin() + vec2(f32(cell / V.rows), f32(cell % V.rows)) * V.cell_px;
    let unit = (QUAD[v] + 1.0) * 0.5;
    out.position = to_clip(corner + unit * V.cell_px);
    out.local = unit * V.cell_px;
    out.color = vec4(unpack_rgb(V.hover_color), 1.0);
    return out;
}

@fragment
fn outline(in: Colored) -> @location(0) vec4<f32> {
    let d = min(in.local, vec2(V.cell_px) - in.local);
    if (min(d.x, d.y) > 1.0) {
        discard;
    }
    return in.color;
}
