// Electrons: sorting them by cell, and moving them (physics::step).

@compute @workgroup_size(1)
fn begin_tick() {
    // The CPU sets the tick counter to u32::MAX before each frame's ticks.
    atomicAdd(&state[ST_TICK], 1u);
    atomicStore(&state[ST_SUBSTEP], 0xffffffffu);
    atomicStore(&state[ST_MAX_SPEED], 0u);
}

@compute @workgroup_size(1)
fn begin_substep() {
    atomicAdd(&state[ST_SUBSTEP], 1u);
}

@compute @workgroup_size(1)
fn end_tick() {
    atomicStore(&state[ST_RESET_DISPLAY], 0u);
}

// --- Counting sort by cell (close_range::CellList) ---------------------------------

@compute @workgroup_size(64)
fn clear_counts(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = thread_index(g);
    if (i < P.cells) {
        atomicStore(&C[P.o_counts + i], 0u);
    }
}

@compute @workgroup_size(64)
fn count_cells(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = thread_index(g);
    if (i >= P.n) {
        return;
    }
    let cell = flat(cell_of(parts[i].xy));
    let rank = atomicAdd(&C[P.o_counts + cell], 1u);
    atomicStore(&C[P.o_pcell + i], cell);
    atomicStore(&C[P.o_prank + i], rank);
}

var<workgroup> chunk_sums: array<u32, 256>;

// Exclusive prefix sum of the counts into starts, by one workgroup: each thread
// sums a chunk of cells, thread 0 scans the chunk sums, then each thread fills in.
@compute @workgroup_size(256)
fn scan_counts(@builtin(local_invocation_id) l: vec3<u32>) {
    let t = l.x;
    let chunk = (P.cells + 255u) / 256u;
    let begin = min(t * chunk, P.cells);
    let end = min(begin + chunk, P.cells);
    var sum = 0u;
    for (var i = begin; i < end; i++) {
        sum += atomicLoad(&C[P.o_counts + i]);
    }
    chunk_sums[t] = sum;
    workgroupBarrier();
    if (t == 0u) {
        var acc = 0u;
        for (var k = 0u; k < 256u; k++) {
            let v = chunk_sums[k];
            chunk_sums[k] = acc;
            acc += v;
        }
        atomicStore(&C[P.o_start + P.cells], acc);
    }
    workgroupBarrier();
    var acc = chunk_sums[t];
    for (var i = begin; i < end; i++) {
        atomicStore(&C[P.o_start + i], acc);
        acc += atomicLoad(&C[P.o_counts + i]);
    }
}

@compute @workgroup_size(64)
fn scatter_cells(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = thread_index(g);
    if (i >= P.n) {
        return;
    }
    let cell = atomicLoad(&C[P.o_pcell + i]);
    let slot = cell_start(cell) + atomicLoad(&C[P.o_prank + i]);
    atomicStore(&C[P.o_order + slot], i);
}

// --- Moving -----------------------------------------------------------------------

@compute @workgroup_size(64)
fn max_speed(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = thread_index(g);
    if (i < P.n) {
        // Non-negative floats order like their bit patterns.
        atomicMax(&state[ST_MAX_SPEED], bitcast<u32>(length(parts[i].zw)));
    }
}

const ORIENTATIONS = array<vec2<f32>, 4>(vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(-1.0, 0.0), vec2(0.0, -1.0));

// The batteries' impressed field in a cell at time t (battery::impressed_field_grid).
fn impressed(cell: CellInfo, t: f32) -> vec2<f32> {
    if (cell.emf == 0.0) {
        return vec2(0.0);
    }
    var value = cell.emf;
    if (cell.stack > 0u) {
        value *= sf(P.o_gain + cell.stack - 1u);
    }
    if (cell.frequency > 0.0) {
        value *= sin(2.0 * PI * cell.frequency * t);
    }
    return value * ORIENTATIONS[(cell.flags >> 2u) & 3u];
}

// Close-range force on electron i at `pos` (CloseRange::force): pairs with nearby
// electrons and the evenly spread ions of nearby cells.
fn close_force(i: u32, pos: vec2<f32>, home: vec2<i32>) -> vec2<f32> {
    let reach = i32(P.reach);
    let cutoff2 = P.cutoff * P.cutoff;
    var force = vec2(0.0);
    for (var c = max(home.x - reach, 0); c <= min(home.x + reach, i32(P.cols) - 1); c++) {
        for (var r = max(home.y - reach, 0); r <= min(home.y + reach, i32(P.rows) - 1); r++) {
            let cell = flat(vec2<i32>(c, r));
            let density = info[cell].density;
            if (density > 0.0) {
                let uv = (pos - vec2<f32>(f32(c), f32(r)) - 0.5 + P.half) * P.square_samples;
                force += P.k_ion * density * P.charge * vec2(square(1u, uv.x, uv.y), square(2u, uv.x, uv.y));
            }
            for (var slot = cell_start(cell); slot < cell_start(cell + 1u); slot++) {
                let j = sorted(slot);
                if (j == i) {
                    continue;
                }
                let d = pos - parts[j].xy;
                let d2 = dot(d, d);
                if (d2 >= cutoff2) {
                    continue;
                }
                let t = d2 * P.g_scale;
                let s = u32(t);
                let gs = T[P.o_g + s];
                force += P.k * P.charge * P.charge * (gs + (T[P.o_g + s + 1u] - gs) * (t - f32(s))) * d;
            }
        }
    }
    return force;
}

fn conductive(c: i32, r: i32) -> bool {
    return (info[flat(vec2<i32>(c, r))].flags & 1u) != 0u;
}

// Whether the face between two neighbouring cells along `axis` conducts; `face` is
// the cell before it along the axis, `other` the index across.
fn face_open(axis: u32, face: i32, other: i32) -> bool {
    if (axis == 0u) {
        if (face < 0 || face >= i32(P.cols) - 1 || other < 0 || other >= i32(P.rows)) {
            return false;
        }
        return conductive(face, other) && conductive(face + 1, other);
    }
    if (face < 0 || face >= i32(P.rows) - 1 || other < 0 || other >= i32(P.cols)) {
        return false;
    }
    return conductive(other, face) && conductive(other, face + 1);
}

// Bounce off closed faces (physics::confine), axis by axis.
fn confine(old_pos: vec2<f32>, new_pos: vec2<f32>, vel: vec2<f32>) -> vec4<f32> {
    var pos = new_pos;
    var v = vel;
    let old_cell = vec2<i32>(floor(old_pos));
    for (var axis = 0u; axis < 2u; axis++) {
        let start = old_cell[axis];
        let end = i32(floor(pos[axis]));
        if (end == start) {
            continue;
        }
        let other = select(i32(floor(pos.x)), old_cell.y, axis == 0u);
        let forward = end > start;
        let face = select(start - 1, start, forward);
        if (face_open(axis, face, other)) {
            continue;
        }
        let wall = f32(select(start, start + 1, forward));
        let bounced = wall - P.restitution * (pos[axis] - wall);
        pos[axis] = clamp(bounced, f32(start) + P.wall_gap, f32(start) + 1.0 - P.wall_gap);
        v[axis] *= -P.restitution;
    }
    return vec4(pos, v);
}

@compute @workgroup_size(64)
fn push(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = thread_index(g);
    if (i >= P.n) {
        return;
    }
    let p = parts[i];
    let pos = p.xy;
    let vel = p.zw;
    let h = bitcast<f32>(atomicLoad(&state[ST_H]));
    let k = atomicLoad(&state[ST_SUBSTEP]);
    let t = sf(P.o_tick_time + atomicLoad(&state[ST_TICK])) + f32(k) * h;
    let home = cell_of(pos);
    let cell = info[flat(home)];
    let m = f32(P.margin);
    // Each component interpolated from its own staggered grid (Maxwell::gather).
    let e = vec2(
        bilinear(P.o_ex, P.nx + 1u, P.ny, pos.x + m, pos.y - 0.5 + m),
        bilinear(P.o_ey, P.nx, P.ny + 1u, pos.x - 0.5 + m, pos.y + m),
    ) + impressed(cell, t);
    let bz = bilinear(P.o_bz, P.nx + 1u, P.ny + 1u, pos.x + m, pos.y + m);
    let lorentz = e + vec2(vel.y * bz, -vel.x * bz);
    var acc = (P.charge / P.mass) * lorentz;
    if (P.cutoff > 0.0) {
        acc += close_force(i, pos, home) / P.mass;
    }
    // Implicit drag, then a move of at most max_step.
    let v = (vel + acc * h) / (1.0 + cell.damping * h);
    var moved = v * h;
    let len = length(moved);
    if (len > P.max_step) {
        moved *= P.max_step / len;
    }
    parts_out[i] = confine(pos, pos + moved, v);
}

@compute @workgroup_size(64)
fn copy_back(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = thread_index(g);
    if (i < P.n) {
        parts[i] = parts_out[i];
    }
}
