// The fields (maxwell.rs) and what couples them to the electrons: deposit,
// smoothing, FDTD, Gauss's law via a sine-transform Poisson solve, batteries,
// display averaging and the info panel's probe.

fn nodes() -> u32 {
    return P.nx * P.ny;
}

fn ex_at(i: u32, j: u32) -> u32 { return P.o_ex + i * P.ny + j; }
fn ey_at(i: u32, j: u32) -> u32 { return P.o_ey + i * (P.ny + 1u) + j; }
fn bz_at(i: u32, j: u32) -> u32 { return P.o_bz + i * (P.ny + 1u) + j; }

// --- Sources ------------------------------------------------------------------------

// Charge and current of the electrons at node i (cloud in cell, gathered from the
// sorted cells around it so no float atomics are needed), plus the static charge
// of ions and test charges. Zero outside the visible grid.
@compute @workgroup_size(64)
fn deposit(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = thread_index(g);
    if (i >= nodes()) {
        return;
    }
    let c = i32(i / P.ny) - i32(P.margin);
    let r = i32(i % P.ny) - i32(P.margin);
    var acc = vec3(0.0);
    if (c >= 0 && r >= 0 && c < i32(P.cols) && r < i32(P.rows)) {
        acc.x = F[P.o_static_rho + flat(vec2(c, r))];
        let last = vec2<i32>(i32(P.cols) - 1, i32(P.rows) - 1);
        for (var cc = max(c - 1, 0); cc <= min(c + 1, last.x); cc++) {
            for (var rr = max(r - 1, 0); rr <= min(r + 1, last.y); rr++) {
                let cell = flat(vec2(cc, rr));
                for (var slot = cell_start(cell); slot < cell_start(cell + 1u); slot++) {
                    let p = parts[sorted(slot)];
                    let u = p.xy - 0.5;
                    let corner = floor(u);
                    let f = u - corner;
                    let lo = clamp(vec2<i32>(corner), vec2(0), last);
                    let hi = clamp(vec2<i32>(corner) + 1, vec2(0), last);
                    let wx = select(0.0, 1.0 - f.x, lo.x == c) + select(0.0, f.x, hi.x == c);
                    let wy = select(0.0, 1.0 - f.y, lo.y == r) + select(0.0, f.y, hi.y == r);
                    acc += wx * wy * P.charge * vec3(1.0, p.z, p.w);
                }
            }
        }
    }
    F[P.o_src + i] = acc.x;
    F[P.o_src + nodes() + i] = acc.y;
    F[P.o_src + 2u * nodes() + i] = acc.z;
}

// One 1-2-1 pass along x (src -> tmp) or y (tmp -> src), zero beyond the edges.
fn smooth_pass(source: u32, dest: u32, i: u32, along_x: bool) {
    let x = i / P.ny;
    let y = i % P.ny;
    let stride = select(1u, P.ny, along_x);
    let pos = select(y, x, along_x);
    let len = select(P.ny, P.nx, along_x);
    for (var ch = 0u; ch < 3u; ch++) {
        let b = ch * nodes() + i;
        var v = 0.5 * F[source + b];
        if (pos > 0u) {
            v += 0.25 * F[source + b - stride];
        }
        if (pos + 1u < len) {
            v += 0.25 * F[source + b + stride];
        }
        F[dest + b] = v;
    }
}

@compute @workgroup_size(64)
fn smooth_x(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = thread_index(g);
    if (i < nodes()) {
        smooth_pass(P.o_src, P.o_tmp, i, true);
    }
}

@compute @workgroup_size(64)
fn smooth_y(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = thread_index(g);
    if (i < nodes()) {
        smooth_pass(P.o_tmp, P.o_src, i, false);
    }
}

// Conventional current per visible cell: the sum of q v over its electrons
// (physics::current_density).
@compute @workgroup_size(64)
fn current(@builtin(global_invocation_id) g: vec3<u32>) {
    let cell = thread_index(g);
    if (cell >= P.cells) {
        return;
    }
    var j = vec2(0.0);
    for (var slot = cell_start(cell); slot < cell_start(cell + 1u); slot++) {
        j += parts[sorted(slot)].zw;
    }
    F[P.o_current + cell] = P.charge * j.x;
    F[P.o_current + P.cells + cell] = P.charge * j.y;
}

// --- FDTD (Maxwell::step) -------------------------------------------------------------

fn curl_e(i: u32, j: u32) -> f32 {
    var v = 0.0;
    if (i < P.nx) {
        v += F[ey_at(i, j)];
    }
    if (i > 0u) {
        v -= F[ey_at(i - 1u, j)];
    }
    if (j < P.ny) {
        v -= F[ex_at(i, j)];
    }
    if (j > 0u) {
        v += F[ex_at(i, j - 1u)];
    }
    return v;
}

@compute @workgroup_size(64)
fn bz_half(@builtin(global_invocation_id) g: vec3<u32>) {
    let k = thread_index(g);
    if (k >= (P.nx + 1u) * (P.ny + 1u)) {
        return;
    }
    let i = k / (P.ny + 1u);
    let j = k % (P.ny + 1u);
    F[P.o_bz + k] -= 0.5 * P.fdtd_h * curl_e(i, j);
}

// The second half step of B, then damping of B in the absorbing margin.
@compute @workgroup_size(64)
fn bz_half_damp(@builtin(global_invocation_id) g: vec3<u32>) {
    let k = thread_index(g);
    if (k >= (P.nx + 1u) * (P.ny + 1u)) {
        return;
    }
    let i = k / (P.ny + 1u);
    let j = k % (P.ny + 1u);
    F[P.o_bz + k] = (F[P.o_bz + k] - 0.5 * P.fdtd_h * curl_e(i, j)) * F[P.o_damp_bz + k];
}

// A full step of E from curl B and the current, Ex and Ey in one dispatch.
@compute @workgroup_size(64)
fn e_step(@builtin(global_invocation_id) g: vec3<u32>) {
    let k = thread_index(g);
    let ex_count = (P.nx + 1u) * P.ny;
    let scale = 0.5 * P.four_pi_k * P.fdtd_h;
    let jx = P.o_src + nodes();
    let jy = P.o_src + 2u * nodes();
    if (k < ex_count) {
        let i = k / P.ny;
        let j = k % P.ny;
        var current = 0.0;
        if (i > 0u && i < P.nx) {
            current = scale * (F[jx + (i - 1u) * P.ny + j] + F[jx + i * P.ny + j]);
        }
        F[P.o_ex + k] += P.fdtd_h * P.c2 * (F[bz_at(i, j + 1u)] - F[bz_at(i, j)]) - current;
        return;
    }
    let q = k - ex_count;
    if (q >= P.nx * (P.ny + 1u)) {
        return;
    }
    let i = q / (P.ny + 1u);
    let j = q % (P.ny + 1u);
    var current = 0.0;
    if (j > 0u && j < P.ny) {
        current = scale * (F[jy + i * P.ny + j - 1u] + F[jy + i * P.ny + j]);
    }
    F[P.o_ey + q] -= P.fdtd_h * P.c2 * (F[bz_at(i + 1u, j)] - F[bz_at(i, j)]) + current;
}

@compute @workgroup_size(64)
fn damp_e(@builtin(global_invocation_id) g: vec3<u32>) {
    let k = thread_index(g);
    let ex_count = (P.nx + 1u) * P.ny;
    if (k < ex_count) {
        F[P.o_ex + k] *= F[P.o_damp_ex + k];
    } else if (k - ex_count < P.nx * (P.ny + 1u)) {
        F[P.o_ey + k - ex_count] *= F[P.o_damp_ey + k - ex_count];
    }
}

// --- Gauss's law (Maxwell::enforce_gauss) ---------------------------------------------

// Right-hand sides of the two Poisson problems: the charge's potential and the
// divergence of E (whose gradient is then taken out of E).
@compute @workgroup_size(64)
fn gauss_rhs(@builtin(global_invocation_id) g: vec3<u32>) {
    let k = thread_index(g);
    if (k >= nodes()) {
        return;
    }
    let i = k / P.ny;
    let j = k % P.ny;
    let div = F[ex_at(i + 1u, j)] - F[ex_at(i, j)] + F[ey_at(i, j + 1u)] - F[ey_at(i, j)];
    F[P.o_work + k] = -P.four_pi_k * F[P.o_src + k];
    F[P.o_work + nodes() + k] = div;
}

// E -= grad(potential + correction), with both zero beyond the grid.
@compute @workgroup_size(64)
fn gauss_correct(@builtin(global_invocation_id) g: vec3<u32>) {
    let k = thread_index(g);
    let ex_count = (P.nx + 1u) * P.ny;
    if (k < ex_count) {
        let i = k / P.ny;
        let j = k % P.ny;
        var grad = 0.0;
        if (i < P.nx) {
            grad += F[P.o_phi + i * P.ny + j] + F[P.o_phi + nodes() + i * P.ny + j];
        }
        if (i > 0u) {
            grad -= F[P.o_phi + (i - 1u) * P.ny + j] + F[P.o_phi + nodes() + (i - 1u) * P.ny + j];
        }
        F[P.o_ex + k] -= grad;
        return;
    }
    let q = k - ex_count;
    if (q >= P.nx * (P.ny + 1u)) {
        return;
    }
    let i = q / (P.ny + 1u);
    let j = q % (P.ny + 1u);
    var grad = 0.0;
    if (j < P.ny) {
        grad += F[P.o_phi + i * P.ny + j] + F[P.o_phi + nodes() + i * P.ny + j];
    }
    if (j > 0u) {
        grad -= F[P.o_phi + i * P.ny + j - 1u] + F[P.o_phi + nodes() + i * P.ny + j - 1u];
    }
    F[P.o_ey + q] -= grad;
}

// --- Display and smoothed current -------------------------------------------------------

fn node_e(c: u32, r: u32) -> vec2<f32> {
    let i = c + P.margin;
    let j = r + P.margin;
    return vec2(0.5 * (F[ex_at(i, j)] + F[ex_at(i + 1u, j)]), 0.5 * (F[ey_at(i, j)] + F[ey_at(i, j + 1u)]));
}

fn node_b(c: u32, r: u32) -> f32 {
    let i = c + P.margin;
    let j = r + P.margin;
    return 0.25 * (F[bz_at(i, j)] + F[bz_at(i + 1u, j)] + F[bz_at(i, j + 1u)] + F[bz_at(i + 1u, j + 1u)]);
}

// E and B at the cell centres, averaged over display_smoothing_time (App::display_fields).
fn update_display(cell: u32) {
    let c = cell / P.rows;
    let r = cell % P.rows;
    let e = node_e(c, r);
    let b = node_b(c, r);
    var alpha = P.alpha_display;
    if (atomicLoad(&state[ST_RESET_DISPLAY]) != 0u) {
        alpha = 1.0;
    }
    let o = P.o_display;
    F[o + cell] += alpha * (e.x - F[o + cell]);
    F[o + P.cells + cell] += alpha * (e.y - F[o + P.cells + cell]);
    F[o + 2u * P.cells + cell] += alpha * (b - F[o + 2u * P.cells + cell]);
}

// End of a tick: the display average and the ammeter-like smoothed current.
@compute @workgroup_size(64)
fn finish_tick(@builtin(global_invocation_id) g: vec3<u32>) {
    let cell = thread_index(g);
    if (cell >= P.cells) {
        return;
    }
    update_display(cell);
    for (var k = 0u; k < 2u; k++) {
        let i = k * P.cells + cell;
        F[P.o_current_smooth + i] += P.alpha_current * (F[P.o_current + i] - F[P.o_current_smooth + i]);
    }
}

// After an edit: only the display (reset to the exact fields).
@compute @workgroup_size(64)
fn finish_sync(@builtin(global_invocation_id) g: vec3<u32>) {
    let cell = thread_index(g);
    if (cell < P.cells) {
        update_display(cell);
    }
}

// --- Batteries (battery::trim) ----------------------------------------------------------

// Per stack, a header of HEADER words in S, then cell lists:
//   0 low list offset, 1 low count, 2 high list offset, 3 high count,
//   4 current list offset, 5 current count, 6 dir x, 7 dir y (f32),
//   8 amplitude, 9 internal resistance (f32), 10 both terminals wired
const HEADER: u32 = 12u;

fn mean_potential(offset: u32, count: u32) -> f32 {
    var sum = 0.0;
    for (var k = 0u; k < count; k++) {
        let cell = S[P.o_stack_cells + offset + k];
        sum += potential_at(i32(cell / P.rows), i32(cell % P.rows));
    }
    return sum / f32(count);
}

fn stack_current(h: u32, current: u32) -> f32 {
    let dir = vec2(sf(h + 6u), sf(h + 7u));
    var sum = 0.0;
    for (var k = 0u; k < S[h + 5u]; k++) {
        let cell = S[P.o_stack_cells + S[h + 4u] + k];
        sum += dot(vec2(F[current + cell], F[current + P.cells + cell]), dir);
    }
    return sum / f32(S[h + 5u]);
}

@compute @workgroup_size(64)
fn trim(@builtin(global_invocation_id) g: vec3<u32>) {
    let s = thread_index(g);
    if (s >= P.n_stacks) {
        return;
    }
    let h = P.o_stack_header + s * HEADER;
    let amplitude = sf(h + 8u);
    let reading = P.o_stack_readings + s * 4u;
    if (S[h + 10u] == 0u || amplitude == 0.0) {
        S[P.o_gain + s] = bitcast<u32>(1.0);
        S[reading + 2u] = 0u;
        return;
    }
    let v_low = mean_potential(S[h], S[h + 1u]);
    let v_high = mean_potential(S[h + 2u], S[h + 3u]);
    let target_emf = sf(P.o_targets + atomicLoad(&state[ST_TICK]) * P.n_stacks + s);
    let delivered = v_high - v_low + stack_current(h, P.o_current) * sf(h + 9u);
    var gain = sf(P.o_gain + s);
    gain += P.trim_rate * (target_emf - delivered) * target_emf / (amplitude * amplitude);
    S[P.o_gain + s] = bitcast<u32>(clamp(gain, P.gain_low, P.gain_high));
    S[reading] = bitcast<u32>(v_low);
    S[reading + 1u] = bitcast<u32>(v_high);
    S[reading + 2u] = 1u;
}

// --- Info panel probe -------------------------------------------------------------------

// Probe request (written by the CPU each frame) and result layout in S.
const REQ_CELL: u32 = 0u;
const REQ_STRIDE: u32 = 1u;
const OUT_VALID: u32 = 0u;

fn out(k: u32, v: f32) {
    S[P.o_probe_out + k] = bitcast<u32>(v);
}

fn smoothed_current(cell: u32) -> vec2<f32> {
    return vec2(F[P.o_current_smooth + cell], F[P.o_current_smooth + P.cells + cell]);
}

fn display_e(cell: u32) -> vec2<f32> {
    return vec2(F[P.o_display + cell], F[P.o_display + P.cells + cell]);
}

fn display_b(cell: u32) -> f32 {
    return F[P.o_display + 2u * P.cells + cell];
}

// Readings for the cell under the mouse (App::cell_reading), one thread.
@compute @workgroup_size(1)
fn probe_cell() {
    let cell = S[P.o_probe_request + REQ_CELL];
    if (cell >= P.cells) {
        out(OUT_VALID, 0.0);
        return;
    }
    let c = i32(cell / P.rows);
    let r = i32(cell % P.rows);
    let e = display_e(cell);
    let b = display_b(cell);
    out(OUT_VALID, 1.0);
    out(1u, f32(cell_start(cell + 1u) - cell_start(cell)));
    out(2u, potential_at(c, r));
    out(3u, e.x);
    out(4u, e.y);
    out(5u, b);
    out(6u, P.inv_mu0 * e.y * b);
    out(7u, -P.inv_mu0 * e.x * b);
    let current = smoothed_current(cell);
    out(8u, current.x);
    out(9u, current.y);
    // Clamp meter: the main direction of flow here, summed across the wire.
    let along = select(1u, 0u, abs(current.x) >= abs(current.y));
    var total = current[along];
    var width = 1.0;
    let across = select(vec2<i32>(0, 1), vec2<i32>(1, 0), along == 1u);
    for (var side = -1; side <= 1; side += 2) {
        var at = vec2(c, r) + side * across;
        while (at.x >= 0 && at.y >= 0 && at.x < i32(P.cols) && at.y < i32(P.rows) && conductive(at.x, at.y)) {
            total += smoothed_current(flat(at))[along];
            width += 1.0;
            at += side * across;
        }
    }
    out(10u, total);
    out(11u, width);
    let stack = info[cell].stack;
    out(12u, 0.0);
    if (stack > 0u) {
        let s = stack - 1u;
        let h = P.o_stack_header + s * HEADER;
        let reading = P.o_stack_readings + s * 4u;
        out(12u, select(1.0, 2.0, S[h + 10u] != 0u && S[reading + 2u] != 0u));
        out(13u, sf(reading));
        out(14u, sf(reading + 1u));
        out(15u, stack_current(h, P.o_current_smooth));
        out(16u, sf(P.o_gain + s));
    }
}

var<workgroup> partial: array<vec2<f32>, 256>;

fn histogram(layer: u32, magnitude: f32) {
    if (magnitude > 0.0) {
        let bin = u32(clamp((log2(magnitude) * 0.30103 + 12.0) / 18.0 * f32(HIST_BINS), 0.0, f32(HIST_BINS - 1u)));
        atomicAdd(&state[ST_HIST + layer * HIST_BINS + bin], 1u);
    }
}

// Mean |E| over the wire cells, and histograms of the displayed magnitudes for the
// layers' auto gain (Layer::follow). One workgroup.
@compute @workgroup_size(256)
fn probe_grid(@builtin(local_invocation_id) l: vec3<u32>) {
    let t = l.x;
    for (var k = t; k < 3u * HIST_BINS; k += 256u) {
        atomicStore(&state[ST_HIST + k], 0u);
    }
    workgroupBarrier();
    let stride = max(S[P.o_probe_request + REQ_STRIDE], 1u);
    var sum = vec2(0.0);
    for (var cell = t; cell < P.cells; cell += 256u) {
        let e = display_e(cell);
        let b = display_b(cell);
        if ((info[cell].flags & 2u) != 0u) {
            sum += vec2(length(e), 1.0);
        }
        histogram(1u, abs(b) * P.c2);
        let c = cell / P.rows;
        let r = cell % P.rows;
        if (c % stride == stride / 2u && r % stride == stride / 2u) {
            histogram(0u, length(e));
            histogram(2u, P.inv_mu0 * length(e) * abs(b));
        }
    }
    partial[t] = sum;
    workgroupBarrier();
    for (var half = 128u; half > 0u; half /= 2u) {
        if (t < half) {
            partial[t] += partial[t + half];
        }
        workgroupBarrier();
    }
    if (t == 0u) {
        out(17u, partial[0].x / max(partial[0].y, 1.0));
        out(18u, partial[0].y);
    }
}
