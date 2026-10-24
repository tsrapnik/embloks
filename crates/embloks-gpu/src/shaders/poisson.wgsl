// Poisson's equation by sine transforms (Maxwell::poisson): the type-I DST along each
// axis is a multiplication by a sine matrix, so the whole solve is four matrix
// products on the two right-hand sides at once (potential and correction):
//   MODE 0: work2 = Sx . work           MODE 1: work = (work2 . Sy) * inverse Laplacian
//   MODE 2: work2 = Sx . work (again)   MODE 3: phi = work2 . Sy
// Dispatch (ceil(nx / 16), ceil(ny / 16), 2): one tile of the output per workgroup.

override MODE: u32 = 0u;

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

fn is_left() -> bool {
    return MODE == 0u || MODE == 2u;
}

// Element (row, k) of the left factor and (k, col) of the right one, for channel ch.
fn factor_a(ch: u32, row: u32, k: u32) -> f32 {
    if (is_left()) {
        return T[P.o_sine_x + row * P.nx + k];
    }
    return F[P.o_work2 + ch * nodes() + row * P.ny + k];
}

fn factor_b(ch: u32, k: u32, col: u32) -> f32 {
    if (is_left()) {
        return F[P.o_work + ch * nodes() + k * P.ny + col];
    }
    return T[P.o_sine_y + k * P.ny + col];
}

@compute @workgroup_size(16, 16)
fn matmul(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let ch = wg.z;
    let row = wg.x * 16u + l.y;
    let col = wg.y * 16u + l.x;
    let inner = select(P.ny, P.nx, is_left());
    var acc = 0.0;
    for (var t = 0u; t < inner; t += 16u) {
        let ka = t + l.x;
        let kb = t + l.y;
        tile_a[l.y][l.x] = select(0.0, factor_a(ch, row, ka), row < P.nx && ka < inner);
        tile_b[l.y][l.x] = select(0.0, factor_b(ch, kb, col), kb < inner && col < P.ny);
        workgroupBarrier();
        for (var k = 0u; k < 16u; k++) {
            acc += tile_a[l.y][k] * tile_b[k][l.x];
        }
        workgroupBarrier();
    }
    if (row >= P.nx || col >= P.ny) {
        return;
    }
    let at = ch * nodes() + row * P.ny + col;
    if (MODE == 1u) {
        F[P.o_work + at] = acc * T[P.o_inv_laplacian + row * P.ny + col];
    } else if (MODE == 3u) {
        F[P.o_phi + at] = acc;
    } else {
        F[P.o_work2 + at] = acc;
    }
}
