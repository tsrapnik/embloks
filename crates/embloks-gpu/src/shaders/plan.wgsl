// How many electron substeps this tick needs (physics::step), from the fastest
// electron: written as indirect dispatch arguments, so substeps beyond the count
// dispatch no workgroups at all. Its own small layout: the arguments buffer must
// not be writable storage in the dispatches that read it as arguments.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> state: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> args: array<u32>;

const ST_SUBSTEPS: u32 = 2u;
const ST_H: u32 = 3u;
const ST_MAX_SPEED: u32 = 4u;

fn put(slot: u32, kind: u32, groups: u32, on: bool) {
    let b = (slot * 3u + kind) * 3u;
    args[b] = select(0u, min(groups, 65535u), on);
    args[b + 1u] = select(1u, (groups + 65534u) / 65535u, on);
    args[b + 2u] = 1u;
}

@compute @workgroup_size(1)
fn plan() {
    var speed = bitcast<f32>(atomicLoad(&state[ST_MAX_SPEED]));
    if (!(speed >= 0.0)) {
        speed = 0.0;
    }
    let substeps = u32(clamp(ceil(speed * P.dt / P.max_step), 1.0, f32(P.max_substeps)));
    atomicStore(&state[ST_SUBSTEPS], substeps);
    atomicStore(&state[ST_H], bitcast<u32>(P.dt / f32(substeps)));
    for (var k = 0u; k < P.max_substeps; k++) {
        let on = k < substeps;
        put(k, 0u, (P.n + 63u) / 64u, on);
        put(k, 1u, (P.cells + 63u) / 64u, on);
        put(k, 2u, 1u, on);
    }
}
