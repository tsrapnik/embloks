import numpy as np

from embloks.config import Settings
from embloks.connectivity import Connectivity
from embloks.fields import ElectricField, interpolate
from embloks.materials import Material

# Particles stopped at a wall stay this far inside their cell.
_WALL_GAP = 1e-6


def damping_rates(world, cols: np.ndarray, rows: np.ndarray, settings: Settings) -> np.ndarray:
    """Drag rate (1/time) for particles in the given cells: proportional to resistance."""
    m = world.materials
    resistance = np.where(
        m.kind[cols, rows] == Material.BATTERY, settings.battery_resistance, m.resistance[cols, rows]
    )
    return resistance * settings.drag_per_resistance


def kick(vel: np.ndarray, acc: np.ndarray, damping: np.ndarray, dt: float) -> np.ndarray:
    """Velocity after dt under acceleration `acc` and drag -damping * v.

    Drag is treated implicitly, which is stable for any damping * dt; at steady
    state v = acc / damping, i.e. drift velocity proportional to E (Ohm's law).
    """
    return (vel + acc * dt) / (1 + damping * dt)[:, None]


def confine(
    old_pos: np.ndarray,
    new_pos: np.ndarray,
    vel: np.ndarray,
    connectivity: Connectivity,
    restitution: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounce particles off closed cell faces, so they never leave their island.

    Assumes each particle moved less than one cell per axis. The x move is
    resolved first, then the y move from the resulting column.
    """
    pos, vel = new_pos.copy(), vel.copy()
    old_cell = np.floor(old_pos).astype(np.int64)
    cols, rows = connectivity.labels.shape

    def resolve(axis: int, start: np.ndarray, other: np.ndarray, open_faces: np.ndarray, n: int):
        end = np.floor(pos[:, axis]).astype(np.int64)
        moved = end != start
        forward = end > start
        # Faces are indexed by the cell before them; the grid edge counts as closed.
        face = np.where(forward, start, start - 1)
        on_grid = moved & (face >= 0) & (face < n - 1)
        is_open = np.zeros(len(pos), dtype=bool)
        if axis == 0:
            is_open[on_grid] = open_faces[face[on_grid], other[on_grid]]
        else:
            is_open[on_grid] = open_faces[other[on_grid], face[on_grid]]
        blocked = moved & ~is_open
        wall = np.where(forward, start + 1, start)[blocked]
        bounced = wall - restitution * (pos[blocked, axis] - wall)
        low, high = start[blocked] + _WALL_GAP, start[blocked] + 1 - _WALL_GAP
        pos[blocked, axis] = np.clip(bounced, low, high)
        vel[blocked, axis] *= -restitution

    resolve(0, old_cell[:, 0], old_cell[:, 1], connectivity.open_x, cols)
    resolve(1, old_cell[:, 1], np.floor(pos[:, 0]).astype(np.int64), connectivity.open_y, rows)
    return pos, vel


def step(world, electric: ElectricField, dt: float, settings: Settings) -> int:
    """Advance the electrons by dt. Returns the number of substeps used.

    Substeps keep every move below max_step cells, so confine() sees at most
    one face crossing per axis. Forces come from the grid field, read back
    with the same weights used to deposit charge (no self-force).
    """
    p = world.particles
    if len(p) == 0:
        return 0
    s = settings
    speed = np.sqrt((p.vel**2).sum(axis=1)).max()
    substeps = int(np.clip(np.ceil(speed * dt / s.max_step), 1, s.max_substeps))
    h = dt / substeps
    for _ in range(substeps):
        field = interpolate(electric.grid(world), p.pos)
        acc = (p.charge / p.mass)[:, None] * field
        cols, rows = p.cells()
        vel = kick(p.vel, acc, damping_rates(world, cols, rows, s), h)
        move = vel * h
        # Safety net if velocity jumped within this substep: never move further than max_step.
        length = np.sqrt((move**2).sum(axis=1))
        move *= np.minimum(1, s.max_step / np.maximum(length, 1e-300))[:, None]
        p.pos, p.vel = confine(p.pos, p.pos + move, vel, world.connectivity, s.wall_restitution)
    return substeps
