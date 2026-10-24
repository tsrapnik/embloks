import numpy as np
import pytest

from embloks import physics, scenes
from embloks.config import Settings
from embloks.connectivity import compute_connectivity
from embloks.close_range import CloseRange
from embloks.maxwell import Maxwell
from embloks.materials import Material, Materials, Orientation
from embloks.world import World
from sim import Sim


def test_kick_reaches_terminal_velocity_acc_over_damping():
    vel = np.zeros((1, 2))
    acc = np.array([[2.0, -1.0]])
    for _ in range(2000):
        vel = physics.kick(vel, acc, np.array([4.0]), 0.01)
    assert np.allclose(vel, acc / 4.0)


def test_kick_is_stable_for_huge_damping():
    vel = np.array([[5.0, 0.0]])
    new = physics.kick(vel, np.zeros((1, 2)), np.array([1e6]), 0.1)
    # Implicit drag only shrinks the velocity, it never flips or blows it up.
    assert 0 <= new[0, 0] < 1e-4


def make_connectivity(cells, battery=None, size=(6, 5)):
    m = Materials(*size)
    cols, rows = np.array(cells).T
    m.paint(cols, rows, Material.CONDUCTOR, resistance=1)
    if battery:
        (c, r), orientation = battery
        m.paint(np.array([c]), np.array([r]), Material.BATTERY, emf=1, orientation=orientation)
    return compute_connectivity(m)


def test_confine_lets_particles_cross_open_faces():
    conn = make_connectivity([(1, 1), (2, 1)])
    pos, vel = physics.confine(
        np.array([[1.9, 1.5]]), np.array([[2.1, 1.5]]), np.array([[1.0, 0.0]]), conn, 0.5
    )
    assert np.allclose(pos, [[2.1, 1.5]]) and np.allclose(vel, [[1.0, 0.0]])


@pytest.mark.parametrize(
    "old,new,vel",
    [
        ((1.9, 1.5), (2.1, 1.5), (1.0, 0.0)),  # Into vacuum on the right.
        ((1.5, 1.1), (1.5, 0.9), (0.0, -1.0)),  # Into vacuum above.
        ((1.1, 1.9), (0.9, 2.1), (-1.0, 1.0)),  # Diagonally out of a corner.
    ],
)
def test_confine_bounces_off_vacuum(old, new, vel):
    conn = make_connectivity([(1, 1)])
    pos, v = physics.confine(np.array([old]), np.array([new]), np.array([vel]), conn, 0.5)
    assert np.floor(pos).astype(int).tolist() == [[1, 1]]
    # Velocity components that hit a wall reverse and shrink.
    for axis in (0, 1):
        if vel[axis]:
            assert v[0, axis] == pytest.approx(-0.5 * vel[axis])


def test_confine_treats_grid_edge_as_wall():
    conn = make_connectivity([(0, 0)])
    pos, _ = physics.confine(
        np.array([[0.1, 0.5]]), np.array([[-0.1, 0.5]]), np.array([[-1.0, 0.0]]), conn, 0.0
    )
    assert 0 <= pos[0, 0] < 1


def test_confine_treats_batteries_as_walls():
    # Battery at (2, 1) pointing right, wired on both terminals.
    conn = make_connectivity([(1, 1), (3, 1)], battery=((2, 1), Orientation.RIGHT))
    pos, _ = physics.confine(
        np.array([[1.9, 1.5]]), np.array([[2.1, 1.5]]), np.array([[1.0, 0.0]]), conn, 0.0
    )
    assert np.floor(pos).astype(int).tolist() == [[1, 1]]


def test_random_walk_never_leaves_island():
    rng = np.random.default_rng(5)
    m = Materials(12, 10)
    shape = rng.uniform(size=(12, 10)) < 0.55
    m.paint(*np.nonzero(shape), Material.CONDUCTOR, resistance=1)
    conn = compute_connectivity(m)
    pos = np.argwhere(shape) + 0.5
    start_island = conn.labels[tuple(np.floor(pos).astype(int).T)]
    for _ in range(3000):
        move = rng.uniform(-0.45, 0.45, pos.shape)
        pos, _ = physics.confine(pos, pos + move, move, conn, rng.uniform())
        cells = tuple(np.floor(pos).astype(int).T)
        assert (conn.labels[cells] == start_island).all()


@pytest.fixture
def block_world():
    s = Settings()
    w = World(40, 32, s, rng=np.random.default_rng(0))
    w.paint(np.repeat(np.arange(14, 26), 6), np.tile(np.arange(13, 19), 12),
            Material.CONDUCTOR, resistance=1)
    return w


def run(world, seconds):
    return Sim(world).run(seconds)


def test_neutral_conductor_stays_quiet(block_world):
    # Electrons spawn spread over their cells; the outermost ones settle in ~4 time units.
    run(block_world, 5)
    speeds = np.hypot(*block_world.particles.vel.T)
    assert speeds.max() < 0.02


def test_conductor_screens_external_charge(block_world):
    w = block_world
    w.place_charges(np.array([8]), np.array([16]), 5)
    inner = np.zeros(w.materials.kind.shape, dtype=bool)
    inner[15:25, 14:18] = True  # Interior cells, one cell in from the surface.
    sim = Sim(w)
    unscreened = sim.e_field()[inner].mean(axis=0)
    sim.run(10)
    screened = sim.e_field()[inner].mean(axis=0)
    assert np.hypot(*screened) < 0.15 * np.hypot(*unscreened)
    # Electrons shift towards the positive charge on the left (block centre x = 20).
    # Screening a weak field needs only a little surface charge, so the shift is small.
    assert w.particles.pos[:, 0].mean() < 20.0 - 0.01
    cells = w.particles.cells()
    assert w.materials.conductive[cells].all()


def test_fast_particles_get_substeps(block_world):
    s = block_world.settings
    block_world.particles.vel[0] = [100.0, 0.0]
    substeps = physics.step(block_world, Maxwell.for_world(block_world, s), CloseRange(s), s.dt, s)
    assert substeps > 1


def test_step_without_particles_does_nothing():
    s = Settings()
    w = World(5, 5, s)
    assert physics.step(w, Maxwell.for_world(w, s), CloseRange(s), s.dt, s) == 0


def test_electrons_spread_over_a_thin_wire_instead_of_stacking():
    """Regression: electrons used to collapse onto the centre line, in exact stacks."""
    s = Settings()
    w = World(20, 14, s, rng=np.random.default_rng(0))
    w.paint(*scenes.outline(3, 16, 3, 10), Material.CONDUCTOR, resistance=1)
    p = w.particles
    p.pos = np.floor(p.pos) + 0.5 + w.rng.uniform(-0.02, 0.02, p.pos.shape)  # All stacked.
    run(w, 5)
    cols, rows = p.cells()
    across = np.where(np.isin(cols, (3, 16)), p.pos[:, 0], p.pos[:, 1]) % 1 - 0.5
    corner = np.isin(cols, (3, 16)) & np.isin(rows, (3, 10))
    # Filling the width evenly gives a mean distance of 0.25 from the centre line.
    assert np.abs(across[~corner]).mean() > 0.2
    gaps = np.hypot(*(p.pos[:, None] - p.pos[None]).transpose(2, 0, 1)) + 9 * np.eye(len(p))
    assert gaps.min() > 0.05
