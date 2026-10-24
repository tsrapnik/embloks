import numpy as np
import pytest

from embloks import battery, physics
from embloks.config import Settings
from embloks.fields import GridKernel, MagneticField, biot_savart, interpolate
from embloks.materials import Material, Orientation
from embloks.world import World

# Screen y points down, so positive B_z points INTO the screen.


def test_right_hand_rule_for_a_moving_charge():
    # +1 charge at the origin moving right (+x).
    above = biot_savart(np.array([[0.0, -3.0]]), np.zeros((1, 2)), np.array([[1.0, 0.0]]), 0.1)
    below = biot_savart(np.array([[0.0, 3.0]]), np.zeros((1, 2)), np.array([[1.0, 0.0]]), 0.1)
    # Current to the right: field comes out of the screen above it, goes in below it.
    assert above[0] < 0 < below[0]


def test_moving_charge_field_magnitude_and_no_field_along_motion():
    c = 2.0
    beside = biot_savart(np.array([[0.0, 10.0]]), np.zeros((1, 2)), np.array([[3.0, 0.0]]), 0.1, c=c)
    ahead = biot_savart(np.array([[10.0, 0.0]]), np.zeros((1, 2)), np.array([[3.0, 0.0]]), 0.1, c=c)
    assert abs(beside[0]) == pytest.approx(3.0 / (c**2 * 100), rel=1e-3)
    assert ahead[0] == pytest.approx(0)


def test_grid_curl_matches_direct_biot_savart():
    rng = np.random.default_rng(4)
    current = rng.normal(size=(9, 7, 2)) * (rng.uniform(size=(9, 7, 1)) < 0.4)
    via_fft = GridKernel(9, 7, softening=0.8, corrected=False).curl(current) / 2.0**2
    cells = np.argwhere(np.ones((9, 7), dtype=bool))
    direct = biot_savart(cells + 0.5, cells + 0.5, current.reshape(-1, 2), 0.8, c=2.0)
    assert np.allclose(via_fft.reshape(-1), direct, atol=1e-10)


def test_straight_current_field_flips_across_wire_and_falls_off_like_1_over_r():
    s = Settings()
    w = World(80, 60, s)
    current = np.zeros((80, 60, 2))
    current[:, 30, 0] = 1.0  # A long wire along row 30 carrying current to the right.
    bz = MagneticField(s).grid(w, current)
    assert bz[40, 26] < 0 < bz[40, 34]  # Out above, into below.
    assert bz[40, 26] / bz[40, 22] == pytest.approx(2, rel=0.1)  # 4 vs 8 cells away.


def test_lorentz_magnetic_part_is_perpendicular_to_velocity():
    rng = np.random.default_rng(1)
    vel = rng.normal(size=(20, 2))
    force = physics.lorentz_field(np.zeros((20, 2)), vel, rng.normal(size=20))
    assert np.allclose((force * vel).sum(axis=1), 0)


def test_parallel_currents_attract():
    """Electrons in wire A, drifting with the current in wire B, are pushed towards B."""
    s = Settings()
    above = biot_savart(  # B at wire A (y = 5) from current +x in wire B (y = 10).
        np.array([[20.0, 5.0]]), np.array([[x, 10.0] for x in range(41)]),
        np.tile([1.0, 0.0], (41, 1)), s.conductor_softening, c=s.speed_of_light,
    )
    # Conventional current +x in A too, so its electrons (charge -1) move -x.
    force = -1.0 * physics.lorentz_field(np.zeros((1, 2)), np.array([[-1.0, 0.0]]), above)
    assert force[0, 1] > 0  # Towards +y, i.e. towards wire B.


def test_no_magnetic_self_force_on_a_lone_moving_electron():
    s = Settings()
    w = World(20, 14, s)
    w.particles.add(np.array([[7.3, 4.8]]), -1.0, 1.0)
    w.particles.vel[:] = [0.7, -0.4]
    magnetic = MagneticField(s)
    own = interpolate(magnetic.grid(w, magnetic.electron_current(w))[..., None], w.particles.pos)
    assert abs(own[0, 0]) < 1e-12


def test_battery_current_runs_along_its_stack():
    s = Settings()
    w = World(10, 6, s)
    w.paint(np.array([1, 2, 5, 6]), np.array([2, 2, 2, 2]), Material.CONDUCTOR, resistance=1)
    w.paint(np.array([3, 4]), np.array([2, 2]), Material.BATTERY, emf=2, orientation=Orientation.LEFT)
    [stack] = w.stacks
    w.pump_current[stack] = -0.5  # Negative: conventional current from high (right) to low.
    grid = battery.current_grid(w)
    assert grid[3, 2].tolist() == [-0.5, 0.0] and grid[4, 2].tolist() == [-0.5, 0.0]
    assert np.count_nonzero(grid) == 2
