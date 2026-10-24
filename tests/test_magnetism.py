import numpy as np
import pytest

from embloks import battery, physics
from embloks.config import Settings
from embloks.materials import Material, Orientation
from embloks.maxwell import Maxwell

# Screen y points down, so positive B_z points INTO the screen.


def steady_b(current: np.ndarray, settings: Settings, seconds: float = 40) -> np.ndarray:
    """B_z at the cell centres once a steady current has been flowing for a while."""
    cols, rows = current.shape[:2]
    mx = Maxwell(cols, rows, settings)
    rho = np.zeros((mx.nx, mx.ny))
    full = mx.smoothed(current)
    for _ in range(int(seconds / settings.dt)):
        mx.step(rho, full, settings.dt)
    return mx.node_fields()[1]


def test_current_loop_is_a_solenoid_following_the_right_hand_rule():
    """In 2D a loop is a long solenoid: uniform B inside, none outside."""
    s = Settings()
    current = np.zeros((30, 30, 2))
    # Clockwise on screen (y down): right along the top, down the right side, ...
    current[8:22, 8, 0] = 1.0
    current[22, 8:22, 1] = 1.0
    current[8:22, 22, 0] = -1.0
    current[8, 8:22, 1] = -1.0
    bz = steady_b(current, s)
    inside, outside = bz[12:19, 12:19], bz[1:5, 1:5]
    # Clockwise on screen is anticlockwise seen from behind it: B points into the screen.
    assert (inside > 0).all()
    assert inside.std() < 0.1 * inside.mean()
    assert np.abs(outside).max() < 0.1 * inside.mean()
    # Ampere's law for a solenoid: B = 4 pi k I / c^2 (current per unit length 1).
    assert inside.mean() == pytest.approx(4 * np.pi * s.coulomb_k / s.speed_of_light**2, rel=0.1)


def test_lorentz_magnetic_part_is_perpendicular_to_velocity():
    rng = np.random.default_rng(1)
    vel = rng.normal(size=(20, 2))
    force = physics.lorentz_field(np.zeros((20, 2)), vel, rng.normal(size=20))
    assert np.allclose((force * vel).sum(axis=1), 0)


def test_lorentz_force_steers_electrons_drifting_with_a_current_towards_it():
    """Parallel currents attract: B below a current to the right points into the
    screen, and an electron drifting left there is pushed up, towards the current."""
    force = -1.0 * physics.lorentz_field(np.zeros((1, 2)), np.array([[-1.0, 0.0]]), np.array([1.0]))
    assert force[0, 1] < 0


def test_battery_current_runs_along_its_stack():
    from embloks.world import World

    s = Settings()
    w = World(10, 6, s)
    w.paint(np.array([1, 2, 5, 6]), np.array([2, 2, 2, 2]), Material.CONDUCTOR, resistance=1)
    w.paint(np.array([3, 4]), np.array([2, 2]), Material.BATTERY, emf=2, orientation=Orientation.LEFT)
    [stack] = w.stacks
    w.pump_flow[stack] = -0.5  # Negative: conventional current from high (right) to low.
    grid = battery.current_grid(w)
    assert grid[3, 2].tolist() == [-0.5, 0.0] and grid[4, 2].tolist() == [-0.5, 0.0]
    assert np.count_nonzero(grid) == 2
