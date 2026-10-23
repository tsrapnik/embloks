import numpy as np
import pytest

from embloks.config import Settings
from embloks.fields import ElectricField, GridKernel, coulomb_field, deposit
from embloks.materials import Material
from embloks.world import World


def test_far_field_matches_coulomb():
    point = np.array([[30.0, 40.0]])  # Distance 50 from the origin.
    field = coulomb_field(point, np.zeros((1, 2)), np.array([2.0]), softening=0.5)[0]
    assert np.hypot(*field) == pytest.approx(2 / 50**2, rel=1e-3)
    # Points away from a positive charge.
    assert np.allclose(field / np.hypot(*field), [0.6, 0.8])


def test_field_points_towards_negative_charge():
    field = coulomb_field(np.array([[3.0, 0.0]]), np.zeros((1, 2)), np.array([-1.0]), 0.1)[0]
    assert field[0] < 0 and field[1] == pytest.approx(0)


def test_softening_keeps_field_finite_and_zero_at_the_charge():
    sources = np.array([[1.0, 1.0]])
    at_charge = coulomb_field(sources, sources, np.array([1.0]), softening=0.3)
    assert np.array_equal(at_charge, [[0.0, 0.0]])
    near = coulomb_field(np.array([[1.01, 1.0]]), sources, np.array([1.0]), softening=0.3)
    assert np.isfinite(near).all() and np.hypot(*near[0]) < 1 / 0.3**2


def test_equal_charges_cancel_at_midpoint():
    sources = np.array([[0.0, 0.0], [4.0, 0.0]])
    field = coulomb_field(np.array([[2.0, 0.0]]), sources, np.array([1.0, 1.0]), 0.5)
    assert np.allclose(field, 0)


def test_superposition_is_linear():
    rng = np.random.default_rng(1)
    points = rng.uniform(0, 10, (7, 2))
    a, b = rng.uniform(0, 10, (3, 2)), rng.uniform(0, 10, (4, 2))
    qa, qb = rng.normal(size=3), rng.normal(size=4)
    both = coulomb_field(points, np.concatenate([a, b]), np.concatenate([qa, qb]), 0.4)
    separate = coulomb_field(points, a, qa, 0.4) + coulomb_field(points, b, qb, 0.4)
    assert np.allclose(both, separate)


def test_grid_fft_field_matches_direct_sum():
    rng = np.random.default_rng(2)
    charge = rng.normal(size=(9, 6)) * (rng.uniform(size=(9, 6)) < 0.4)
    via_fft = GridKernel(9, 6, softening=0.8).field(charge)
    cells = np.argwhere(np.ones_like(charge, dtype=bool))
    sources = np.argwhere(charge != 0)
    direct = coulomb_field(cells + 0.5, sources + 0.5, charge[charge != 0], 0.8)
    assert np.allclose(via_fft.reshape(-1, 2), direct, atol=1e-10)


@pytest.fixture
def world():
    return World(20, 14, Settings(), rng=np.random.default_rng(0))


def test_deposit_conserves_charge_and_is_exact_on_centres():
    rng = np.random.default_rng(3)
    positions = rng.uniform(0, 5, (50, 2))  # Includes points near the edges.
    charges = rng.normal(size=50)
    assert deposit(positions, charges, 5, 5).sum() == pytest.approx(charges.sum())
    on_centre = deposit(np.array([[2.5, 1.5]]), np.array([3.0]), 5, 5)
    assert on_centre[2, 1] == 3.0 and on_centre.sum() == 3.0


def test_world_grid_field_matches_exact_field_away_from_electrons(world):
    world.paint(np.arange(3, 12), np.full(9, 5), Material.CONDUCTOR, resistance=1)
    world.place_charges(np.array([15]), np.array([10]), -2)
    electric = ElectricField(world.settings)
    grid = electric.grid(world)
    # Ion and test charge parts are exact everywhere.
    centres = np.argwhere(np.ones((20, 14), dtype=bool)) + 0.5
    exact_without_electrons = electric.at(world, centres) - coulomb_field(
        centres, world.particles.pos, world.particles.charge, world.settings.conductor_softening
    )
    approx_without_electrons = grid - electric.electron_grid(world)
    assert np.allclose(approx_without_electrons.reshape(-1, 2), exact_without_electrons)
    # Electrons are blurred, so compare only cells a few cells away from the wire.
    far = np.array([[0, 0], [7, 12], [19, 1], [15, 10]])
    exact = electric.at(world, far + 0.5)
    assert np.allclose(grid[far[:, 0], far[:, 1]], exact, rtol=0.05, atol=1e-3)


def test_ion_cache_is_refreshed_after_painting(world):
    electric = ElectricField(world.settings)
    assert np.allclose(electric.ion_grid(world), 0)
    world.paint(np.array([10]), np.array([7]), Material.CONDUCTOR, resistance=1)
    assert electric.ion_grid(world)[12, 7, 0] > 0  # Ion at (10, 7) pushes right.


def test_neutral_cell_has_weak_far_field(world):
    world.paint(np.array([10]), np.array([7]), Material.CONDUCTOR, resistance=1)
    electric = ElectricField(world.settings)
    far = electric.grid(world)[0, 0]
    lone_ion = coulomb_field(np.array([[0.5, 0.5]]), np.array([[10.5, 7.5]]), np.array([1.0]), 0.8)
    # The electron (within its cell) cancels almost all of the ion's field from afar.
    assert np.hypot(*far) < 0.15 * np.hypot(*lone_ion[0])


def test_place_charges_replaces_and_remove_clears(world):
    cells = (np.array([1, 2, 2]), np.array([1, 1, 1]))  # Duplicate cell in the stroke.
    world.place_charges(*cells, 1)
    assert len(world.fixed_charge) == 2
    world.place_charges(np.array([2]), np.array([1]), -5)
    assert sorted(world.fixed_charge) == [-5, 1]
    world.remove_charges(np.array([1]), np.array([1]))
    assert world.fixed_charge.tolist() == [-5]
    assert np.allclose(world.fixed_pos, [[2.5, 1.5]])


def test_neutral_conductor_with_centred_electrons_has_no_field(world):
    world.paint(np.repeat(np.arange(4, 16), 6), np.tile(np.arange(4, 10), 12),
                Material.CONDUCTOR, resistance=1)
    world.particles.pos = np.argwhere(world.materials.conductive) + 0.5
    electric = ElectricField(world.settings)
    centres = np.argwhere(np.ones((20, 14), dtype=bool)) + 0.5
    assert np.abs(electric.at(world, centres)).max() < 1e-12
    assert np.abs(electric.grid(world)).max() < 1e-12
