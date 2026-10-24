import numpy as np
import pytest

from embloks.config import Settings
from embloks.fields import (
    CloudInCell, ElectricField, GridKernel, coulomb_field, deposit, grid_kernel, interpolate,
    ion_charge, spread_over_cells,
)
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
    via_fft = GridKernel(9, 6, softening=0.8, corrected=False).field(charge)
    cells = np.argwhere(np.ones_like(charge, dtype=bool))
    sources = np.argwhere(charge != 0)
    direct = coulomb_field(cells + 0.5, sources + 0.5, charge[charge != 0], 0.8)
    assert np.allclose(via_fft.reshape(-1, 2), direct, atol=1e-10)


def test_assignment_correction_restores_the_short_range_grid_force():
    """Cloud-in-cell weakens the grid force between nearby charges; the optimal
    influence function brings it back towards the softened Coulomb force."""
    rng = np.random.default_rng(2)
    n = 24
    errors = {}
    for corrected in (False, True):
        kernel = GridKernel(n, n, softening=0.8, corrected=corrected)
        error = []
        for _ in range(50):
            a = rng.uniform(10, 14, 2)
            angle = rng.uniform(0, 2 * np.pi)
            b = a + 0.75 * np.array([np.cos(angle), np.sin(angle)])
            field = kernel.field(CloudInCell(a[None], n, n).deposit(np.array([1.0])))
            force = CloudInCell(b[None], n, n).gather(field)[0]
            exact = coulomb_field(b[None], a[None], np.array([1.0]), 0.8)[0]
            error.append(np.linalg.norm(force - exact) / np.linalg.norm(exact))
        errors[corrected] = np.mean(error)
    assert errors[True] < 0.6 * errors[False]


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
    # Test charges sit on cell centres and use the uncorrected kernel: exact everywhere.
    centres = np.argwhere(np.ones((20, 14), dtype=bool)) + 0.5
    s = world.settings
    exact_fixed = coulomb_field(centres, world.fixed_pos, world.fixed_charge, s.test_charge_softening)
    ions = ion_charge(world.materials, s.cell_charge)
    ion_grid = grid_kernel(20, 14, s.conductor_softening).field(ions)
    fixed_grid = grid - electric.electron_grid(world) - ion_grid
    assert np.allclose(fixed_grid.reshape(-1, 2), exact_fixed)
    # The assignment correction changes the ions' field on the cell centres nearby
    # (it is tuned for cloud-in-cell read-back); further out it approaches the direct sum.
    exact_ions = coulomb_field(centres, np.argwhere(ions != 0) + 0.5, ions[ions != 0], s.conductor_softening)
    away = np.abs(centres[:, 1] - 5.5) >= 5
    assert np.allclose(ion_grid.reshape(-1, 2)[away], exact_ions[away], rtol=0.02, atol=1e-3)
    # Electrons are blurred, so compare only cells a few cells away from the wire.
    # Spread over their cells, their blur adds a small quadrupole per cell (~1e-3 here).
    far = np.array([[0, 0], [7, 12], [19, 1], [15, 10]])
    exact = electric.at(world, far + 0.5)
    assert np.allclose(grid[far[:, 0], far[:, 1]], exact, rtol=0.05, atol=3e-3)


def test_static_cache_is_refreshed_after_painting_and_placing_charges(world):
    electric = ElectricField(world.settings)
    assert np.allclose(electric.static_grid(world), 0)
    world.paint(np.array([10]), np.array([7]), Material.CONDUCTOR, resistance=1)
    assert electric.static_grid(world)[12, 7, 0] > 0  # Ion at (10, 7) pushes right.
    world.place_charges(np.array([13]), np.array([7]), 5)
    assert electric.static_grid(world)[12, 7, 0] < 0  # Now the nearby +5 charge dominates.


def test_interpolate_reads_back_grid_values_and_is_bilinear():
    grid = np.zeros((4, 3, 2))
    grid[1, 1] = [2.0, -4.0]
    grid[2, 1] = [4.0, 0.0]
    on_centre = interpolate(grid, np.array([[1.5, 1.5]]))
    halfway = interpolate(grid, np.array([[2.0, 1.5]]))
    assert np.allclose(on_centre, [[2.0, -4.0]])
    assert np.allclose(halfway, [[3.0, -2.0]])


def test_grid_force_on_a_lone_electron_is_zero(world):
    """Same deposit and read-back weights => a particle never pushes itself."""
    world.particles.add(np.array([[7.3, 4.8]]), -1.0, 1.0)
    electric = ElectricField(world.settings)
    own = interpolate(electric.grid(world), world.particles.pos)
    assert np.abs(own).max() < 1e-12


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


def test_neutral_conductor_with_evenly_spread_electrons_has_no_grid_field(world):
    """Ions are spread over their cells, so electrons evenly filling them cancel them."""
    world.settings.carriers_per_cell = 16
    world.paint(np.repeat(np.arange(4, 16), 6), np.tile(np.arange(4, 10), 12),
                Material.CONDUCTOR, resistance=1)
    sub = (np.arange(4) + 0.5) / 4
    pattern = np.stack(np.meshgrid(sub, sub, indexing="ij"), axis=-1).reshape(-1, 2)
    cells = np.argwhere(world.materials.conductive)
    world.particles.pos = (cells[:, None, :] + pattern[None]).reshape(-1, 2)
    assert np.abs(ElectricField(world.settings).grid(world)).max() < 1e-12


def test_spread_over_cells_conserves_charge():
    values = np.zeros((5, 4))
    values[2, 1] = 64.0
    values[0, 3] = 8.0  # Corner cell: the parts beyond the edge stay on it.
    spread = spread_over_cells(values)
    assert spread.sum() == pytest.approx(72.0)
    assert spread[2, 1] == pytest.approx(36.0)  # (3/4)^2
    assert spread[1, 1] == pytest.approx(6.0)  # 3/4 * 1/8
    assert spread[1, 0] == pytest.approx(1.0)  # (1/8)^2
    assert spread[0, 3] == pytest.approx(8.0 * (7 / 8) ** 2)
