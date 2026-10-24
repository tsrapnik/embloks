import numpy as np
import pytest

from embloks.config import Settings
from embloks.fields import CloudInCell, spread_over_cells
from embloks.materials import Material
from embloks.maxwell import Maxwell
from embloks.world import World


@pytest.fixture
def world():
    return World(20, 14, Settings(), rng=np.random.default_rng(0))


def test_deposit_conserves_charge_and_is_exact_on_centres():
    rng = np.random.default_rng(3)
    positions = rng.uniform(0, 5, (50, 2))  # Includes points near the edges.
    charges = rng.normal(size=50)
    assert CloudInCell(positions, 5, 5).deposit(charges).sum() == pytest.approx(charges.sum())
    on_centre = CloudInCell(np.array([[2.5, 1.5]]), 5, 5).deposit(np.array([3.0]))
    assert on_centre[2, 1] == 3.0 and on_centre.sum() == 3.0


def test_deposit_of_several_columns_matches_one_at_a_time():
    rng = np.random.default_rng(4)
    cic = CloudInCell(rng.uniform(0, 6, (30, 2)), 6, 4)
    values = rng.normal(size=(30, 2))
    both = cic.deposit(values)
    assert both.shape == (6, 4, 2)
    for k in range(2):
        np.testing.assert_allclose(both[..., k], cic.deposit(values[:, k]))
    assert CloudInCell(np.empty((0, 2)), 6, 4).deposit(np.empty(0)).shape == (6, 4)


def test_gather_reads_back_grid_values_and_is_bilinear():
    grid = np.zeros((4, 3, 2))
    grid[1, 1] = [2.0, -4.0]
    grid[2, 1] = [4.0, 0.0]
    gather = CloudInCell(np.array([[1.5, 1.5], [2.0, 1.5]]), 4, 3).gather(grid)
    assert np.allclose(gather, [[2.0, -4.0], [3.0, -2.0]])


def test_neutral_cell_has_weak_far_field(world):
    world.paint(np.array([10]), np.array([7]), Material.CONDUCTOR, resistance=1)
    far = Maxwell.for_world(world, world.settings).node_fields()[0][0, 0]
    # A lone ion's charge, +cell_charge as a line charge: |E| = 2 k q / r.
    lone_ion = 2 * world.settings.cell_charge / np.hypot(10, 7)
    # The electrons (within its cell) cancel almost all of the ion's field from afar.
    assert np.hypot(*far) < 0.05 * lone_ion


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
    e, _ = Maxwell.for_world(world, world.settings).node_fields()
    assert np.abs(e).max() < 1e-12


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
