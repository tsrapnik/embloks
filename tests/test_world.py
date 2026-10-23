import numpy as np
import pytest

from embloks.config import Settings
from embloks.materials import Material, Orientation
from embloks.world import World


def make_world(n=1, cols=12, rows=8):
    return World(cols, rows, Settings(carriers_per_cell=n), rng=np.random.default_rng(0))


def paint(world, cells, material=Material.CONDUCTOR, **kwargs):
    cols, rows = np.array(cells).reshape(-1, 2).T
    world.paint(cols, rows, material, resistance=1, **kwargs)


def row_of(c0, c1, r):
    return [(c, r) for c in range(c0, c1)]


def electrons_per_cell(world):
    counts = np.zeros(world.materials.kind.shape, dtype=int)
    np.add.at(counts, world.particles.cells(), 1)
    return counts


def assert_electrons_inside_conductors(world):
    cell_c, cell_r = world.particles.cells()
    assert world.materials.conductive[cell_c, cell_r].all()


@pytest.mark.parametrize("n", [1, 4])
def test_painting_spawns_n_electrons_per_cell_inside_it(n):
    world = make_world(n)
    paint(world, row_of(2, 7, 3))
    assert len(world.particles) == 5 * n
    assert (electrons_per_cell(world)[2:7, 3] == n).all()
    assert np.allclose(world.particles.charge, -1 / n)
    assert np.allclose(world.particles.charge / world.particles.mass, -1)


def test_repainting_conductive_cells_keeps_their_electrons():
    world = make_world()
    paint(world, row_of(2, 7, 3))
    before = world.particles.pos.copy()
    paint(world, row_of(2, 7, 3), Material.BATTERY, orientation=Orientation.RIGHT)
    assert np.array_equal(world.particles.pos, before)


def test_erasing_removes_electrons_of_erased_cells():
    world = make_world(n=2)
    paint(world, row_of(2, 7, 3))
    world.paint(np.array([4]), np.array([3]), Material.VACUUM)
    assert len(world.particles) == 8
    assert_electrons_inside_conductors(world)


def test_erasing_an_empty_cell_takes_nearest_electron_from_its_island():
    world = make_world()
    paint(world, row_of(2, 7, 3))
    # Pile all electrons into the leftmost cell, leaving cell 6 empty.
    world.particles.pos[:] = [2.5, 3.5]
    world.paint(np.array([6]), np.array([3]), Material.VACUUM)
    assert len(world.particles) == 4


def test_extra_electrons_in_erased_cell_move_to_nearest_cell_of_same_island():
    world = make_world()
    paint(world, row_of(2, 7, 3))
    paint(world, [(10, 6)])  # A separate island far away.
    world.particles.pos[:5] = [6.5, 3.5]  # Whole row's electrons crowd the end cell.
    world.paint(np.array([6]), np.array([3]), Material.VACUUM)
    assert len(world.particles) == 5
    counts = electrons_per_cell(world)
    assert counts[5, 3] == 4  # Leftovers landed in the nearest remaining cell...
    assert counts[10, 6] == 1  # ...not in the other island.
    assert_electrons_inside_conductors(world)


def test_erasing_whole_island_removes_all_its_electrons_even_if_charged():
    world = make_world()
    paint(world, row_of(2, 4, 3))
    paint(world, [(9, 6)])
    world.particles.pos[2] = [2.5, 3.5]  # Move the other island's electron in (net -1).
    world.paint(np.array([2, 3]), np.array([3, 3]), Material.VACUUM)
    assert len(world.particles) == 0


def test_clear_removes_everything():
    world = make_world()
    paint(world, row_of(2, 7, 3))
    world.clear()
    assert len(world.particles) == 0
    assert world.connectivity.count == 0
