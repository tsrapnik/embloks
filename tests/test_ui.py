import pytest

from embloks.config import Settings
from embloks.grid import Grid
from embloks.materials import Material
from embloks.ui import Brush, line_cells


@pytest.fixture
def grid():
    return Grid(cols=20, rows=10, cell_px=10, origin_x=0, origin_y=0)


def cells_of(footprint):
    return set(zip(footprint[0].tolist(), footprint[1].tolist()))


def test_radius_zero_paints_one_cell(grid):
    brush = Brush(Settings())
    assert cells_of(brush.footprint((5, 5), grid)) == {(5, 5)}


def test_radius_one_paints_3x3_block(grid):
    brush = Brush(Settings(), radius=1)
    expected = {(c, r) for c in (4, 5, 6) for r in (4, 5, 6)}
    assert cells_of(brush.footprint((5, 5), grid)) == expected


def test_footprint_is_clipped_to_grid(grid):
    brush = Brush(Settings(), radius=2)
    cells = cells_of(brush.footprint((0, 0), grid))
    assert (0, 0) in cells
    assert all(0 <= c < grid.cols and 0 <= r < grid.rows for c, r in cells)


def test_resize_is_clamped():
    s = Settings()
    brush = Brush(s)
    brush.resize(-5)
    assert brush.radius == 0
    brush.resize(1000)
    assert brush.radius == s.max_brush_radius


def test_adjust_value_steps_the_current_tools_ladder():
    s = Settings()
    brush = Brush(s, tool=Material.CONDUCTOR)
    assert brush.resistance == s.default_resistance
    brush.adjust_value(1)
    assert brush.resistance > s.default_resistance
    assert brush.emf == s.default_emf
    brush.tool = Material.BATTERY
    brush.adjust_value(-100)
    assert brush.emf == s.emf_values[0]


@pytest.mark.parametrize(
    "start,end",
    [((0, 0), (0, 0)), ((0, 0), (7, 0)), ((3, 9), (3, 1)), ((0, 0), (5, 5)), ((9, 2), (1, 6))],
)
def test_line_cells_is_4_connected_and_hits_both_ends(start, end):
    cells = line_cells(start, end)
    assert cells[0] == start and cells[-1] == end
    for (x0, y0), (x1, y1) in zip(cells, cells[1:]):
        # Neighbours share an edge: exactly one coordinate changes by one.
        assert abs(x1 - x0) + abs(y1 - y0) == 1


def test_line_cells_stays_close_to_the_ideal_line():
    cells = line_cells((0, 0), (10, 4))
    for x, y in cells:
        assert abs(y - x * 4 / 10) <= 1
