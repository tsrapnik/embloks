import pytest

from embloks.grid import Grid


def test_fit_uses_requested_columns_when_they_divide_evenly():
    grid = Grid.fit(density=40, width=1000, height=800)
    assert (grid.cols, grid.rows, grid.cell_px) == (40, 32, 25)
    assert (grid.origin_x, grid.origin_y) == (0, 0)


def test_fit_centers_grid_and_stays_inside_window():
    grid = Grid.fit(density=33, width=1000, height=800)
    assert grid.pixel_width <= 1000 and grid.pixel_height <= 800
    assert grid.origin_x == (1000 - grid.pixel_width) // 2
    assert grid.origin_y == (800 - grid.pixel_height) // 2


@pytest.mark.parametrize("col,row", [(0, 0), (5, 7), (39, 31)])
def test_cell_at_finds_cell_from_any_point_inside_its_rect(col, row):
    grid = Grid.fit(density=40, width=1000, height=800)
    x, y, w, h = grid.cell_rect(col, row)
    assert grid.cell_at(x, y) == (col, row)
    assert grid.cell_at(x + w - 1, y + h - 1) == (col, row)
    assert grid.cell_at(x + w / 2, y + h / 2) == (col, row)


@pytest.mark.parametrize("x,y", [(-1, 10), (10, -1), (1000, 10), (10, 800)])
def test_cell_at_returns_none_outside_grid(x, y):
    grid = Grid.fit(density=40, width=1000, height=800)
    assert grid.cell_at(x, y) is None


def test_cell_at_returns_none_in_centering_margin():
    grid = Grid.fit(density=33, width=1000, height=800)
    assert grid.origin_x > 0
    assert grid.cell_at(grid.origin_x - 1, grid.origin_y) is None
