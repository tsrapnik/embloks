import numpy as np

from embloks.connectivity import compute_connectivity
from embloks.materials import Material, Materials, Orientation


def paint(m: Materials, cells, material=Material.CONDUCTOR, **kwargs):
    cols, rows = np.array(cells).T
    m.paint(cols, rows, material, resistance=1, **kwargs)


def test_vacuum_has_no_islands():
    c = compute_connectivity(Materials(4, 3))
    assert c.count == 0
    assert (c.labels == -1).all()


def test_edge_neighbours_form_one_island():
    m = Materials(5, 5)
    paint(m, [(1, 1), (2, 1), (2, 2)])
    c = compute_connectivity(m)
    assert c.count == 1
    assert c.labels[1, 1] == c.labels[2, 1] == c.labels[2, 2] == 0


def test_diagonal_contact_does_not_connect():
    m = Materials(5, 5)
    paint(m, [(1, 1), (2, 2)])
    c = compute_connectivity(m)
    assert c.count == 2
    assert c.labels[1, 1] != c.labels[2, 2]


def test_battery_blocks_electrons_on_all_sides():
    m = Materials(5, 5)
    paint(m, [(2, 2)], Material.BATTERY, orientation=Orientation.RIGHT)
    paint(m, [(1, 2), (3, 2), (2, 1)])
    c = compute_connectivity(m)
    assert c.labels[2, 2] == -1
    assert len({c.labels[1, 2], c.labels[3, 2], c.labels[2, 1]}) == 3
    assert not c.open_x[1, 2] and not c.open_x[2, 2] and not c.open_y[2, 1]


def test_open_arrays_have_face_shapes():
    c = compute_connectivity(Materials(6, 4))
    assert c.open_x.shape == (5, 4)
    assert c.open_y.shape == (6, 3)
