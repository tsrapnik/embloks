import numpy as np

from embloks.materials import Material, Materials, Orientation


def test_new_materials_are_all_vacuum():
    m = Materials(5, 4)
    assert m.kind.shape == (5, 4)
    assert (m.kind == Material.VACUUM).all()


def test_paint_conductor_sets_resistance_only():
    m = Materials(5, 4)
    m.paint(np.array([1]), np.array([2]), Material.CONDUCTOR, resistance=2, emf=9)
    assert m.kind[1, 2] == Material.CONDUCTOR
    assert m.resistance[1, 2] == 2
    assert m.emf[1, 2] == 0


def test_paint_battery_sets_emf_and_orientation():
    m = Materials(5, 4)
    m.paint(np.array([3]), np.array([0]), Material.BATTERY, resistance=2, emf=7,
            orientation=Orientation.UP)
    assert m.kind[3, 0] == Material.BATTERY
    assert (m.emf[3, 0], m.orientation[3, 0], m.resistance[3, 0]) == (7, Orientation.UP, 2)
    assert m.conductive[3, 0]


def test_painting_vacuum_clears_properties():
    m = Materials(5, 4)
    cells = (np.array([0, 1]), np.array([0, 0]))
    m.paint(*cells, Material.BATTERY, emf=7, orientation=Orientation.LEFT)
    m.paint(*cells, Material.VACUUM)
    assert (m.kind == 0).all() and (m.emf == 0).all() and (m.orientation == 0).all()


def test_paint_and_clear_bump_version():
    m = Materials(5, 4)
    v = m.version
    m.paint(np.array([0]), np.array([0]), Material.CONDUCTOR, resistance=1)
    assert m.version > v
    v = m.version
    m.clear()
    assert m.version > v
    assert (m.kind == Material.VACUUM).all()


def test_orientation_rotates_clockwise_on_screen():
    o = Orientation.RIGHT
    seen = []
    for _ in range(4):
        seen.append(o.vector)
        o = o.rotated()
    # Screen y points down, so right -> down -> left -> up is clockwise.
    assert seen == [(1, 0), (0, 1), (-1, 0), (0, -1)]
    assert o == Orientation.RIGHT
