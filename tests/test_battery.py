import numpy as np
import pytest

from embloks import battery, physics
from embloks.config import Settings
from embloks.materials import Material, Materials, Orientation
from embloks.world import World
from sim import Sim


def paint_battery(m, cells, orientation, emf=1.0):
    cols, rows = np.array(cells).T
    m.paint(cols, rows, Material.BATTERY, resistance=0.1, emf=emf, orientation=orientation)


def test_single_battery_stack_and_terminals():
    m = Materials(8, 6)
    paint_battery(m, [(3, 2)], Orientation.RIGHT, emf=5)
    [stack] = battery.find_stacks(m)
    assert stack.low == (2, 2) and stack.high == (4, 2)
    assert stack.emf == 5 and stack.internal_resistance == pytest.approx(0.1)


def test_pointing_left_or_up_gives_negative_emf_along_axis():
    m = Materials(8, 6)
    paint_battery(m, [(3, 2)], Orientation.LEFT, emf=5)
    paint_battery(m, [(6, 3)], Orientation.UP, emf=2)
    stacks = {s.cells: s for s in battery.find_stacks(m)}
    assert stacks[((3, 2),)].emf == -5
    vertical = stacks[((6, 3),)]
    assert (vertical.low, vertical.high, vertical.emf) == ((6, 2), (6, 4), -2)


def test_batteries_in_series_add_and_opposing_ones_subtract():
    m = Materials(10, 6)
    paint_battery(m, [(2, 1), (3, 1), (4, 1)], Orientation.RIGHT, emf=2)
    paint_battery(m, [(2, 3), (3, 3)], Orientation.RIGHT, emf=5)
    paint_battery(m, [(4, 3)], Orientation.LEFT, emf=1)
    stacks = {s.cells[0]: s for s in battery.find_stacks(m)}
    assert stacks[(2, 1)].emf == 6 and stacks[(2, 1)].internal_resistance == pytest.approx(0.3)
    assert stacks[(2, 3)].emf == 9 and stacks[(2, 3)].high == (5, 3)


def test_batteries_on_different_axes_are_separate_stacks():
    m = Materials(8, 6)
    paint_battery(m, [(3, 2)], Orientation.RIGHT)
    paint_battery(m, [(4, 2)], Orientation.DOWN)
    assert len(battery.find_stacks(m)) == 2


def rectangle(c0, c1, r0, r1):
    cells = {(c, r) for c in range(c0, c1 + 1) for r in (r0, r1)}
    cells |= {(c, r) for c in (c0, c1) for r in range(r0, r1 + 1)}
    return np.array(sorted(cells))


def simulate(world, seconds):
    return Sim(world).run(seconds)


def terminal_voltage(world, sim, stack):
    potential = sim.potential()
    high = battery.terminal_potential(world, potential, stack.high)
    low = battery.terminal_potential(world, potential, stack.low)
    return high - low


def test_open_circuit_terminal_voltage_reaches_emf():
    w = World(40, 20, Settings(), rng=np.random.default_rng(0))
    w.paint(np.arange(8, 20), np.full(12, 10), Material.CONDUCTOR, resistance=0.5)
    w.paint(np.arange(21, 33), np.full(12, 10), Material.CONDUCTOR, resistance=0.5)
    w.paint(np.array([20]), np.array([10]), Material.BATTERY, emf=3, orientation=Orientation.RIGHT)
    sim = Sim(w).run(15)
    [stack] = w.stacks
    voltages = []
    for _ in range(int(5 / w.settings.dt)):
        sim.step()
        voltages.append(terminal_voltage(w, sim, stack))
    assert np.mean(voltages) == pytest.approx(3, rel=0.03)
    # Electrons move smoothly, so the voltage holds steady instead of dithering.
    assert np.std(voltages) < 0.1
    # The + wire lost electrons to the - wire.
    c, _ = w.particles.cells()
    assert np.count_nonzero(c > 20) < np.count_nonzero(c < 20)


@pytest.mark.parametrize("orientation,sign", [(Orientation.RIGHT, 1), (Orientation.LEFT, -1)])
def test_loop_current_follows_ohms_law_and_polarity(orientation, sign):
    s = Settings()
    w = World(30, 20, s, rng=np.random.default_rng(0))
    loop = rectangle(4, 23, 3, 12)
    resistance = 0.5
    w.paint(loop[:, 0], loop[:, 1], Material.CONDUCTOR, resistance=resistance)
    w.paint(np.array([14]), np.array([12]), Material.BATTERY, emf=2, orientation=orientation)
    wire_cells = len(loop) - 1
    expected = 2 / (wire_cells * resistance + s.battery_resistance)

    sim = Sim(w).run(30)
    top = (np.arange(6, 22), np.full(16, 3))
    samples = []
    for _ in range(20):
        sim.run(0.5)
        samples.append(physics.current_density(w)[top][:, 0].mean())
    # Battery + on the right of the bottom edge pushes conventional current out of its
    # right terminal, up the right side and leftwards (-x) along the top.
    assert -sign * np.mean(samples) == pytest.approx(expected, rel=0.25)


def test_unconnected_battery_does_nothing():
    w = World(10, 6, Settings(), rng=np.random.default_rng(0))
    w.paint(np.arange(1, 4), np.full(3, 2), Material.CONDUCTOR, resistance=1)
    w.paint(np.array([4]), np.array([2]), Material.BATTERY, emf=5, orientation=Orientation.RIGHT)
    before = w.particles.pos.copy()
    simulate(w, 1)
    assert len(w.particles) == len(before)
    assert not w.battery_gain


@pytest.mark.parametrize("charge_to_mass", [0.5, 1.0, 4.0])
def test_ohms_law_does_not_depend_on_charge_to_mass(charge_to_mass):
    """Drift current I = E / R must hold for any electron charge/mass (drag scales with it)."""
    s = Settings(carrier_charge_to_mass=charge_to_mass)
    w = World(30, 20, s, rng=np.random.default_rng(0))
    loop = rectangle(4, 23, 3, 12)
    w.paint(loop[:, 0], loop[:, 1], Material.CONDUCTOR, resistance=0.5)
    w.paint(np.array([14]), np.array([12]), Material.BATTERY, emf=2, orientation=Orientation.RIGHT)
    expected = 2 / ((len(loop) - 1) * 0.5 + s.battery_resistance)
    sim = Sim(w).run(30)
    top = (np.arange(6, 22), np.full(16, 3))
    samples = []
    for _ in range(20):
        sim.run(0.5)
        samples.append(physics.current_density(w)[top][:, 0].mean())
    assert -np.mean(samples) == pytest.approx(expected, rel=0.25)


def test_impressed_field_pushes_electrons_to_the_minus_side():
    w = World(10, 6, Settings(), rng=np.random.default_rng(0))
    w.paint(np.array([3, 4]), np.array([2, 2]), Material.BATTERY, emf=2, orientation=Orientation.LEFT,
            frequency=0.25)
    w.paint(np.array([6]), np.array([4]), Material.BATTERY, emf=5, orientation=Orientation.DOWN)
    cols, rows = np.array([3, 6, 1]), np.array([2, 4, 1])
    # Fields point to each cell's + side; vacuum gets none. sin(2 pi * 0.25 * 1) = 1.
    field = battery.impressed_field(w, cols, rows, time=1.0)
    np.testing.assert_allclose(field, [[-2, 0], [0, 5], [0, 0]], atol=1e-12)
    np.testing.assert_allclose(battery.impressed_field(w, cols[:1], rows[:1], time=2.0), 0, atol=1e-12)
    [stack] = [st for st in w.stacks if (3, 2) in st.cells]
    w.battery_gain[stack] = 1.5
    np.testing.assert_allclose(battery.impressed_field(w, cols[:1], rows[:1], time=1.0), [[-3, 0]])


def test_battery_electrons_drift_instead_of_jumping():
    s = Settings()
    w = World(30, 20, s, rng=np.random.default_rng(0))
    loop = rectangle(4, 23, 3, 12)
    w.paint(loop[:, 0], loop[:, 1], Material.CONDUCTOR, resistance=0.5)
    w.paint(np.array([14]), np.array([12]), Material.BATTERY, emf=10, orientation=Orientation.RIGHT)
    sim = Sim(w)
    for _ in range(int(5 / s.dt)):
        before = w.particles.pos.copy()
        substeps = sim.step()
        assert np.abs(w.particles.pos - before).max() <= substeps * s.max_step + 1e-9


def test_trim_corrects_a_shortfall_and_follows_the_load():
    s = Settings()
    w = World(10, 6, s)
    w.paint(np.arange(1, 4), np.full(3, 2), Material.CONDUCTOR, resistance=1)
    w.paint(np.arange(5, 8), np.full(3, 2), Material.CONDUCTOR, resistance=1)
    w.paint(np.array([4]), np.array([2]), Material.BATTERY, emf=4, orientation=Orientation.RIGHT)
    [stack] = w.stacks
    potential = np.zeros((10, 6))
    potential[5:] = 3.0  # Delivers 3 of 4: the gain goes up.
    current = np.zeros((10, 6, 2))
    battery.trim(w, potential, current, 0.1, s)
    assert w.battery_gain[stack] == pytest.approx(1 + 0.1 / s.battery_trim_time * 1 / 4)
    # Under load, V = EMF - I R is exactly right: nothing to correct.
    w.battery_gain[stack] = 1.2
    current[3:6, 2, 0] = 10.0
    battery.trim(w, potential, current, 0.1, s)
    assert w.battery_gain[stack] == pytest.approx(1.2)


def test_ac_stack_follows_a_sine_and_dc_stays_constant():
    m = Materials(10, 6)
    paint_battery(m, [(3, 2)], Orientation.RIGHT, emf=4)
    m.paint(np.array([6]), np.array([2]), Material.BATTERY, resistance=0.1, emf=2,
            orientation=Orientation.LEFT, frequency=0.25)
    stacks = {s.cells[0]: s for s in battery.find_stacks(m)}
    dc, ac = stacks[(3, 2)], stacks[(6, 2)]
    assert not dc.is_ac and dc.emf_at(0.0) == dc.emf_at(7.3) == 4
    assert ac.is_ac and ac.emf == -2
    assert ac.emf_at(1.0) == pytest.approx(-2)  # sin(2 pi * 0.25 * 1) = 1.
    assert ac.emf_at(2.0) == pytest.approx(0, abs=1e-12)


def test_parallel_batteries_hold_steady():
    """Two stacks feeding the same terminals must not fight over them."""
    from embloks import scenes

    w = World(40, 32, Settings(), rng=np.random.default_rng(0))
    scenes.parallel_paths(w)
    sim = Sim(w).run(5)
    assert np.hypot(*w.particles.vel.T).max() < 2
    assert all(0.5 < gain < 2 for gain in w.battery_gain.values())


def test_ac_battery_voltage_follows_its_sine():
    s = Settings()
    w = World(30, 20, s, rng=np.random.default_rng(0))
    loop = rectangle(4, 23, 3, 12)
    w.paint(loop[:, 0], loop[:, 1], Material.CONDUCTOR, resistance=0.5)
    w.paint(np.array([14]), np.array([12]), Material.BATTERY, emf=5, orientation=Orientation.RIGHT,
            frequency=0.05)
    sim = Sim(w).run(10)
    [stack] = w.stacks
    voltages, emfs = [], []
    for _ in range(int(20 / s.dt)):
        sim.step()
        voltages.append(terminal_voltage(w, sim, stack))
        emfs.append(stack.emf_at(sim.time))
    voltages, emfs = np.array(voltages), np.array(emfs)
    assert (voltages @ emfs) / (emfs @ emfs) == pytest.approx(1, abs=0.05)
    assert np.corrcoef(voltages, emfs)[0, 1] > 0.99


def test_a_steady_circuit_leaves_the_field_around_it_calm():
    """Regression: batteries that moved electrons by jumps shook E in the empty space
    around them every tick (tick-to-tick rms ~0.015 in this scene)."""
    from embloks import scenes

    s = Settings()
    w = World(40, 30, s, rng=np.random.default_rng(0))
    scenes.battery_and_resistor(w)
    sim = Sim(w).run(20)
    vacuum = w.materials.kind == Material.VACUUM
    fields = []
    for _ in range(int(5 / s.dt)):
        sim.step()
        fields.append(sim.e_field()[vacuum])
    assert np.sqrt((np.diff(fields, axis=0) ** 2).mean()) < 0.005
