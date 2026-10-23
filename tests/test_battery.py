import numpy as np
import pytest

from embloks import battery, physics
from embloks.config import Settings
from embloks.fields import ElectricField
from embloks.materials import Material, Materials, Orientation
from embloks.world import World


def paint_battery(m, cells, orientation, emf=1.0):
    cols, rows = np.array(cells).T
    m.paint(cols, rows, Material.BATTERY, emf=emf, orientation=orientation)


def test_single_battery_stack_and_terminals():
    m = Materials(8, 6)
    paint_battery(m, [(3, 2)], Orientation.RIGHT, emf=5)
    [stack] = battery.find_stacks(m, 0.1)
    assert stack.low == (2, 2) and stack.high == (4, 2)
    assert stack.emf == 5 and stack.internal_resistance == pytest.approx(0.1)


def test_pointing_left_or_up_gives_negative_emf_along_axis():
    m = Materials(8, 6)
    paint_battery(m, [(3, 2)], Orientation.LEFT, emf=5)
    paint_battery(m, [(6, 3)], Orientation.UP, emf=2)
    stacks = {s.cells: s for s in battery.find_stacks(m, 0.1)}
    assert stacks[((3, 2),)].emf == -5
    vertical = stacks[((6, 3),)]
    assert (vertical.low, vertical.high, vertical.emf) == ((6, 2), (6, 4), -2)


def test_batteries_in_series_add_and_opposing_ones_subtract():
    m = Materials(10, 6)
    paint_battery(m, [(2, 1), (3, 1), (4, 1)], Orientation.RIGHT, emf=2)
    paint_battery(m, [(2, 3), (3, 3)], Orientation.RIGHT, emf=5)
    paint_battery(m, [(4, 3)], Orientation.LEFT, emf=1)
    stacks = {s.cells[0]: s for s in battery.find_stacks(m, 0.1)}
    assert stacks[(2, 1)].emf == 6 and stacks[(2, 1)].internal_resistance == pytest.approx(0.3)
    assert stacks[(2, 3)].emf == 9 and stacks[(2, 3)].high == (5, 3)


def test_batteries_on_different_axes_are_separate_stacks():
    m = Materials(8, 6)
    paint_battery(m, [(3, 2)], Orientation.RIGHT)
    paint_battery(m, [(4, 2)], Orientation.DOWN)
    assert len(battery.find_stacks(m, 0.1)) == 2


def rectangle(c0, c1, r0, r1):
    cells = {(c, r) for c in range(c0, c1 + 1) for r in (r0, r1)}
    cells |= {(c, r) for c in (c0, c1) for r in range(r0, r1 + 1)}
    return np.array(sorted(cells))


def simulate(world, seconds):
    s = world.settings
    electric = ElectricField(s)
    for _ in range(int(round(seconds / s.dt))):
        battery.pump(world, electric.potential_grid(world), s.dt, s)
        physics.step(world, electric, s.dt, s)
    return electric


def terminal_voltage(world, electric, stack):
    potential = electric.potential_grid(world)
    high = battery.terminal_potential(world, potential, stack.high)
    low = battery.terminal_potential(world, potential, stack.low)
    return high - low


def test_open_circuit_terminal_voltage_reaches_emf():
    w = World(40, 20, Settings(), rng=np.random.default_rng(0))
    w.paint(np.arange(8, 20), np.full(12, 10), Material.CONDUCTOR, resistance=0.5)
    w.paint(np.arange(21, 33), np.full(12, 10), Material.CONDUCTOR, resistance=0.5)
    w.paint(np.array([20]), np.array([10]), Material.BATTERY, emf=3, orientation=Orientation.RIGHT)
    electric = simulate(w, 10)
    [stack] = w.stacks
    voltages = []
    for _ in range(5):  # Average a few samples: single electrons make it jitter.
        electric = simulate(w, 1)
        voltages.append(terminal_voltage(w, electric, stack))
    assert np.mean(voltages) == pytest.approx(3, rel=0.15)
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

    simulate(w, 30)
    top = (np.arange(6, 22), np.full(16, 3))
    samples = []
    for _ in range(20):
        simulate(w, 0.5)
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
    assert not w.pump_current


@pytest.mark.parametrize("charge_to_mass", [0.5, 1.0, 4.0])
def test_ohms_law_does_not_depend_on_charge_to_mass(charge_to_mass):
    """Drift current I = E / R must hold for any electron charge/mass (drag scales with it)."""
    s = Settings(carrier_charge_to_mass=charge_to_mass)
    w = World(30, 20, s, rng=np.random.default_rng(0))
    loop = rectangle(4, 23, 3, 12)
    w.paint(loop[:, 0], loop[:, 1], Material.CONDUCTOR, resistance=0.5)
    w.paint(np.array([14]), np.array([12]), Material.BATTERY, emf=2, orientation=Orientation.RIGHT)
    expected = 2 / ((len(loop) - 1) * 0.5 + s.battery_resistance)
    simulate(w, 30)
    top = (np.arange(6, 22), np.full(16, 3))
    samples = []
    for _ in range(20):
        simulate(w, 0.5)
        samples.append(physics.current_density(w)[top][:, 0].mean())
    assert -np.mean(samples) == pytest.approx(expected, rel=0.25)


def test_reported_current_is_the_charge_actually_moved_not_the_commanded_rate():
    s = Settings(carriers_per_cell=8, cell_charge=4)  # Electrons of -0.5 e.
    w = World(10, 6, s, rng=np.random.default_rng(0))
    w.paint(np.arange(1, 4), np.full(3, 2), Material.CONDUCTOR, resistance=1)
    w.paint(np.arange(5, 8), np.full(3, 2), Material.CONDUCTOR, resistance=1)
    w.paint(np.array([4]), np.array([2]), Material.BATTERY, emf=5, orientation=Orientation.RIGHT)
    [stack] = w.stacks
    # Flat potential: V = 0, so the battery commands (5 - 0) / 0.1 = 50 per time unit,
    # i.e. 50 * dt = 1.67 e this tick: three whole electrons (1.5 e) get moved.
    dt = 1 / 30
    battery.pump(w, np.zeros((10, 6)), dt, s)
    alpha = dt / s.current_smoothing_time
    assert w.pump_current[stack] == pytest.approx(alpha * 1.5 / dt)
    assert w.pump_charge[stack] == pytest.approx(50 * dt - 1.5)
    # At V = EMF the battery commands nothing: no move, and the reading decays.
    potential = np.zeros((10, 6))
    potential[5:, :] = 5.0  # The + terminal side sits 5 above the - side.
    before = w.pump_current[stack]
    battery.pump(w, potential, dt, s)
    assert w.pump_current[stack] == pytest.approx(before * (1 - alpha))


def test_ac_stack_follows_a_sine_and_dc_stays_constant():
    m = Materials(10, 6)
    paint_battery(m, [(3, 2)], Orientation.RIGHT, emf=4)
    m.paint(np.array([6]), np.array([2]), Material.BATTERY, emf=2, orientation=Orientation.LEFT,
            frequency=0.25)
    stacks = {s.cells[0]: s for s in battery.find_stacks(m, 0.1)}
    dc, ac = stacks[(3, 2)], stacks[(6, 2)]
    assert not dc.is_ac and dc.emf_at(0.0) == dc.emf_at(7.3) == 4
    assert ac.is_ac and ac.emf == -2
    assert ac.emf_at(1.0) == pytest.approx(-2)  # sin(2 pi * 0.25 * 1) = 1.
    assert ac.emf_at(2.0) == pytest.approx(0, abs=1e-12)
