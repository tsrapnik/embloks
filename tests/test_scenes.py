import numpy as np
import pygame
import pytest

from embloks import battery, physics
from embloks.app import App
from embloks.config import Settings
from embloks.fields import poynting
from embloks.materials import Material
from embloks.scenes import SCENES


@pytest.fixture
def app(tmp_path):
    app = App(Settings(save_dir=str(tmp_path)))
    yield app
    pygame.quit()


def press(app, key):
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0))
    app.handle_events()


def run(app, time_units):
    for _ in range(int(round(time_units / app.settings.dt))):
        app.step(app.settings.dt)


@pytest.mark.parametrize("index", range(len(SCENES)))
def test_every_scene_loads_from_its_f_key_and_runs(app, index):
    press(app, getattr(pygame, f"K_F{index + 1}"))
    assert SCENES[index].title in app.message
    assert len(app.world.particles) > 0
    run(app, 1)
    app.hovered_cell = (20, 15)
    app.render()


def test_loading_a_scene_resets_settings_and_layers(app):
    app.settings.speed_of_light = 9.0
    app.layers["E"].opacity = 0.1
    app.layers["S"].visible = True
    app.load_scene(1)  # Screening: default layers except B hidden.
    assert app.settings.speed_of_light == Settings().speed_of_light
    assert app.layers["E"].opacity == 1.0 and not app.layers["S"].visible
    assert not app.layers["B"].visible


def test_textbook_circuit_sends_energy_into_the_resistor(app):
    app.load_scene(0)
    app.world.rng = np.random.default_rng(0)
    m = app.world.materials
    resistor = (m.kind == Material.CONDUCTOR) & (m.resistance == 5)
    col = np.nonzero(resistor.any(axis=1))[0][0]
    rows = np.nonzero(resistor[col])[0]
    run(app, 40)
    flow, current = np.zeros((m.cols, m.rows, 2)), []
    for i in range(int(20 / app.settings.dt)):
        app.step(app.settings.dt)
        if i % 3 == 0:
            b = app.magnetic.grid(app.world, app.current + battery.current_grid(app.world))
            flow += poynting(app.electric.grid(app.world), b, app.settings)
        current.append(physics.current_density(app.world)[13:27, 7, 0].mean())
    # Time-averaged S points into the resistor from outside and from inside the loop.
    assert (flow[col + 1, rows, 0] < 0).all()
    assert (flow[col - 1, rows, 0] > 0).all()
    # Current around the loop follows Ohm's law (thin wires cut corners: allow 25%).
    wire = m.resistance[(m.kind == Material.CONDUCTOR) & ~resistor].sum()
    total = wire + m.resistance[resistor].sum() + app.settings.battery_resistance
    [stack] = app.world.stacks
    assert abs(np.mean(current)) == pytest.approx(abs(stack.emf) / total, rel=0.25)


def test_series_scene_drives_three_times_the_current(app):
    app.load_scene(2)
    app.world.rng = np.random.default_rng(0)
    run(app, 30)
    left, right = [], []
    for _ in range(int(30 / app.settings.dt)):
        app.step(app.settings.dt)
        current = physics.current_density(app.world)
        left.append(current[5:16, 8, 0].mean())
        right.append(current[24:35, 8, 0].mean())
    assert np.mean(right) / np.mean(left) == pytest.approx(3, rel=0.2)


def test_magnetic_field_next_to_the_battery_is_steady(app):
    """Regression for the jitter next to the battery (was ~0.6, now ~0.16)."""
    app.load_scene(0)
    app.world.rng = np.random.default_rng(0)
    run(app, 40)
    samples = []
    for i in range(int(20 / app.settings.dt)):
        app.step(app.settings.dt)
        if i % 4 == 0:  # One displayed frame at 8x speed.
            b = app.magnetic.grid(app.world, app.current + battery.current_grid(app.world))
            samples.append(b[11:13, 13:18].ravel())
    samples = np.array(samples)
    flicker = samples.std(axis=0).mean() / np.abs(samples.mean(axis=0)).mean()
    assert flicker < 0.3


def test_capacitor_plates_reach_the_emf_with_a_uniform_field_between(app):
    app.load_scene(3)
    app.world.rng = np.random.default_rng(0)
    run(app, 60)
    potential = app.electric.potential_grid(app.world)
    left_plate, right_plate = potential[16, 10:22].mean(), potential[22, 10:22].mean()
    [stack] = app.world.stacks
    assert right_plate - left_plate == pytest.approx(abs(stack.emf), rel=0.1)
    # Interior of the gap: away from the plates (charges are smeared ~1 cell) and edges.
    gap = app.electric.grid(app.world)[18:21, 12:20]
    assert (gap[..., 0] < 0).all()  # From the + plate (right) to the - plate (left).
    strength = np.hypot(gap[..., 0], gap[..., 1])
    assert strength.std() / strength.mean() < 0.1  # Nearly uniform.


def test_parallel_paths_split_the_current_by_resistance(app):
    app.load_scene(4)
    app.world.rng = np.random.default_rng(0)
    run(app, 50)
    # Average over every cross-section along each resistor (same current in steady
    # state) and over time: few discrete electrons pass any one line (shot noise).
    thin_a, thin_b, wide = [], [], []
    for _ in range(int(40 / app.settings.dt)):
        app.step(app.settings.dt)
        for col in range(13, 27):
            thin_a.append(app.wire_current(col, 10)[0])
            thin_b.append(app.wire_current(col, 17)[0])
            wide.append(app.wire_current(col, 22)[0])
    thin_a, thin_b, wide = (abs(np.mean(x)) for x in (thin_a, thin_b, wide))
    assert thin_a == pytest.approx(thin_b, rel=0.15)
    assert wide / ((thin_a + thin_b) / 2) == pytest.approx(2, rel=0.2)


def sine_fit(t, x, frequency):
    """Amplitude and phase (degrees) of the component of x at `frequency`."""
    basis = np.stack([np.sin(2 * np.pi * frequency * t), np.cos(2 * np.pi * frequency * t),
                      np.ones_like(t)], axis=1)
    (s, c, _), *_ = np.linalg.lstsq(basis, x, rcond=None)
    return np.hypot(s, c), np.degrees(np.arctan2(c, s))


def record(app, seconds, probes):
    t, data = [], {name: [] for name in probes}
    for _ in range(int(seconds / app.settings.dt)):
        app.step(app.settings.dt)
        t.append(app.sim_time)
        current = physics.current_density(app.world)
        for name, probe in probes.items():
            data[name].append(probe(current))
    return np.array(t), {name: np.array(v) for name, v in data.items()}


def test_transformer_drives_the_inner_loop_by_induction_only(app):
    probes = {
        # Primary: mean tangential current over three sides (one loop current).
        "primary": lambda j: (j[10:30, 4, 0].mean() + j[35, 8:24, 1].mean() - j[10:30, 27, 0].mean()) / 3,
        "load": lambda j: j[27, 13:19, 1].mean(),  # Through the inner loop's resistor.
    }
    results = {}
    for dynamic in (True, False):
        app.load_scene(5)
        app.world.rng = np.random.default_rng(0)
        if not dynamic:
            app.toggle_dynamic_fields()
        run(app, 60)
        t, data = record(app, 100, probes)
        results[dynamic] = {name: sine_fit(t, x, 0.02) for name, x in data.items()}
    load_dynamic, load_instant = results[True]["load"][0], results[False]["load"][0]
    assert load_dynamic > 5 * load_instant  # No induction, no current through the load.
    # Faraday + Lenz with a resistive load: I_load ~ -dI_primary/dt, i.e. 90 degrees behind.
    lag = (results[True]["load"][1] - results[True]["primary"][1] + 180) % 360 - 180
    assert lag == pytest.approx(-90, abs=40)


def test_antenna_radiates_waves_that_the_receiver_picks_up(app):
    frequency, columns = 0.1, list(range(12, 27, 2))
    results = {}
    for dynamic in (True, False):
        app.load_scene(6)
        app.world.rng = np.random.default_rng(0)
        if not dynamic:
            app.toggle_dynamic_fields()
        run(app, 30)
        t, fields, receiver = [], {x: [] for x in columns}, []
        for _ in range(int(30 / app.settings.dt)):
            app.step(app.settings.dt)
            t.append(app.sim_time)
            e = app.display_fields()[0] if dynamic else app.electric.grid(app.world)
            for x in columns:
                fields[x].append(e[x, 16, 1])
            receiver.append(physics.current_density(app.world)[29, 15:18, 1].mean())
        t = np.array(t)
        phases = np.unwrap(np.radians([sine_fit(t, np.array(fields[x]), frequency)[1] for x in columns]))
        results[dynamic] = (np.polyfit(columns, phases, 1)[0], sine_fit(t, np.array(receiver), frequency)[0])
    wave = 2 * np.pi * frequency / app.settings.speed_of_light
    # A travelling wave's phase falls with distance; an instant field's does not.
    assert results[True][0] == pytest.approx(-wave, rel=0.35)
    assert abs(results[False][0]) < 0.05
    assert results[True][1] > 3 * results[False][1]
