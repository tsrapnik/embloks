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
