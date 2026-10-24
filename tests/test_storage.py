import json

import numpy as np
import pygame
import pytest

from embloks import storage
from embloks.app import App
from embloks.config import Settings
from embloks.materials import Material, Orientation


@pytest.fixture
def app(tmp_path):
    app = App(Settings(save_dir=str(tmp_path)))
    app.world.rng = np.random.default_rng(0)
    yield app
    pygame.quit()


def build_circuit(app):
    w = app.world
    cells = {(c, r) for c in range(5, 20) for r in (5, 12)} | {(c, r) for c in (5, 19) for r in range(5, 13)}
    cols, rows = np.array(sorted(cells)).T
    w.paint(cols, rows, Material.CONDUCTOR, resistance=0.5)
    w.paint(np.array([12]), np.array([12]), Material.BATTERY, emf=3, orientation=Orientation.RIGHT)
    w.place_charges(np.array([30]), np.array([20]), -2)
    for _ in range(60):
        app.step(app.settings.dt)


def snapshot(app):
    w, m, p = app.world, app.world.materials, app.world.particles
    return {
        "kind": m.kind.copy(), "resistance": m.resistance.copy(), "emf": m.emf.copy(),
        "orientation": m.orientation.copy(), "pos": p.pos.copy(), "vel": p.vel.copy(),
        "charge": p.charge.copy(), "mass": p.mass.copy(), "fixed_pos": w.fixed_pos.copy(),
        "fixed_charge": w.fixed_charge.copy(), "current": app.current.copy(),
        "battery_gain": [w.battery_gain.get(s) for s in w.stacks],
        "sim_time": app.sim_time, "sim_speed": app.sim_speed,
    }


def test_round_trip_restores_everything(app, tmp_path):
    build_circuit(app)
    app.layers["S"].visible, app.layers["E"].opacity, app.layers["B"].log_scale = True, 0.5, False
    app.sim_speed = 16
    app.settings.speed_of_light = 3.0
    before = snapshot(app)
    path = tmp_path / "scene.npz"
    storage.save(app, path)

    # Wreck the state, including grid size and a physics setting.
    app.change_density(2.0)
    app.world.clear()
    app.layers["S"].visible = False
    app.settings.speed_of_light = 9.0
    app.sim_speed = 1

    storage.load(app, path)
    after = snapshot(app)
    for key, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(after[key], value, err_msg=key)
        else:
            assert after[key] == value, key
    assert app.settings.speed_of_light == 3.0
    assert app.layers["S"].visible and app.layers["E"].opacity == 0.5 and not app.layers["B"].log_scale
    assert (app.grid.cols, app.grid.rows) == app.world.materials.kind.shape
    assert len(app.world.stacks) == 1
    # The loaded scene keeps running.
    for _ in range(10):
        app.step(app.settings.dt)
    app.render()


def test_loading_resizes_the_grid_to_the_saved_scene(app, tmp_path):
    build_circuit(app)
    shape = app.world.materials.kind.shape
    storage.save(app, tmp_path / "scene.npz")
    app.change_density(0.5)
    assert app.world.materials.kind.shape != shape
    storage.load(app, tmp_path / "scene.npz")
    assert app.world.materials.kind.shape == shape
    assert (app.grid.cols, app.grid.rows) == shape


def press(app, key, mod):
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod))
    app.handle_events()


def test_ctrl_digit_saves_and_shift_digit_loads(app, tmp_path):
    build_circuit(app)
    electrons = len(app.world.particles)
    press(app, pygame.K_3, pygame.KMOD_LCTRL)
    assert (tmp_path / "slot3.npz").exists()
    assert "Saved slot 3" in app.message
    app.world.clear()
    press(app, pygame.K_3, pygame.KMOD_LSHIFT)
    assert len(app.world.particles) == electrons
    assert "Loaded slot 3" in app.message
    # Plain digits still pick tools.
    press(app, pygame.K_2, 0)
    assert app.brush.tool.label == "battery"


def test_empty_or_incompatible_slots_give_a_message(app, tmp_path):
    press(app, pygame.K_5, pygame.KMOD_LSHIFT)
    assert "empty" in app.message
    storage.save(app, tmp_path / "slot6.npz")
    with np.load(tmp_path / "slot6.npz") as data:
        arrays = dict(data)
    meta = json.loads(str(arrays["meta"]))
    meta["format"] = 999
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez(tmp_path / "slot6.npz", **arrays)
    press(app, pygame.K_6, pygame.KMOD_LSHIFT)
    assert "Could not load" in app.message


def test_ac_frequency_and_light_speed_survive_save_and_load(app, tmp_path):
    app.load_scene(6)  # Antenna: AC source, slow light.
    for _ in range(10):
        app.step(app.settings.dt)
    frequency = app.world.materials.frequency.copy()
    storage.save(app, tmp_path / "antenna.npz")
    app.load_scene(0)  # DC, default light speed.
    storage.load(app, tmp_path / "antenna.npz")
    assert app.settings.speed_of_light == 2.0
    np.testing.assert_array_equal(app.world.materials.frequency, frequency)
    assert app.world.stacks[0].is_ac
    for _ in range(10):
        app.step(app.settings.dt)
