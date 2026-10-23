import pygame
import pytest

from embloks.app import App
from embloks.config import Settings


@pytest.fixture
def app():
    app = App(Settings())
    yield app
    pygame.quit()


def press(app: App, key: int) -> None:
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
    app.handle_events()


def test_fixed_timestep_runs_ticks_matching_elapsed_time(app):
    app.advance(app.settings.dt * 3.5)
    assert app.tick_count == 3


def test_slow_frame_is_capped(app):
    app.advance(10.0)
    assert app.tick_count == app.settings.max_ticks_per_frame


def test_space_pauses_physics(app):
    press(app, pygame.K_SPACE)
    assert app.paused
    app.advance(1.0)
    assert app.tick_count == 0


def test_escape_quits(app):
    press(app, pygame.K_ESCAPE)
    assert not app.running


def test_plus_and_minus_change_density_and_reset_scene(app):
    app.advance(app.settings.dt * 2)
    cols = app.grid.cols
    press(app, pygame.K_EQUALS)
    assert app.grid.cols > cols
    assert app.sim_time == 0.0
    press(app, pygame.K_MINUS)
    press(app, pygame.K_MINUS)
    assert app.grid.cols < cols


def test_density_is_clamped(app):
    for _ in range(50):
        press(app, pygame.K_MINUS)
    assert app.settings.grid_density == app.settings.min_grid_density
    for _ in range(50):
        press(app, pygame.K_EQUALS)
    assert app.settings.grid_density == app.settings.max_grid_density
