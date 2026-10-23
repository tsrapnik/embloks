import pygame
import pytest

from embloks.app import App
from embloks.config import Settings
from embloks.materials import Material, Orientation


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


def mouse(app: App, kind: int, cell: tuple[int, int], **kwargs) -> None:
    x, y, w, h = app.grid.cell_rect(*cell)
    pygame.event.post(pygame.event.Event(kind, pos=(x + w // 2, y + h // 2), **kwargs))
    app.handle_events()


def test_left_drag_paints_a_connected_wire_with_current_tool(app):
    mouse(app, pygame.MOUSEBUTTONDOWN, (2, 2), button=1)
    mouse(app, pygame.MOUSEMOTION, (8, 6), buttons=(1, 0, 0))
    mouse(app, pygame.MOUSEBUTTONUP, (8, 6), button=1)
    painted = app.materials.kind == Material.CONDUCTOR
    # A 4-connected path from (2,2) to (8,6) has 6 + 4 + 1 cells.
    assert painted.sum() == 11
    assert painted[2, 2] and painted[8, 6]


def test_motion_without_button_does_not_paint(app):
    mouse(app, pygame.MOUSEMOTION, (3, 3), buttons=(0, 0, 0))
    assert (app.materials.kind == Material.VACUUM).all()


def test_right_click_erases(app):
    mouse(app, pygame.MOUSEBUTTONDOWN, (4, 4), button=1)
    mouse(app, pygame.MOUSEBUTTONUP, (4, 4), button=1)
    mouse(app, pygame.MOUSEBUTTONDOWN, (4, 4), button=3)
    mouse(app, pygame.MOUSEBUTTONUP, (4, 4), button=3)
    assert app.materials.kind[4, 4] == Material.VACUUM


def test_battery_tool_paints_with_brush_orientation(app):
    press(app, pygame.K_3)
    press(app, pygame.K_r)
    mouse(app, pygame.MOUSEBUTTONDOWN, (1, 1), button=1)
    assert app.materials.kind[1, 1] == Material.BATTERY
    assert app.materials.orientation[1, 1] == Orientation.DOWN
    assert app.materials.emf[1, 1] == app.settings.default_emf


def test_render_runs_with_all_materials(app):
    for key, cell in ((pygame.K_2, (1, 1)), (pygame.K_3, (2, 1))):
        press(app, key)
        mouse(app, pygame.MOUSEBUTTONDOWN, cell, button=1)
        mouse(app, pygame.MOUSEBUTTONUP, cell, button=1)
    app.render()
