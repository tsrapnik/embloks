import numpy as np
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


def press(app: App, key: int, mod: int = 0) -> None:
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod))
    app.handle_events()


def test_fixed_timestep_runs_ticks_matching_elapsed_time(app):
    app.sim_speed = 1
    app.advance(app.settings.dt * 3.5)
    assert app.tick_count == 3


def test_speed_multiplies_simulated_time_per_real_second(app):
    app.sim_speed = 4
    app.advance(app.settings.dt * 3.5)
    assert app.tick_count == 14


def test_comma_and_period_step_through_speeds(app):
    speeds = app.settings.sim_speeds
    for _ in range(20):
        press(app, pygame.K_PERIOD)
    assert app.sim_speed == speeds[-1]
    press(app, pygame.K_COMMA)
    assert app.sim_speed == speeds[-2]
    for _ in range(20):
        press(app, pygame.K_COMMA)
    assert app.sim_speed == speeds[0]


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
    painted = app.world.materials.kind == Material.CONDUCTOR
    # A 4-connected path from (2,2) to (8,6) has 6 + 4 + 1 cells.
    assert painted.sum() == 11
    assert painted[2, 2] and painted[8, 6]


def test_motion_without_button_does_not_paint(app):
    mouse(app, pygame.MOUSEMOTION, (3, 3), buttons=(0, 0, 0))
    assert (app.world.materials.kind == Material.VACUUM).all()


def test_right_click_erases(app):
    mouse(app, pygame.MOUSEBUTTONDOWN, (4, 4), button=1)
    mouse(app, pygame.MOUSEBUTTONUP, (4, 4), button=1)
    mouse(app, pygame.MOUSEBUTTONDOWN, (4, 4), button=3)
    mouse(app, pygame.MOUSEBUTTONUP, (4, 4), button=3)
    assert app.world.materials.kind[4, 4] == Material.VACUUM


def test_battery_tool_paints_with_brush_orientation(app):
    press(app, pygame.K_2)
    press(app, pygame.K_r)
    mouse(app, pygame.MOUSEBUTTONDOWN, (1, 1), button=1)
    assert app.world.materials.kind[1, 1] == Material.BATTERY
    assert app.world.materials.orientation[1, 1] == Orientation.DOWN
    assert app.world.materials.emf[1, 1] == app.settings.default_emf


def click(app, cell, button):
    mouse(app, pygame.MOUSEBUTTONDOWN, cell, button=button)
    mouse(app, pygame.MOUSEBUTTONUP, cell, button=button)


def test_render_runs_with_info_for_every_kind_of_cell(app):
    for key, cell in ((pygame.K_1, (1, 1)), (pygame.K_2, (2, 1)), (pygame.K_3, (5, 5))):
        press(app, key)
        click(app, cell, 1)
    for cell in ((1, 1), (2, 1), (5, 5), (9, 9)):
        app.hovered_cell = cell
        app.render()


def test_right_click_removes_test_charge_first_then_material(app):
    click(app, (3, 3), 1)  # Conductor (default tool).
    press(app, pygame.K_3)
    press(app, pygame.K_DOWN)
    click(app, (3, 3), 1)
    assert app.world.fixed_charge.tolist() == [app.brush.charge]
    assert app.brush.charge < app.settings.default_test_charge
    click(app, (3, 3), 3)
    assert len(app.world.fixed_charge) == 0
    assert app.world.materials.kind[3, 3] == Material.CONDUCTOR
    click(app, (3, 3), 3)
    assert app.world.materials.kind[3, 3] == Material.VACUUM


def test_erase_drag_starting_on_a_charge_never_erases_material(app):
    for col in range(2, 8):
        click(app, (col, 4), 1)
    press(app, pygame.K_3)
    click(app, (2, 4), 1)
    click(app, (6, 4), 1)
    mouse(app, pygame.MOUSEBUTTONDOWN, (2, 4), button=3)
    mouse(app, pygame.MOUSEMOTION, (7, 4), buttons=(0, 0, 1))
    mouse(app, pygame.MOUSEMOTION, (2, 4), buttons=(0, 0, 1))
    mouse(app, pygame.MOUSEBUTTONUP, (2, 4), button=3)
    assert len(app.world.fixed_charge) == 0
    assert (app.world.materials.kind[2:8, 4] == Material.CONDUCTOR).all()


def test_h_cycles_info_panel(app):
    seen = [app.hud_mode]
    for _ in range(3):
        press(app, pygame.K_h)
        seen.append(app.hud_mode)
        app.render()
    assert seen == ["full", "status", "hidden", "full"]


def test_s_toggles_poynting_arrows_and_render_shows_them(app):
    assert not app.layers["S"].visible
    press(app, pygame.K_s)
    assert app.layers["S"].visible
    click(app, (3, 3), 1)
    app.hovered_cell = (3, 3)
    app.render()


def test_clicking_the_layer_panel_does_not_paint(app):
    app.render()
    rect = app.layer_panel.button_rects["E field:show"]
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=rect.center, button=1))
    app.handle_events()
    assert not app.layers["E"].visible
    assert (app.world.materials.kind == Material.VACUUM).all()


def test_l_hides_layer_panel(app):
    press(app, pygame.K_l)
    app.render()
    assert not app.layer_panel.visible and app.layer_panel.rect is None


def test_displayed_e_field_is_smoothed_but_follows_edits_immediately(app):
    click(app, (10, 10), 1)
    app.render()
    app.paused = True
    # Painting while paused (no simulated time passes) must still show up at once.
    press(app, pygame.K_3)
    click(app, (14, 10), 1)
    np.testing.assert_allclose(app.display_e_field(), app.electric.grid(app.world))
    # While running, the display lags behind a sudden change in the electrons.
    app.paused = False
    app.world.particles.pos[:] += 0.3
    app.sim_time += 0.01
    shown, actual = app.display_e_field(), app.electric.grid(app.world)
    assert not np.allclose(shown, actual)
