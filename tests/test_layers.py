import numpy as np
import pygame
import pytest

from embloks.config import Settings
from embloks.layers import GAINS, OPACITIES, Layer, LayerPanel


def test_log_intensity_spans_bounds_and_gain_shifts_it():
    layer = Layer("E")
    t = layer.intensity(np.array([0.0, 1e-3, np.sqrt(1e-3 * 1.0), 1.0, 5.0]), 1e-3, 1.0)
    assert np.allclose(t, [0, 0, 0.5, 1, 1])
    layer.gain = 10
    assert layer.intensity(np.array([0.1]), 1e-3, 1.0)[0] == pytest.approx(1)


def test_linear_intensity_is_proportional():
    layer = Layer("B", log_scale=False)
    assert np.allclose(layer.intensity(np.array([0.0, 0.25, 0.5, 2.0]), 1e-3, 0.5), [0, 0.5, 1, 1])


def test_gain_and_opacity_steps_are_clamped():
    layer = Layer("E")
    for _ in range(20):
        layer.step_gain(1)
        layer.step_opacity(-1)
    assert layer.gain == GAINS[-1] and layer.opacity == OPACITIES[0]


@pytest.fixture
def panel():
    pygame.init()
    surface = pygame.Surface((1000, 800))
    layers = [Layer("E field"), Layer("Particles", adjustable=False)]
    panel = LayerPanel(layers, pygame.font.Font(None, 20), Settings())
    panel.draw(surface)
    yield panel, layers, surface
    pygame.quit()


def test_panel_buttons_change_their_layer(panel):
    panel, (e, particles), surface = panel
    for name in ("E field:show", "E field:scale+", "E field:opacity-", "E field:mode",
                 "Particles:show"):
        assert panel.handle_click(panel.button_rects[name].center)
        panel.draw(surface)
    assert not e.visible and e.gain == 2 and e.opacity == 0.75 and not e.log_scale
    assert not particles.visible
    assert "Particles:scale+" not in panel.button_rects


def test_panel_sits_top_right_and_ignores_clicks_elsewhere_or_when_hidden(panel):
    panel, _, surface = panel
    assert panel.rect.right == surface.get_width() and panel.rect.top == 0
    assert not panel.handle_click((5, 790))
    assert panel.handle_click((panel.rect.right - 1, panel.rect.bottom - 1))  # Panel, no button.
    panel.visible = False
    panel.draw(surface)
    assert not panel.handle_click(panel.button_rects["E field:show"].center)
