"""Display layers (E, B, S, particles) and the clickable panel that controls them."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pygame

from embloks.config import Settings

GAINS = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16)
OPACITIES = (0.1, 0.25, 0.5, 0.75, 1.0)


@dataclass
class Layer:
    label: str
    visible: bool = True
    # Field values are multiplied by this before mapping to arrow length / colour.
    gain: float = 1.0
    opacity: float = 1.0
    log_scale: bool = True
    # Particles can only be shown or hidden.
    adjustable: bool = True

    def step_gain(self, steps: int) -> None:
        self.gain = _step(GAINS, self.gain, steps)

    def step_opacity(self, steps: int) -> None:
        self.opacity = _step(OPACITIES, self.opacity, steps)

    def intensity(self, magnitude: np.ndarray, low: float, high: float) -> np.ndarray:
        """Display strength 0..1 for field magnitudes.

        Log scale maps [low, high] onto [0, 1] logarithmically; linear scale maps
        [0, high] linearly. The gain shifts which magnitudes reach full strength.
        """
        scaled = magnitude * self.gain
        if self.log_scale:
            with np.errstate(divide="ignore"):
                t = np.log(scaled / low) / np.log(high / low)
        else:
            t = scaled / high
        return np.clip(t, 0, 1)


def _step(values: tuple[float, ...], current: float, steps: int) -> float:
    index = min(range(len(values)), key=lambda i: abs(values[i] - current))
    return values[max(0, min(len(values) - 1, index + steps))]


class LayerPanel:
    """Grid of buttons at the top-right corner: one row per layer.

    Columns: show/hide, scale -/+, opacity -/+, log/linear. Layout is computed
    while drawing, so clicks are matched against what was last on screen.
    """

    def __init__(self, layers: list[Layer], font: pygame.font.Font, settings: Settings):
        self.layers = layers
        self.font = font
        self.settings = settings
        self.visible = True
        self.rect: pygame.Rect | None = None
        self._buttons: list[tuple[pygame.Rect, Callable[[], None]]] = []
        # Button rectangles by name ("<layer label>:<control>"), as last drawn.
        self.button_rects: dict[str, pygame.Rect] = {}

    def handle_click(self, pos: tuple[int, int]) -> bool:
        """Run the button under `pos`. True if the click landed on the panel at all."""
        if not self.visible or self.rect is None or not self.rect.collidepoint(pos):
            return False
        for rect, action in self._buttons:
            if rect.collidepoint(pos):
                action()
                break
        return True

    def draw(self, surface: pygame.Surface) -> None:
        if not self.visible:
            self.rect = None
            return
        s, font = self.settings, self.font
        pad, gap = 10, 6
        line = font.get_linesize()
        height = line + 6
        small = font.size("−")[0] + 12  # Width of a -/+ button.
        widths = {
            "label": max(font.size(layer.label)[0] for layer in self.layers),
            "show": font.size("off")[0] + 16,
            "value": font.size("100%")[0] + 4,
            "mode": font.size("linear")[0] + 12,
        }
        row_width = (
            widths["label"] + widths["show"] + 2 * (2 * small + widths["value"])
            + widths["mode"] + 6 * gap
        )
        width = row_width + 2 * pad
        total_height = pad * 2 + line + gap + len(self.layers) * (height + gap) - gap
        x0 = surface.get_width() - width
        self.rect = pygame.Rect(x0, 0, width, total_height)

        box = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        box.fill(s.hud_background)
        surface.blit(box, self.rect.topleft)

        # Column headers.
        x = x0 + pad + widths["label"] + gap
        y = pad
        for title, w in (
            ("show", widths["show"]),
            ("scale", 2 * small + widths["value"]),
            ("opacity", 2 * small + widths["value"]),
            ("mode", widths["mode"]),
        ):
            text = font.render(title, True, s.hud_label_color)
            surface.blit(text, (x + (w - text.get_width()) // 2, y))
            x += w + gap
        surface.blit(font.render("Layers", True, s.hud_label_color), (x0 + pad, y))

        self._buttons = []
        self.button_rects = {}
        y += line + gap
        for layer in self.layers:
            x = x0 + pad
            label = font.render(layer.label, True, s.hud_text_color)
            surface.blit(label, (x, y + (height - label.get_height()) // 2))
            x += widths["label"] + gap
            name = layer.label
            self._button(surface, f"{name}:show", (x, y, widths["show"], height),
                         "on" if layer.visible else "off", layer.visible,
                         lambda layer=layer: setattr(layer, "visible", not layer.visible))
            x += widths["show"] + gap
            if layer.adjustable:
                x = self._stepper(surface, f"{name}:scale", x, y, small, widths["value"], height,
                                  f"×{layer.gain:g}", layer.step_gain) + gap
                x = self._stepper(surface, f"{name}:opacity", x, y, small, widths["value"], height,
                                  f"{layer.opacity:.0%}", layer.step_opacity) + gap
                self._button(surface, f"{name}:mode", (x, y, widths["mode"], height),
                             "log" if layer.log_scale else "linear", False,
                             lambda layer=layer: setattr(layer, "log_scale", not layer.log_scale))
            y += height + gap

    def _stepper(self, surface, name, x, y, small, value_width, height, text, step) -> int:
        """[-] value [+]; returns the x just past it."""
        self._button(surface, f"{name}-", (x, y, small, height), "−", False, lambda: step(-1))
        value = self.font.render(text, True, self.settings.hud_text_color)
        surface.blit(value, (x + small + (value_width - value.get_width()) // 2,
                             y + (height - value.get_height()) // 2))
        x += small + value_width
        self._button(surface, f"{name}+", (x, y, small, height), "+", False, lambda: step(1))
        return x + small

    def _button(self, surface, name, rect, text, active: bool, action: Callable[[], None]) -> None:
        s = self.settings
        rect = pygame.Rect(rect)
        fill = s.button_active_color if active else s.button_color
        pygame.draw.rect(surface, fill, rect, border_radius=4)
        pygame.draw.rect(surface, s.hud_rule_color, rect, width=1, border_radius=4)
        label = self.font.render(text, True, s.hud_text_color)
        surface.blit(label, label.get_rect(center=rect.center))
        self._buttons.append((rect, action))
        self.button_rects[name] = rect
