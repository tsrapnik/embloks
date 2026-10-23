import numpy as np
import pygame

from embloks.config import Settings
from embloks.grid import Grid
from embloks.materials import Material, Materials, Orientation


class MaterialLayer:
    """Draws materials, rebuilding the cached image only when they change."""

    def __init__(self):
        self._surface: pygame.Surface | None = None
        self._key: tuple | None = None

    def draw(self, surface: pygame.Surface, grid: Grid, materials: Materials, settings: Settings):
        key = (id(materials), materials.version, grid)
        if key != self._key:
            self._surface = self._build(grid, materials, settings)
            self._key = key
        surface.blit(self._surface, (grid.origin_x, grid.origin_y))
        _draw_battery_marks(surface, grid, materials, settings)

    @staticmethod
    def _build(grid: Grid, materials: Materials, settings: Settings) -> pygame.Surface:
        image = np.empty((grid.cols, grid.rows, 3), dtype=np.uint8)
        image[:] = settings.background_color

        conductor = materials.kind == Material.CONDUCTOR
        r_values = settings.resistance_values
        # Position of each resistance between the ladder's ends, on a log scale.
        t = np.log(materials.resistance[conductor] / r_values[0]) / np.log(r_values[-1] / r_values[0])
        t = np.clip(t, 0, 1)[:, None]
        low = np.array(settings.conductor_color_low_r)
        high = np.array(settings.conductor_color_high_r)
        image[conductor] = (low + (high - low) * t).astype(np.uint8)

        image[materials.kind == Material.BATTERY] = settings.battery_color

        small = pygame.surfarray.make_surface(image)
        # pygame.transform.scale is nearest-neighbour, so cells stay crisp squares.
        return pygame.transform.scale(small, (grid.pixel_width, grid.pixel_height))


def _draw_battery_marks(
    surface: pygame.Surface, grid: Grid, materials: Materials, settings: Settings
) -> None:
    """Red stripe on each battery's + side, blue on its - side, plus glyphs on large cells."""
    size = grid.cell_px
    stripe = max(1, size // 5)
    for col, row in zip(*np.nonzero(materials.kind == Material.BATTERY)):
        x, y, _, _ = grid.cell_rect(col, row)
        ox, oy = Orientation(materials.orientation[col, row]).vector
        cx, cy = x + size / 2, y + size / 2
        for sign, color in ((1, settings.battery_plus_color), (-1, settings.battery_minus_color)):
            # Stripe hugging the cell edge in direction sign * (ox, oy).
            ex, ey = sign * ox, sign * oy
            if ex:
                rect = (x + size - stripe if ex > 0 else x, y, stripe, size)
            else:
                rect = (x, y + size - stripe if ey > 0 else y, size, stripe)
            pygame.draw.rect(surface, color, rect)
            if size >= settings.min_cell_px_for_glyphs:
                gx, gy = cx + ex * size / 4, cy + ey * size / 4
                half = size / 8
                width = max(1, size // 12)
                pygame.draw.line(surface, color, (gx - half, gy), (gx + half, gy), width)
                if sign > 0:
                    pygame.draw.line(surface, color, (gx, gy - half), (gx, gy + half), width)


def draw_grid_lines(surface: pygame.Surface, grid: Grid, settings: Settings) -> None:
    if grid.cell_px < settings.min_cell_px_for_lines:
        return
    left, top = grid.origin_x, grid.origin_y
    right, bottom = left + grid.pixel_width, top + grid.pixel_height
    for col in range(grid.cols + 1):
        x = left + col * grid.cell_px
        pygame.draw.line(surface, settings.grid_line_color, (x, top), (x, bottom))
    for row in range(grid.rows + 1):
        y = top + row * grid.cell_px
        pygame.draw.line(surface, settings.grid_line_color, (left, y), (right, y))


def draw_brush_outline(
    surface: pygame.Surface, grid: Grid, cells: tuple[np.ndarray, np.ndarray], settings: Settings
) -> None:
    for col, row in zip(*cells):
        pygame.draw.rect(surface, settings.hover_color, grid.cell_rect(col, row), width=1)


def draw_hud(surface: pygame.Surface, font: pygame.font.Font, lines: list[str], settings: Settings):
    """Text lines in a translucent box at the top-left corner."""
    rendered = [font.render(line, True, settings.hud_text_color) for line in lines]
    pad = 6
    width = max(r.get_width() for r in rendered) + 2 * pad
    height = sum(r.get_height() for r in rendered) + 2 * pad
    box = pygame.Surface((width, height), pygame.SRCALPHA)
    box.fill(settings.hud_background)
    surface.blit(box, (0, 0))
    y = pad
    for r in rendered:
        surface.blit(r, (pad, y))
        y += r.get_height()
