import pygame

from embloks.config import Settings
from embloks.grid import Grid


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


def draw_hover(
    surface: pygame.Surface, grid: Grid, cell: tuple[int, int] | None, settings: Settings
) -> None:
    if cell is None:
        return
    pygame.draw.rect(surface, settings.hover_color, grid.cell_rect(*cell), width=1)
