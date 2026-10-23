import numpy as np
import pygame

from embloks.config import Settings
from embloks.grid import Grid
from embloks.materials import Material, Materials, Orientation
from embloks.particles import Particles


class MaterialLayer:
    """Draws materials, rebuilding the cached image only when they change."""

    def __init__(self):
        self._surface: pygame.Surface | None = None
        self._key: tuple | None = None

    def draw(
        self,
        surface: pygame.Surface,
        grid: Grid,
        materials: Materials,
        settings: Settings,
        show_ions: bool = True,
    ):
        key = (id(materials), materials.version, grid, show_ions)
        if key != self._key:
            self._surface = self._build(grid, materials, settings)
            if show_ions:
                _draw_ions(self._surface, grid, materials, settings)
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


def _draw_ions(surface: pygame.Surface, grid: Grid, materials: Materials, settings: Settings):
    """A small ring at the centre of every conductive cell: its fixed positive ion.

    Drawn onto the cached material image (in grid-local pixels), and only on cells
    big enough to show it.
    """
    size = grid.cell_px
    if size < settings.min_cell_px_for_glyphs:
        return
    radius = max(2, round(size * 0.12))
    for col, row in zip(*np.nonzero(materials.conductive)):
        centre = (col * size + size // 2, row * size + size // 2)
        pygame.draw.circle(surface, settings.ion_color, centre, radius, width=max(1, size // 16))


def draw_electrons(surface: pygame.Surface, grid: Grid, particles: Particles, settings: Settings):
    if len(particles) == 0:
        return
    xy = np.array([grid.origin_x, grid.origin_y]) + particles.pos * grid.cell_px
    radius = grid.cell_px * settings.electron_radius
    if len(particles) <= settings.max_electrons_as_circles and radius >= 1.5:
        for x, y in xy:
            pygame.draw.circle(surface, settings.electron_color, (x, y), radius)
        return
    pixels = pygame.surfarray.pixels3d(surface)
    ix = np.clip(xy[:, 0].astype(int), 0, pixels.shape[0] - 1)
    iy = np.clip(xy[:, 1].astype(int), 0, pixels.shape[1] - 1)
    pixels[ix, iy] = settings.electron_color
    del pixels  # Release the surface lock.


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


def draw_hud(
    surface: pygame.Surface,
    font: pygame.font.Font,
    sections: list[list[tuple[str, str]]],
    settings: Settings,
) -> None:
    """Info panel at the top-left: sections of (label, value) rows, split by thin rules.

    Labels form their own column in an accent colour, so it is always clear which
    text belongs to which label.
    """
    pad, label_gap, row_gap, section_gap = 10, 14, 3, 14
    rendered = [
        [
            (font.render(label, True, settings.hud_label_color),
             font.render(value, True, settings.hud_text_color))
            for label, value in section
        ]
        for section in sections
        if section
    ]
    label_width = max(lab.get_width() for section in rendered for lab, _ in section)
    value_x = pad + label_width + label_gap
    width = value_x + max(val.get_width() for section in rendered for _, val in section) + pad
    line_height = font.get_linesize() + row_gap
    rows = sum(len(section) for section in rendered)
    height = 2 * pad + rows * line_height + (len(rendered) - 1) * section_gap

    box = pygame.Surface((width, height), pygame.SRCALPHA)
    box.fill(settings.hud_background)
    surface.blit(box, (0, 0))
    y = pad
    for i, section in enumerate(rendered):
        if i:
            rule_y = y + section_gap // 2 - row_gap
            pygame.draw.line(surface, settings.hud_rule_color, (pad, rule_y), (width - pad, rule_y))
            y += section_gap
        for label, value in section:
            surface.blit(label, (pad, y))
            surface.blit(value, (value_x, y))
            y += line_height


def arrow_stride(grid: Grid, settings: Settings) -> int:
    """Draw an arrow on every n-th cell, so arrows stay at least min_arrow_spacing_px apart."""
    return max(1, -(-settings.min_arrow_spacing_px // grid.cell_px))


def arrow_sample_cells(grid: Grid, settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    """Cells that get a field arrow: all of them, or every n-th on fine grids."""
    stride = arrow_stride(grid, settings)
    cols, rows = np.meshgrid(
        np.arange(stride // 2, grid.cols, stride),
        np.arange(stride // 2, grid.rows, stride),
        indexing="ij",
    )
    return cols.ravel(), rows.ravel()


def draw_field_arrows(
    surface: pygame.Surface,
    grid: Grid,
    cells: tuple[np.ndarray, np.ndarray],
    field: np.ndarray,
    settings: Settings,
) -> None:
    """Arrows centred on cells, with length and brightness growing with log10|field|."""
    cols, rows = cells
    magnitude = np.hypot(field[:, 0], field[:, 1])
    lo, hi = settings.e_field_min, settings.e_field_max
    with np.errstate(divide="ignore"):
        t = np.clip(np.log(magnitude / lo) / np.log(hi / lo), 0, 1)
    spacing = grid.cell_px * arrow_stride(grid, settings)
    centre_x = grid.origin_x + (cols + 0.5) * grid.cell_px
    centre_y = grid.origin_y + (rows + 0.5) * grid.cell_px
    dim = np.array(settings.e_arrow_dim_color)
    bright = np.array(settings.e_arrow_color)
    for i in np.flatnonzero(t > 0):
        length = spacing * 0.9 * t[i]
        if length < 2:
            continue
        ux, uy = field[i] / magnitude[i]
        cx, cy = centre_x[i], centre_y[i]
        tail = (cx - ux * length / 2, cy - uy * length / 2)
        tip = (cx + ux * length / 2, cy + uy * length / 2)
        color = dim + (bright - dim) * t[i]
        pygame.draw.line(surface, color, tail, tip)
        head = length * 0.3
        # Two barbs at +-30 degrees from the reversed direction.
        for sx, sy in ((0.866, 0.5), (0.866, -0.5)):
            bx = -(ux * sx - uy * sy) * head
            by = -(uy * sx + ux * sy) * head
            pygame.draw.line(surface, color, tip, (tip[0] + bx, tip[1] + by))


def draw_fixed_charges(
    surface: pygame.Surface, grid: Grid, positions: np.ndarray, charges: np.ndarray, settings: Settings
) -> None:
    radius = max(3, grid.cell_px * 0.35)
    for (x, y), q in zip(positions, charges):
        cx, cy = grid.origin_x + x * grid.cell_px, grid.origin_y + y * grid.cell_px
        color = settings.positive_charge_color if q > 0 else settings.negative_charge_color
        pygame.draw.circle(surface, color, (cx, cy), radius)
        if radius >= 6:
            half, width = radius * 0.55, max(1, round(radius / 5))
            pygame.draw.line(surface, (255, 255, 255), (cx - half, cy), (cx + half, cy), width)
            if q > 0:
                pygame.draw.line(surface, (255, 255, 255), (cx, cy - half), (cx, cy + half), width)
