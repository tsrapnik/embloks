from dataclasses import dataclass


@dataclass
class Settings:
    window_width: int = 1000
    window_height: int = 800
    background_color: tuple[int, int, int] = (18, 18, 24)

    # Render rate and fixed physics timestep (simulation time per tick).
    fps: int = 60
    dt: float = 1 / 120
    # Upper bound on physics ticks per frame, so a slow frame can't spiral.
    max_ticks_per_frame: int = 8

    # Number of grid columns; rows follow from the window aspect ratio.
    grid_density: int = 40
    min_grid_density: int = 4
    max_grid_density: int = 200
    # Factor applied per +/- key press.
    grid_density_step: float = 1.25

    grid_line_color: tuple[int, int, int] = (40, 40, 52)
    hover_color: tuple[int, int, int] = (150, 150, 190)
    # Below this cell size (pixels) grid lines are hidden to avoid a solid mesh.
    min_cell_px_for_lines: int = 5
    # Below this cell size battery +/- glyphs are left out (only the colored stripes remain).
    min_cell_px_for_glyphs: int = 14

    # Brush values step through these ladders with Up/Down.
    resistance_values: tuple[float, ...] = (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100)
    default_resistance: float = 1
    emf_values: tuple[float, ...] = tuple(range(1, 21))
    default_emf: float = 5
    max_brush_radius: int = 10

    # Conductors shade from bright (lowest resistance) to dark (highest).
    conductor_color_low_r: tuple[int, int, int] = (235, 160, 90)
    conductor_color_high_r: tuple[int, int, int] = (95, 60, 40)
    battery_color: tuple[int, int, int] = (70, 80, 70)
    battery_plus_color: tuple[int, int, int] = (220, 60, 60)
    battery_minus_color: tuple[int, int, int] = (60, 110, 230)

    hud_font_size: int = 20
    hud_text_color: tuple[int, int, int] = (220, 220, 230)
    hud_background: tuple[int, int, int, int] = (0, 0, 0, 160)
