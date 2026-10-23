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
    hover_color: tuple[int, int, int] = (90, 90, 120)
    # Below this cell size (pixels) grid lines are hidden to avoid a solid mesh.
    min_cell_px_for_lines: int = 5
