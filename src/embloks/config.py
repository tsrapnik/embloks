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
