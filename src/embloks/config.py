from dataclasses import dataclass


@dataclass
class Settings:
    window_width: int = 1000
    window_height: int = 800
    background_color: tuple[int, int, int] = (18, 18, 24)

    # Render rate and fixed physics timestep (simulation time per tick). Results were
    # checked to be unchanged up to dt = 1/8; 1/30 leaves a safety margin.
    fps: int = 60
    dt: float = 1 / 30
    # Simulation time per real second, changed live with , and . keys. Circuits need
    # tens of time units to settle, so the default runs fast.
    sim_speeds: tuple[float, ...] = (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64)
    default_sim_speed: float = 8
    # Upper bound on physics ticks per frame; beyond it the simulation runs slower
    # than the chosen speed instead of freezing the app.
    max_ticks_per_frame: int = 32

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
    # Kept modest: holding a voltage takes separated charge, and small circuits run
    # out of electrons above ~10 (see cell_charge).
    emf_values: tuple[float, ...] = tuple(range(1, 11))
    default_emf: float = 3
    max_brush_radius: int = 10

    # Conductors shade from bright (lowest resistance) to dark (highest).
    conductor_color_low_r: tuple[int, int, int] = (235, 160, 90)
    conductor_color_high_r: tuple[int, int, int] = (95, 60, 40)
    battery_color: tuple[int, int, int] = (70, 80, 70)
    battery_plus_color: tuple[int, int, int] = (220, 60, 60)
    battery_minus_color: tuple[int, int, int] = (60, 110, 230)

    # Simulation units: lengths in cells, Coulomb constant 1 (below), charges in units e.
    # Charge of the fixed ion in each conductor cell: the conductor's carrier density.
    # Batteries separate charge to build up their voltage; this must be large enough
    # that wires don't run out of electrons doing so (real metals have plenty).
    cell_charge: float = 4.0
    # Electrons per conductor cell; each carries -cell_charge / n so cells stay neutral.
    carriers_per_cell: int = 4
    # |charge| / mass of an electron, the same for any carriers_per_cell.
    carrier_charge_to_mass: float = 1.0

    # Drag on electrons: damping rate (1/time) = resistance * drag_per_resistance *
    # cell_charge. Scaling with the carrier density makes a cell of resistance R carry
    # current I = E / R (charge per time), i.e. R is in ohms.
    drag_per_resistance: float = 1.0
    # Internal resistance per battery cell. A perfect battery would have 0; a small value
    # keeps the pump from overshooting: it moves charge at (EMF - V) / resistance.
    battery_resistance: float = 0.1
    # Displayed currents are averaged over roughly this much simulation time.
    current_smoothing_time: float = 1.0
    # Fraction of normal velocity kept when an electron bounces off a closed cell face.
    wall_restitution: float = 0.3
    # Largest move (cells) per substep, and the cap on substeps per tick.
    max_step: float = 0.4
    max_substeps: int = 16

    # Coulomb constant 1/(4 pi eps0).
    coulomb_k: float = 1.0
    # Speed of light (cells per time unit); mu0 / 4 pi = coulomb_k / c^2. Magnetic forces
    # are ~ (v/c)^2 times electric ones, so a lower c makes them stronger. Electrons
    # drift at ~0.01-1, far below this.
    speed_of_light: float = 2.0
    # Softening length (cells) of ions and electrons. Ions are smeared over ~their cell,
    # so neighbouring ions form a smooth positive background electrons can roam over
    # (a lattice ripple of roughly exp(-2 pi * 0.8) ~ 1%). Electrons must use the same
    # value: softened charges only cancel exactly when smeared alike, otherwise every
    # neutral cell leaks a ~1/r^4 field that adds up across a conductor.
    conductor_softening: float = 0.8
    test_charge_softening: float = 0.5

    # Fixed test charges (in units of e) painted with the charge tool.
    test_charge_values: tuple[float, ...] = (-10, -5, -2, -1, -0.5, 0.5, 1, 2, 5, 10)
    default_test_charge: float = 1
    positive_charge_color: tuple[int, int, int] = (230, 70, 70)
    negative_charge_color: tuple[int, int, int] = (70, 120, 240)

    # E-field arrows: lengths and brightness scale with log10|E| between these bounds.
    e_field_min: float = 1e-3
    e_field_max: float = 2.0
    e_arrow_color: tuple[int, int, int] = (255, 215, 80)
    e_arrow_dim_color: tuple[int, int, int] = (90, 75, 30)
    # On fine grids only every n-th cell gets an arrow, keeping arrows this far apart.
    min_arrow_spacing_px: int = 16

    # Poynting arrows, scaled like the E arrows between these bounds of |S|.
    s_field_min: float = 1e-5
    s_field_max: float = 0.1
    s_arrow_color: tuple[int, int, int] = (90, 235, 140)
    s_arrow_dim_color: tuple[int, int, int] = (30, 80, 50)

    # B heatmap: opacity grows with log10(c^2 |B|) between these bounds. Scaling by c^2
    # (i.e. showing the current-like part of B) keeps the colours independent of c.
    b_field_min: float = 1e-3
    b_field_max: float = 1.0
    b_out_color: tuple[int, int, int] = (235, 70, 70)
    b_in_color: tuple[int, int, int] = (70, 130, 255)
    b_max_alpha: int = 150

    ion_color: tuple[int, int, int] = (200, 40, 40)
    electron_color: tuple[int, int, int] = (90, 170, 255)
    # Electron dot radius as a fraction of the cell size.
    electron_radius: float = 0.12
    # Above this many electrons, draw single pixels instead of circles (much faster).
    max_electrons_as_circles: int = 4000

    # Save slots (Ctrl/Shift + digit) live here, relative to the working directory.
    save_dir: str = "saves"
    # How long (real seconds) save/load messages stay in the info panel.
    message_seconds: float = 6.0

    hud_font_size: int = 20
    hud_text_color: tuple[int, int, int] = (220, 220, 230)
    hud_label_color: tuple[int, int, int] = (130, 190, 255)
    hud_rule_color: tuple[int, int, int] = (70, 70, 90)
    hud_background: tuple[int, int, int, int] = (0, 0, 0, 175)
    button_color: tuple[int, int, int] = (40, 40, 55)
    button_active_color: tuple[int, int, int] = (45, 90, 140)

    # Displayed E (and S) are averaged over roughly this much simulated time, so single
    # electrons jittering don't make the arrows flicker. 0 shows the instantaneous field.
    display_smoothing_time: float = 0.5
