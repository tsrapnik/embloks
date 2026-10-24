use serde::{Deserialize, Serialize};

pub type Rgb = [u8; 3];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub background_color: Rgb,

    // Fixed physics timestep (simulation time per tick). Results were checked to be
    // unchanged up to dt = 1/8; 1/30 leaves a safety margin.
    pub dt: f64,
    // Simulation time per real second, changed live with , and . keys. Circuits need
    // tens of time units to settle, so the default runs fast.
    pub sim_speeds: Vec<f64>,
    pub default_sim_speed: f64,
    // Upper bound on physics ticks per frame; beyond it the simulation runs slower
    // than the chosen speed instead of freezing the app.
    pub max_ticks_per_frame: u32,

    // Number of grid columns; rows follow from the window aspect ratio.
    pub grid_density: u32,
    pub min_grid_density: u32,
    pub max_grid_density: u32,
    // Factor applied per +/- key press.
    pub grid_density_step: f64,

    pub grid_line_color: Rgb,
    pub hover_color: Rgb,
    // Below this cell size (pixels) grid lines are hidden to avoid a solid mesh.
    pub min_cell_px_for_lines: f64,
    // Below this cell size battery +/- glyphs are left out (only the colored stripes remain).
    pub min_cell_px_for_glyphs: f64,

    // Brush values step through these ladders with Up/Down.
    pub resistance_values: Vec<f64>,
    pub default_resistance: f64,
    // Kept modest: holding a voltage takes separated charge, and small circuits run
    // out of electrons above ~10 (see cell_charge).
    pub emf_values: Vec<f64>,
    pub default_emf: f64,
    pub max_brush_radius: i32,

    // Conductors shade from bright (lowest resistance) to dark (highest).
    pub conductor_color_low_r: Rgb,
    pub conductor_color_high_r: Rgb,
    pub battery_color: Rgb,
    pub battery_plus_color: Rgb,
    pub battery_minus_color: Rgb,

    // Simulation units: lengths in cells, Coulomb constant 1 (below), charges in units e.
    // Charge of the fixed ion in each conductor cell: the conductor's carrier density.
    // Batteries separate charge to build up their voltage; this must be large enough
    // that wires don't run out of electrons doing so (real metals have plenty).
    pub cell_charge: f64,
    // Electrons per conductor cell; each carries -cell_charge / n so cells stay neutral.
    // Finer charge means less graininess: a battery moves one electron at a time, and
    // with 8 (charge -0.5 each) its current and the B field near it are ~2x steadier
    // than with 4, at little cost.
    pub carriers_per_cell: u32,
    // |charge| / mass of an electron, the same for any carriers_per_cell.
    pub carrier_charge_to_mass: f64,

    // Drag on electrons: damping rate (1/time) = resistance * drag_per_resistance *
    // cell_charge * carrier_charge_to_mass. Scaling with carrier density and charge/mass
    // makes a cell of resistance R carry current I = E / R (charge per time): R is in ohms.
    pub drag_per_resistance: f64,
    // Internal resistance per battery cell: the drag on its own electrons. A perfect
    // battery would have 0; under load its terminal voltage drops by I * resistance.
    pub battery_resistance: f64,
    // Battery cells hold this many times a wire's charge (ion and electrons). To hold its
    // voltage a battery must shift some of its electrons towards its - side; with a
    // dense charge that empties only a small part of it, so it stays steady.
    pub battery_density: u32,
    // A battery scales its impressed field to deliver its EMF (battery::trim); the gain
    // corrects its shortfall over roughly this much simulation time, within this range.
    // Fast enough to cancel the jitter of the few electrons at its faces before the
    // circuit turns it into a wobbling current (~2x quicker becomes unstable).
    pub battery_trim_time: f64,
    pub battery_gain_range: (f64, f64),
    // AC frequencies (cycles per time unit) the battery tool steps through with F; 0 is DC.
    pub ac_frequencies: Vec<f64>,
    // Displayed currents (wires and batteries) are averaged over roughly this much
    // simulation time, like an ammeter's response time.
    pub current_smoothing_time: f64,
    // Fraction of normal velocity kept when an electron bounces off a closed cell face.
    pub wall_restitution: f64,
    // Largest move (cells) per substep, and the cap on substeps per tick.
    pub max_step: f64,
    pub max_substeps: u32,

    // Fields: Maxwell's equations in 2D, stepped in time (see maxwell.rs).
    // Coulomb constant 1/(4 pi eps0). In 2D a charge is a line charge: div E = 4 pi k rho
    // gives a field 2 k q / r.
    pub coulomb_k: f64,
    // Speed of light (cells per time unit); mu0 / 4 pi = coulomb_k / c^2. Magnetic forces
    // are ~ (v/c)^2 times electric ones. In 2D a loop is a solenoid with inductance
    // mu0 * area ~ 1/c^2: at 8 circuits settle in tens of time units, at 2 they ring for
    // hundreds. Electrons drift at ~0.01-1, far below this.
    pub speed_of_light: f64,
    // Passes of a 1-2-1 smoothing filter applied to charge and current: the grid's
    // softening. Each pass smears every charge over about another half cell. Two passes
    // keep the grid force on an electron steady wherever it sits in its cell; fewer let
    // it jitter with the electron's position (grid heating), so thin wires carry less.
    pub smoothing_passes: usize,
    // Width (cells) of the absorbing layer around the visible grid, and how strongly it
    // damps (per time unit, at its outer edge; it ramps up gradually).
    pub absorbing_margin: usize,
    pub absorbing_strength: f64,
    // Within electron_cutoff (cells), electrons repel as line charges smeared over only
    // electron_softening, and ions attract them as charge spread evenly over their cells
    // (close_range.rs). The grid alone smears each electron over ~a cell, too coarse to
    // keep them apart: they stacked up on a wire's centre line. With this they fill it
    // evenly. The correction is what the smoothed grid leaves out of that force; it falls
    // to ~0.3% of its peak at 3 cells, where it is cut off. Cost grows with cutoff^2;
    // 0 turns it off.
    pub electron_softening: f64,
    pub electron_cutoff: f64,

    // Fixed test charges (in units of e) painted with the charge tool.
    pub test_charge_values: Vec<f64>,
    pub default_test_charge: f64,
    pub positive_charge_color: Rgb,
    pub negative_charge_color: Rgb,

    // E-field arrows: lengths and brightness scale with log10|E| between these bounds.
    pub e_field_min: f64,
    pub e_field_max: f64,
    pub e_arrow_color: Rgb,
    pub e_arrow_dim_color: Rgb,
    // On fine grids only every n-th cell gets an arrow, keeping arrows this far apart.
    pub min_arrow_spacing_px: f64,

    // Poynting arrows, scaled like the E arrows between these bounds of |S|.
    pub s_field_min: f64,
    pub s_field_max: f64,
    pub s_arrow_color: Rgb,
    pub s_arrow_dim_color: Rgb,

    // B heatmap: opacity grows with log10(c^2 |B|) between these bounds. Scaling by c^2
    // (i.e. showing the current-like part of B) keeps the colours independent of c.
    pub b_field_min: f64,
    pub b_field_max: f64,
    pub b_out_color: Rgb,
    pub b_in_color: Rgb,
    pub b_max_alpha: f64,

    pub ion_color: Rgb,
    pub electron_color: Rgb,
    // Electron dot radius as a fraction of the cell size.
    pub electron_radius: f64,

    // Numbers in the info panel are low-pass filtered over about this many real seconds,
    // so they are readable instead of flickering. 0 shows the instantaneous values.
    pub readout_smoothing_seconds: f64,
    // How long (real seconds) save/load messages stay in the info panel.
    pub message_seconds: f64,

    // Displayed E (and S) are averaged over roughly this much simulated time, so single
    // electrons jittering don't make the arrows flicker. 0 shows the instantaneous field.
    pub display_smoothing_time: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            background_color: [18, 18, 24],
            dt: 1.0 / 30.0,
            sim_speeds: vec![0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
            default_sim_speed: 8.0,
            max_ticks_per_frame: 32,
            grid_density: 40,
            min_grid_density: 4,
            max_grid_density: 200,
            grid_density_step: 1.25,
            grid_line_color: [40, 40, 52],
            hover_color: [150, 150, 190],
            min_cell_px_for_lines: 5.0,
            min_cell_px_for_glyphs: 14.0,
            resistance_values: vec![0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
            default_resistance: 1.0,
            emf_values: (1..=10).map(f64::from).collect(),
            default_emf: 3.0,
            max_brush_radius: 10,
            conductor_color_low_r: [235, 90, 90],
            conductor_color_high_r: [95, 60, 40],
            battery_color: [70, 80, 70],
            battery_plus_color: [220, 60, 60],
            battery_minus_color: [60, 110, 230],
            cell_charge: 4.0,
            carriers_per_cell: 8,
            carrier_charge_to_mass: 1.0,
            drag_per_resistance: 1.0,
            battery_resistance: 0.1,
            battery_density: 4,
            battery_trim_time: 0.25,
            battery_gain_range: (0.5, 4.0),
            ac_frequencies: vec![0.0, 0.02, 0.05, 0.1, 0.2],
            current_smoothing_time: 2.0,
            wall_restitution: 0.3,
            max_step: 0.4,
            max_substeps: 16,
            coulomb_k: 1.0,
            speed_of_light: 8.0,
            smoothing_passes: 2,
            absorbing_margin: 16,
            absorbing_strength: 3.0,
            electron_softening: 0.25,
            electron_cutoff: 3.0,
            test_charge_values: vec![-10.0, -5.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0, 10.0],
            default_test_charge: 1.0,
            positive_charge_color: [230, 70, 70],
            negative_charge_color: [70, 120, 240],
            e_field_min: 1e-3,
            e_field_max: 2.0,
            e_arrow_color: [255, 215, 80],
            e_arrow_dim_color: [90, 75, 30],
            min_arrow_spacing_px: 16.0,
            s_field_min: 1e-5,
            s_field_max: 0.1,
            s_arrow_color: [90, 235, 140],
            s_arrow_dim_color: [30, 80, 50],
            b_field_min: 1e-3,
            b_field_max: 1.0,
            b_out_color: [235, 70, 70],
            b_in_color: [70, 130, 255],
            b_max_alpha: 150.0,
            ion_color: [200, 40, 40],
            electron_color: [90, 170, 255],
            electron_radius: 0.12,
            readout_smoothing_seconds: 1.0,
            message_seconds: 6.0,
            display_smoothing_time: 0.5,
        }
    }
}
