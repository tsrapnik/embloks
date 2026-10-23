# Embloks — grid-based EM sandbox in Python

## Context
Greenfield project (repo only contains `.devcontainer/`, Python 3.13, X11 display available via WSLg).
Goal: an interactive sandbox where you paint materials (vacuum / conductor / battery) on a grid,
conductors carry charge carriers, and every tick the E, B and Poynting fields are computed and
drawn as toggleable vector layers. Build it in small, runnable steps.

Stack: **pygame-ce** (window, input, drawing; has 3.13 wheels) + **numpy** (all physics vectorised,
struct-of-arrays). `pytest` for physics sanity tests. Package layout `src/embloks/`, run with
`python -m embloks`.

---

## Physics decisions (resolving the paradoxes)

### 1. Magnetism in 2D — your intuition is right
Charges only move in the plane, so `v` is in-plane. Biot–Savart for a moving point charge:
`B = (μ0/4π) q (v × r̂)/r²` → cross product of two in-plane vectors is **purely along z**.
So per cell we store:
- `E = (Ex, Ey)` — 2D vector
- `B = Bz` — a **signed scalar** (out of / into the screen), drawn as heatmap or ⊙/⊗ symbols
- `S = E × B / μ0 = (Ey·Bz, −Ex·Bz)/μ0` — back in-plane, 2D vector
- Lorentz force `q(v × B)` = `q·Bz·(vy, −vx)` — also in-plane. Fully self-consistent, no 3D needed.

We treat it as "3D point charges confined to a plane" (fields fall off as 1/r²), not true 2D
electromagnetism (line charges, 1/r). The former matches intuition better.

### 2. What the field model can and cannot do (be aware)
Coulomb + Biot–Savart summed over particles is **quasi-static**: fields propagate instantly.
You get electrostatics, currents, Ohm's law, magnetic forces between currents, and a correct
Poynting picture (energy flowing from battery through space into resistors — the classic demo).
You do **not** get induction (Faraday: changing B → E), self-inductance, or EM waves.
Those need a real field solver; planned as an optional final stage (2D TE-mode FDTD/PIC, which
uses exactly `Ex, Ey, Bz` — the same fields, so the data model carries over).

### 3. Point ions would trap electrons → conductor becomes an insulator
If the + ion is a point charge at the cell centre, each electron falls into a deep Coulomb well
and stays there: the material acts as a dielectric, not a metal. Fix (jellium model): treat the
ion's charge as **smeared over its cell** (softening length ≈ cell size for the ion field).
Neighbouring smeared ions form a nearly uniform positive background, so electrons roam freely and
only feel *net* charge imbalances — exactly how a metal behaves.
Electron–electron interaction also gets a smaller softening to avoid 1/r² blow-ups.

### 4. Self-force and "the force from the block I'm in"
Sampling the field at the block centre includes the particle's own field (it would push itself).
Rule: **forces on a particle are evaluated at the particle's position, excluding itself.**
The per-block field grid is for visualisation (and Poynting). One function
`field_at(points, exclude_idx)` serves both.

### 5. Graininess: carriers per cell
One electron per block gives very noisy fields/current. Each conductor cell has an ion of
`cell_charge` and `carriers_per_cell = n` electrons of `−cell_charge/n` (defaults 4 and 4, so
electrons are −1 e; see §8 for why the density is 4), so neutrality holds and fields smooth out. Display fields are also **time-averaged** (EMA over a few
ticks), physics uses instantaneous values.

### 6. Resistance as drag (Drude model) — works as you intended
`F_drag = −γ·v`, with γ taken from the cell's resistance. At terminal velocity `q E = γ v`,
which *is* Ohm's law (resistivity ∝ γ). Integrate drag implicitly
(`v = (v + a·dt) / (1 + γ·dt/m)`) so very resistive cells don't blow up.

### 7. Confinement to continuous conductor
- Connectivity is **4-neighbour** (diagonal touching does *not* conduct).
- After each step, if a particle would leave its allowed region, reflect the velocity component
  across that wall and clamp it inside (inelastic, small restitution).
- Limit per-substep displacement to < ½ cell (adaptive substeps) so particles can't tunnel
  through one-cell walls.
- Painting/erasing: painting a conductor cell adds 1 ion + n electrons. Erasing removes the ion
  and n electrons (those inside the cell first, else nearest in the same island); remaining
  electrons in the erased cell are pushed to the nearest conductor cell of the same island.

### 8. Voltage source — "perfect battery" = feedback charge pump
*Original plan: a uniform EMF force on electrons inside the battery cell. Built and measured in
step 7; it does not work here.* Electrons in the + wire sit still (screened), so nothing feeds
the pump. And at the battery's faces the force jumps, so an electron there is pushed back from
both sides for any static field between 0 and ℰ: every terminal voltage in [0, ℰ] is a resting
state, set by history (measured: 2.4 V of 5 V with R = 1 wires).

What is built instead (step 7, `battery.py`):
- A battery cell holds no charges and is a wall for electrons on all sides.
- Battery cells lined up along their axis form a **stack**; EMFs add with sign, and internal
  resistances add (0.1 per cell). The conductor cells beyond its ends are its terminals.
- Each tick: measure V = V(+) − V(−) (grid potential averaged over the terminal's 3×3
  same-island neighbourhood, which damps single-electron noise), then move charge + → − at
  `I = (ℰ − V)/R_int`, one electron at a time. That is exactly a Thevenin voltage source:
  V = ℰ at zero current, ℰ − I·R_int under load. This stands in for the electrolyte's ions.
- The electron taken is the one nearest the source terminal **along the wire** (cached BFS
  distances), not in a straight line: in a closed loop the nearest one in a straight line can be
  the one just delivered across the battery.
- Charge density matters: holding ℰ needs separated charge of order ℰ, so cells carry
  `cell_charge` = 4 (4 electrons of −1 each) and EMF is capped at 10, otherwise small loops
  run out of electrons. Drag scales with `cell_charge`, so a cell of resistance R carries I = E/R.
- Verified: open-circuit V → ℰ; loop current → ℰ/(N·R + R_int) (59-cell loop: 0.085 vs 0.0846);
  reversed polarity reverses the current.

### 9. Units & constants
Simulation units: `k = 1/(4πε0)` = 1, electron mass = 1, e = 1, **length unit = 1 cell**.
Physics is then independent of grid density (density only sets how many cells fit on screen).
Ions are not stored as particles: every conductive cell implicitly has one at its centre, derived
from the materials, so ions and cells can never get out of sync. `c` is a tunable setting (magnetic effects scale with v²/c², so a low c makes
them visible). Changing grid density resets the scene (or nearest-neighbour resample — later).

### 10. Performance
*Time to equilibrium:* circuits settle on an RC time scale (tens of time units), so the app has
a simulation-speed control (default 8×) and dt = 1/30 (results unchanged up to dt = 1/8).
Direct summation is O(particles × (cells + particles)). Mitigations, in order:
1. Ion field is static → cache it, recompute only on paint.
2. Fully vectorised numpy (chunked to bound memory).
3. If too slow at target grid sizes: deposit charge/current onto the grid and compute E, Bz by
   **FFT convolution** with the Coulomb/Biot–Savart kernels (O(M log M)), particles sample via
   bilinear interpolation.

---

## Project layout
```
pyproject.toml                  # deps: pygame-ce, numpy; dev: pytest
src/embloks/
  __main__.py                   # entry: python -m embloks
  config.py                     # Settings dataclass (grid density, carriers_per_cell, c, dt, ...)
  app.py                        # main loop: input → physics substeps → render
  grid.py                       # Grid: screen layout, cell <-> pixel mapping
  materials.py                  # Materials: kind, resistance, emf, orientation arrays [col, row]
  connectivity.py               # islands (4-neighbour), open faces incl. battery side walls
  world.py                      # World: materials + connectivity + electrons, kept in sync on paint
  particles.py                  # Particles SoA (electrons only): pos, vel, charge, mass
  fields.py                     # field_at(points, exclude) -> E, Bz; ion-field cache; poynting
  physics.py                    # step(): forces, implicit drag, integrate, confine
  render.py                     # layers: materials, grid lines, particles, E, B, S
  ui.py                         # brush, tool selection, layer toggles/scale/opacity, HUD, inspector
tests/                          # physics sanity tests
```

## Build steps (each ends with something runnable)

1. **Empty canvas** — `pyproject.toml`, pygame-ce window, main loop, fixed timestep, FPS in title,
   Esc quits, Space pauses.
2. **Grid** — `Settings.grid_density`; draw grid lines; mouse → cell mapping; hovered cell highlight;
   `+/-` changes density (resets scene).
3. **Materials & brush** — material arrays in `materials.py`; tools: vacuum (erase), conductor (with
   resistance value), battery (with EMF, `R` rotates orientation); brush size; material colours,
   battery drawn with +/− marks.
4. **Particles (static)** — painting spawns ion + n electrons, erasing removes them (rule §7);
   connectivity islands computed on paint; render ions/electrons.
5. **Electrostatics + E layer** — `fields.py` with softened Coulomb (§3), cached ion field;
   per-cell E arrows with log-magnitude scaling. Add a "fixed test charge" tool to poke conductors.
   Done with the §10 FFT approach for the *display* grid already: ions and test charges sit on
   cell centres (exact), electrons are cloud-in-cell deposited (blurred within ~1 cell).
   `ElectricField.at` stays an exact direct sum for arbitrary points.
6. **Electron dynamics** — Lorentz (E only for now), implicit drag, substeps, confinement (§7).
   Expect: electrons screen the test charge, E inside conductor → ~0.
   Done particle-in-cell style: forces read from the FFT grid field with the same cloud-in-cell
   weights used for deposition (zero self-force, momentum conserving). Electrons spawn on their
   ion (+tiny jitter): random spawning left a disordered "electron glass" that relaxed slowly
   and kept visible noise. Measured: ~95% of an external field screened within ~5 time units.
7. **Battery** — feedback charge pump (§8, replaces the planned EMF force). HUD inspector: hover
   a cell to see V, E and current through it. Verified Ohm's law with a battery + wire loop.
8. **Magnetism** — Bz from moving charges (Biot–Savart), v×B force, B layer (red/blue heatmap or
   ⊙/⊗), `c` setting.
   Done: Bz = (1/c²)[Jx ⊛ Ky − Jy ⊛ Kx] with the existing FFT kernels, J deposited cloud-in-cell
   (no magnetic self-force). Battery pump current is added along stack cells (pumped electrons
   teleport, so they carry no velocity). Display uses the smoothed current; red = out of screen,
   blue = into (screen y is down, so +Bz is *into* the screen). c = 2 by default.
9. **Poynting** — S = E×B/μ0 per cell, S layer.
   Done (green arrows, `S` key, off by default). Battery → resistor demo shows energy flowing
   through space into the resistor. In-plane flux around a resistor is ~1/6 of I²R: expected,
   since 3D point-charge fields also carry energy above/below the plane (§1 geometry choice).
10. **Visualisation controls** — per layer: on/off, scale, opacity, linear/log; display EMA
    smoothing; small side panel (keyboard shortcuts first, clickable panel after).
    Done: `layers.py` (Layer state + clickable LayerPanel, top right, `L` toggles). Display E is
    an EMA in *simulated* time (frozen while paused, reset on paint); B from smoothed current.
11. **Performance** — profile; FFT-convolution field solver if needed (§10).
    Done: FFT solver was already in (steps 5–6). Profiling showed rendering dominated:
    vectorised arrows + direct draw at full opacity + batched electron sprites (render 25 → 7 ms
    at default grid). Physics: cloud-in-cell weights shared per substep, B combined in frequency
    space. Fused/batched E+B FFT measured *slower* at large grids; not used.
12. **Save / load** — store the full state (materials, electrons with velocities, test charges,
    battery pump state, smoothed current, the physics settings that shape the state, view
    layers and speed) so a settled circuit resumes exactly. Slots via keyboard (no file
    dialogs in pygame).
13. **Built-in scenes** — a few loadable example scenes. Most important: the textbook
    battery + resistor circuit (rectangular loop, battery on one side, resistor on the
    opposite side), where the Poynting vector clearly flows from battery to resistor through
    the space around the wires. Others: screening around a test charge, parallel currents,
    series batteries.
    Done: `scenes.py`, F1-F4 (battery + resistor, screening, series, open circuit). Tests check
    the physics each scene demonstrates (S into the resistor, 3x current in series).
    Measured for step 14: F1 loop current is +18% over Ohm's law; the battery's pump current
    has a std of ~0.36 around a mean of ~0.1-0.2 (discrete jumps); electrons clump in wires.
14. **Tune the physics for the textbook scene — no cheating.** It must stay a physics
    simulation that shows how real physics works, not an animation faked to match the picture.
    Known problems to investigate and fix at their cause:
    - B jitters fast next to the battery, presumably because the pump makes electrons jump
      (discrete teleports → spiky current near the terminals).
    - With very low-resistance wires the magnetic field goes all over the place (underdamped
      sloshing currents).
    Look at algorithms and parameters (pump smoothing, carrier density, damping, time step,
    display averaging) and verify each change against the existing physics tests.
    Done. Findings (measured on F1, B "flicker" = frame-to-frame std / mean near battery):
    - Battery jitter, two causes. (1) The B display took the battery current from the
      *commanded* pump rate (ℰ − V)/R_int, which swings ±4.7 between single-electron moves;
      now it is the charge actually moved, smoothed like wire currents. (2) Charge quantum:
      noise ∝ electron charge. carriers 4 → 8 (−0.5 e) plus 2 time units of current averaging:
      flicker 0.60 → 0.16, tick 0.7 → 0.8 ms. Not helping: larger R_int (no effect), filtering
      the battery's voltage (feedback lag, 2-4x worse).
    - "B all over the place" with low R: not sloshing (current settles smoothly, ±10%) but
      display — 10x current saturated a fixed log colour window, and log made the weak return
      field outside a loop look as strong as inside. Now B and S auto-scale; B defaults to linear.
    - Electron stacking (60-90% within 0.02 cells): the grid's pair force is ~4.5x too weak
      inside a cell (double smoothing). An exact P3M short-range correction was built and
      verified; it removes stacking only at 4 carriers, costs 3-10x per tick, and with 8+
      carriers stacking costs little energy anyway (spacing ≪ softening). Not kept.
    - Ohm's law in 1-cell wires: +16..27% / −17% depending on geometry; 3-cell wires within
      ~4%. A resolution limit (field can't step faster than the ~1 cell smoothing), documented.
    - Drag now scales with q/m too (Γ = R·cell_charge·q/m), so I = E/R for any q/m.
15. **Optional: induction & waves (FDTD/PIC)** — decide later whether to add it at all.
    Today fields follow charges instantly (Coulomb + Biot–Savart). A field *solver* instead
    stores E and B on the grid and steps Maxwell's equations forward in time (FDTD, 2D TE mode:
    Ex, Ey, Bz — the same quantities we already draw), with electrons depositing current into
    it (particle-in-cell). What that adds: a changing B creates an E (induction: coils,
    transformers, back-EMF, self-inductance), and fields travel at the finite speed c, so
    changes ripple outward as electromagnetic waves (an oscillating current radiates). Costs:
    much smaller time step (light must not cross a cell per step), absorbing boundaries at the
    edges, interaction with the electrostatics we have, and a whole new class of numerical
    artefacts. Likely a separate "dynamic fields" mode next to the current quasi-static one.

## Verification
- `pytest`: single charge E matches kq/r² (outside softening); field excluding self is zero for a
  lone particle; current loop gives Bz with correct sign (right-hand rule); S is perpendicular to
  E and has magnitude |E||Bz|/μ0; drag-only electron reaches terminal velocity qE/γ; particles
  never leave their conductor island over many random steps; open-circuit battery terminal
  potential difference → ℰ; closed loop current ≈ ℰ/R (within noise).
- Manual: `python -m embloks` after every step; paint a ring of wire with a battery and check
  steady current, S arrows pointing from battery into resistive sections.
