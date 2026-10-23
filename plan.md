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
One electron per block gives very noisy fields/current. Setting `carriers_per_cell = n`
(default 1, try 4): each conductor cell gets n electrons of charge `−e/n` (ion stays `+e`), so
neutrality holds and fields smooth out. Display fields are also **time-averaged** (EMA over a few
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

### 8. Voltage source — "perfect battery" = EMF force inside the block
Textbook definition of EMF: a **non-electrostatic force** that pushes charges against E inside the
source, with `∫ f/q · dl = ℰ`. Implementation:
- A battery cell is a conductor cell (has ion + electrons) plus an orientation (→ ↑ ← ↓) and EMF ℰ.
- Electrons inside it feel an extra uniform force `F = e·ℰ / L` (L = cell size) pointing from
  the + face to the − face (electrons are pumped to the − terminal).
- The two **side faces are walls** (directional connectivity), only + and − faces connect —
  matches your "remaining sides neutral".
- "Perfect" = no internal resistance → γ_battery ≈ 0 (small value for numerical stability).
  In open circuit, charge piles up at the terminals until the electrostatic field exactly
  cancels the pump: terminal voltage = ℰ. Under load the voltage stays ℰ, current = ℰ/R.
- Multi-cell batteries in series just add EMF (each cell has its own orientation).
- Rejected alternative: teleporting electrons from + to − face — that's a *current* source,
  not a voltage source.

### 9. Units & constants
Simulation units: `k = 1/(4πε0)` = 1, electron mass = 1, e = 1, **length unit = 1 cell**.
Physics is then independent of grid density (density only sets how many cells fit on screen).
Ions are not stored as particles: every conductive cell implicitly has one at its centre, derived
from the materials, so ions and cells can never get out of sync. `c` is a tunable setting (magnetic effects scale with v²/c², so a low c makes
them visible). Changing grid density resets the scene (or nearest-neighbour resample — later).

### 10. Performance
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
7. **Battery** — EMF force + side walls (§8). HUD inspector: hover a cell to see E, Bz, S, and
   current through it (charge crossing per second). Verify Ohm's law with a battery + wire loop.
8. **Magnetism** — Bz from moving charges (Biot–Savart), v×B force, B layer (red/blue heatmap or
   ⊙/⊗), `c` setting.
9. **Poynting** — S = E×B/μ0 per cell, S layer.
10. **Visualisation controls** — per layer: on/off, scale, opacity, linear/log; display EMA
    smoothing; small side panel (keyboard shortcuts first, clickable panel after).
11. **Performance** — profile; FFT-convolution field solver if needed (§10).
12. **Stretch** — save/load scenes (JSON/npz); optional FDTD/PIC field mode for induction & waves.

## Verification
- `pytest`: single charge E matches kq/r² (outside softening); field excluding self is zero for a
  lone particle; current loop gives Bz with correct sign (right-hand rule); S is perpendicular to
  E and has magnitude |E||Bz|/μ0; drag-only electron reaches terminal velocity qE/γ; particles
  never leave their conductor island over many random steps; open-circuit battery terminal
  potential difference → ℰ; closed loop current ≈ ℰ/R (within noise).
- Manual: `python -m embloks` after every step; paint a ring of wire with a battery and check
  steady current, S arrows pointing from battery into resistive sections.
