# Embloks

A grid-based electromagnetism sandbox. Paint vacuum, conductors and batteries onto a grid and
watch the electric, magnetic and Poynting fields evolve. See [plan.md](plan.md) for the design
and the step-by-step build plan.

## Running

Requires Python 3.11+ and a display (in the devcontainer on WSL2, WSLg provides one).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python -m embloks      # or simply: embloks
```

## Tests

```bash
pytest
```

Tests live in [tests/](tests/) and run without a display (`tests/conftest.py` selects SDL's dummy
video driver). Configuration is the `[tool.pytest.ini_options]` section of `pyproject.toml`.

## Using the app

| Input               | Action                                                        |
|---------------------|---------------------------------------------------------------|
| Left mouse (drag)   | Paint with the current tool                                   |
| Right mouse (drag)  | Erase: test charges if one is under the brush, else material  |
| `1` / `2` / `3`     | Tool: conductor / battery / test charge                       |
| Up / Down           | Tool value: resistance / EMF / charge                         |
| `R`                 | Rotate battery orientation (clockwise)                        |
| `[` / `]` or wheel  | Smaller / larger brush                                        |
| `C`                 | Clear the canvas                                              |
| Ctrl + `0`-`9`      | Save the scene to a slot                                      |
| Shift + `0`-`9`     | Load the scene from a slot                                    |
| `P`                 | Hide / show particles (ions and electrons)                    |
| `E`                 | Hide / show electric field arrows                             |
| `B`                 | Hide / show magnetic field colours                            |
| `S`                 | Hide / show energy flow (Poynting) arrows (off at start)      |
| `L`                 | Hide / show the layers panel (top right, clickable)           |
| `H`                 | Info panel: full / status only / hidden                       |
| `+` / `-`           | Finer / coarser grid (resets the scene)                       |
| Space               | Pause / resume physics                                        |
| `,` / `.` (`<` `>`) | Slower / faster simulation (¼× to 64×, default 8×)            |
| Esc                 | Quit                                                          |

**Erasing:** right-click decides once, when you press the button: if a test charge is under the
brush, that stroke removes test charges only (dragging over wires leaves them intact); otherwise
it erases materials back to vacuum. So right-clicking a charged wire twice first removes the
charge, then the wire.

### Grid

The canvas is divided into a grid of square cells. The grid density (about how many columns fit
across the window, 4 to 200) is set by `grid_density` in
[src/embloks/config.py](src/embloks/config.py) and changed live with `+`/`-`, which clears the
canvas. On very fine grids the grid lines are hidden so they don't drown out the canvas.

### Materials

Every cell is one of (erase with right-click to get vacuum back):

- **Vacuum** (dark): empty space.
- **Conductor** (copper): carries charge. Each conductor cell has a resistance in ohms, picked
  from 0.1 to 100 with Up/Down before painting. Low-resistance cells are bright copper,
  high-resistance cells dark brown. Painting over a conductor replaces its resistance.
- **Battery** (grey, red and blue stripes): an ideal voltage source, see *Batteries* below. The
  red stripe/`+` marks the positive terminal, the blue stripe/`-` the negative one. Set the EMF
  (1 to 10) with Up/Down and the direction of the `+` side with `R` before painting.

Conductor cells only connect through shared edges: two cells touching only at a corner are *not*
connected. Dragging the brush always paints an edge-connected path, so a quick diagonal stroke
still gives an unbroken wire. A group of connected conductor cells is called an *island*;
electrons can never leave their island. Batteries are walls for electrons; they move charge
between the islands at their terminals instead.

### Charges

Every conductor cell holds:

- one fixed **positive ion** (red ring) at its centre, with charge +4 e (`cell_charge`), and
- 4 **electrons** (blue dots) of charge -1 e each (`carriers_per_cell`), which move freely
  within their island (see *Electron motion*), so the cell is neutral overall.

The charge per cell sets how many free electrons a wire has. Batteries need to pile up some
charge on their terminals to hold their voltage, and a wire with too few electrons would simply
run empty (real metals have enormously many). Both values are in
[config.py](src/embloks/config.py).

Painting a conductor cell adds its ion and electrons. Erasing a cell (or painting a battery over
it) removes its ion and the same number of electrons from that island: first electrons inside the
erased cell, then the nearest ones. If more electrons than that were in the erased cell, the extras
are pushed into the nearest remaining cell of the island (or disappear if the whole island is
erased). Repainting a conductor with another resistance keeps its charges.

Ion rings are only drawn when cells are large enough; on fine grids electrons become single pixels.

### Test charges

Tool `3` places fixed **test charges** (red with `+`, blue with `-`) at cell centres, anywhere on
the canvas, including on top of conductors. Pick the charge (-10 e to +10 e) with Up/Down. A larger
brush paints a whole charged plate, one charge per cell. Right-click removes test
charges (see *Erasing* above). Test charges never move; use them to poke at conductors.

### Electric field

Yellow arrows show the electric field **E** at cell centres: the force a positive charge would
feel there. It is the sum of the fields of all charges (ions, electrons and test charges), using
Coulomb's law. Both arrow length and brightness grow with the *logarithm* of the field strength
(between `e_field_min` and `e_field_max` in [config.py](src/embloks/config.py)), because fields
near a charge are many times stronger than fields far away. On fine grids only every n-th cell
gets an arrow, so arrows don't overlap. The info panel shows the E value under the mouse. Scale,
opacity and linear/log display can be changed in the layers panel (see *Display layers*).

Each charge is slightly "smeared" (a softening length in config.py), so fields never become
infinite at a charge. Ions are smeared over about a cell: together, neighbouring ions then act
like a smooth positive background, which is what lets electrons move freely through a conductor
instead of each one getting stuck on an ion. Electrons are smeared exactly the same way
(`conductor_softening`), so an electron sitting on its ion cancels it completely.

### Electron motion

While the simulation runs (Space pauses it), every electron is pushed by the electric field at
its position (F = qE). Moving electrons feel a **drag** proportional to the resistance of the cell
they are in. Drag is what makes resistance resist: an electron in a steady field settles at a
drift speed proportional to the field and inversely proportional to the resistance (Ohm's law).
Low resistance means fast, sloshing electrons; high resistance means slow, creeping ones.

Electrons can never leave their island: they bounce off any cell edge that doesn't conduct
(vacuum, batteries, the edge of the canvas), losing part of their speed
(`wall_restitution`). Fast electrons are moved in several small substeps, so they can't skip
through a one-cell-wide wall.

What to try: place a test charge next to a conductor. Electrons flow towards a positive charge
(away from a negative one) until the field *inside* the conductor is nearly zero. This is
**screening**, and it's why field lines always meet a conductor's surface at right angles. The info
panel shows the mean |E| over all conductor cells, which drops as the electrons settle. It
never reaches exactly zero: a few discrete electrons per cell are grainy, so some small arrows of
random direction remain. More `carriers_per_cell` makes it smoother.

Newly painted cells get their electrons right on top of their ions, so a fresh conductor is
neutral everywhere and has no field until something disturbs it.

### Magnetic field

Moving charges make a magnetic field **B** (Biot–Savart law), and B pushes on moving charges
(the Lorentz force `F = q(E + v × B)`). Because every charge moves within the plane, B always
points straight *out of* or *into* the screen. So it is a single signed number per cell, shown
as a colour rather than arrows:

- **red**: B points out of the screen,
- **blue**: B points into the screen,
- stronger colour: stronger field (logarithmic scale, like the E arrows).

Right-hand rule: curl the fingers of your right hand along the (conventional) current; your
thumb points along B. A loop with current flowing anticlockwise on screen is red inside and
faintly blue outside. A battery's own current counts too, so the loop is closed.

The magnetic force is perpendicular to an electron's motion: it steers, never speeds up. It is
roughly (v/c)² times weaker than the electric force, where `speed_of_light` (2 cells per time
unit in [config.py](src/embloks/config.py)) is much larger than the electrons' drift speeds, so
its effect on the motion is tiny, as in real circuits. Lower `speed_of_light` to exaggerate it.
The colours are scaled by c², so they look the same whatever c is. The info panel shows B under
the mouse. `B` toggles the colours.

Limitation: fields here follow the charges instantly. A *changing* magnetic field does not
create an electric field, so there is no induction (no coils, transformers or EM waves).

### Energy flow (Poynting vector)

Green arrows (toggle with `S`, off at start) show the **Poynting vector** `S = E × B / μ0`: which
way electromagnetic energy flows, and how much. Since E lies in the plane and B points out of it,
S lies in the plane again. Arrows are log-scaled like the E arrows. Show S with E hidden (`E`), as
both sit on the same cells.

The surprising textbook result shows up clearly here. Build a loop of low-resistance wire with
a short high-resistance section (a resistor) and a battery. The energy does **not** travel inside
the wires: it leaves the battery, flows through the empty space beside the wires (inside and
outside the loop), and pours into the resistor sideways, where it turns into heat (the drag on
the electrons). The wires only guide it.

Caveat: charges here are 3D point charges confined to a plane, so in a real 3D world part of the
energy also flows above and below the screen. The arrows show the direction and relative strength
correctly, but adding up S around a resistor *in the plane* gives only part of the I²R power it
dissipates (about 1/6 in a quick check).

### Saving and loading

**Ctrl + a digit** saves the whole scene to one of ten slots, **Shift + the digit** loads it
back. A slot is a file `saves/slot<N>.npz` (relative to where you start the app), so you can copy
or share them. A short message in the info panel confirms each save or load.

A save holds everything needed to continue exactly where you left off: the materials, every
electron with its velocity, test charges, the batteries' internal state, the averaged currents,
the simulated time, the speed and the display layers. It also stores the physics settings that
shaped the scene (charge per cell, softening, speed of light, ...), and restores them on load,
so a scene behaves the same even after `config.py` was edited. Loading resizes the grid to the
saved scene.

### Display layers

What you see is built from four layers: **E field** arrows, **B field** colours,
**S energy flow** arrows and **particles** (ions and electrons). The **layers panel** in the
top-right corner (`L` hides it) has a row per layer; click its buttons to change:

- **show**: on / off (same as the `E`, `B`, `S`, `P` keys),
- **scale**: −/+ halves or doubles the gain (×1/16 to ×16). Fields are multiplied by it
  before drawing, so weak fields become visible (or strong ones stop saturating),
- **opacity**: −/+ steps through 10% to 100%, e.g. keep E faintly in the background while
  watching S,
- **mode**: **log** (default) or **linear**. Log scale shows weak and strong fields side by
  side; linear scale makes arrow length (or colour strength) proportional to the field, which
  is honest about *how much* stronger one place is than another. Use scale to fit it.

Particles can only be shown or hidden. Clicks on the panel never paint.

The E arrows (and S, which uses E) are averaged over about half a time unit of simulated time
(`display_smoothing_time`), so they don't flicker as single electrons jitter. The average
holds still while paused, and restarts from the exact field whenever you paint, so edits show
up immediately. B is drawn from the averaged current for the same reason. The physics always
uses the exact, instantaneous fields.

### Simulation speed

Physics advances in fixed ticks of `dt` = 1/30 time unit. Electrons respond within a few time
units, but a circuit needs tens of units to settle: the battery has to build up small surface
charges along the wires through their resistance (like charging a capacitor through a
resistor), and only then does the current flow evenly around the loop. So by default the
simulation runs at **8×**: 8 time units per real second, and a typical loop reaches its steady
current within about 10 seconds. `,` and `.` halve or double the speed (¼× to 64×); the info
panel shows it. If the computer can't keep up (large grids, high speeds) the simulation runs
slower than chosen rather than freezing; the window title shows the frame rate and simulated
time.

### Batteries

A battery is an ideal **voltage source**: it keeps the voltage between its `+` and `-` terminals
at its EMF. The terminals are the conductor cells touching its `+` and `-` faces.

Electrons never pass *through* a battery. Instead, like the ions in a real battery's electrolyte,
it takes electrons from the wire at its `+` terminal and delivers them to the wire at its `-`
terminal. Each tick it measures the voltage V between its terminals and moves charge at a rate

    I = (EMF - V) / R_internal

with a small internal resistance (`battery_resistance`, 0.1 per cell). With nothing connected,
charge builds up until V = EMF and the battery stops. With a circuit connected, it keeps
supplying exactly the current the circuit draws, at V = EMF - I·R_internal. You'll see electrons
vanish on the `+` side and appear on the `-` side.

- **Series:** battery cells lined up end to end along their axis form one battery. Their EMFs
  add, and a cell pointing the other way subtracts. Each cell adds 0.1 internal resistance.
- **Ohm's law:** in a closed loop the current settles at `EMF / total resistance`, where a wire
  of N cells of resistance R has resistance N·R. Very small loops carry a little more, because
  electrons cut the corners and the wire is effectively shorter than its cell count.
- **Patience:** the surface charges that steer the current around the loop take a while to
  build up: about 60 time units for a long loop of resistance-1 wire (under 10 seconds at the
  default 8× speed), less for lower resistance. Speed up with `.` if it's still too slow.
- **Limits:** a battery whose terminals aren't both wired does nothing. Very high EMFs on
  small circuits empty the wires of electrons, which is why EMF stops at 10.

Hover over a battery to see its stack's EMF, the measured terminal voltage and its current.

### Info panel

The brush outline follows the mouse. The info panel (top-left) has up to three sections, with
labels in blue on the left: the **status** (current tool and its value, brush size, numbers of
electrons, islands and test charges, mean |E| inside conductors), the **cell under the mouse**,
and the **keys**. `H` cycles between showing all of it, only status and cell, and nothing. For
the cell under the mouse it shows:

- the material, island number and electron count (or the battery's stack details),
- the electric potential **V**, field **E**, magnetic field **B** and energy flow **S** at
  its centre,
- for conductors, the **current** through it: conventional current (the direction positive
  charge would flow, opposite to the electrons), averaged over about a second
  (`current_smoothing_time`).

The window title shows the frame rate, simulation time, paused state and grid size
(columns x rows).

## Under the hood

### Libraries

- **[pygame-ce](https://pyga.me/)**: window, mouse/keyboard input and drawing.
- **[NumPy](https://numpy.org/)**: all simulation state lives in arrays (one entry per cell or
  per electron), so every step is a handful of vectorised operations instead of Python loops.
- **[SciPy](https://scipy.org/)**: FFTs for the field calculation, connected-component
  labelling to find islands, and graph distances along wires for the batteries.
- **[pytest](https://pytest.org/)**: tests, including physics checks (Coulomb's law, screening,
  electrons never escaping their island).

### Techniques and optimizations

- **Charge smoothing (softening).** Each charge is treated as a small blurred cloud rather than
  a point: `E = k·q·r / (r² + ε²)^1.5`. This keeps fields finite when charges get close, makes a
  charge's field at its own centre zero (so nothing pushes itself), and smooths the ions into an
  even positive background that electrons can glide over. Ions and electrons share the same ε, so
  an electron sitting on its ion cancels it exactly.
- **Fields via FFT.** Summing every charge's field at every cell costs
  `cells × charges` operations. Instead, the charges are put on the grid and the field (and the
  potential, which the batteries need) is computed as a convolution with the Coulomb kernel using
  FFTs, which costs about `cells × log(cells)` no matter how many electrons there are. The
  kernel's FFT is computed once per grid size. The magnetic field reuses the same kernels: B is
  the current density (charge × velocity) convolved with them, crossed.
- **Cloud-in-cell.** Electrons sit anywhere, not just on cell centres, so each one's charge (and
  current) is shared over the 4 nearest cell centres in proportion to how close it is. Each
  electron reads the fields back with exactly the same weights. This symmetry means an electron
  feels no electric or magnetic force from itself, and every push between two electrons is equal
  and opposite.
- **Caching.** Ions and test charges only change when you paint, so their field is computed once
  and reused until then. Each tick only the electrons' part is recomputed. The material image is
  likewise only redrawn after painting.
- **Stable drag.** Resistance is applied implicitly, `v ← (v + a·dt) / (1 + Γ·dt)`, which stays
  stable for any resistance; a simple explicit update would blow up at high resistance.
- **Substeps.** If electrons move fast, a tick is split into smaller steps so that nothing moves
  more than 0.4 cell at a time and can't tunnel through a one-cell wall.
- **Shared weights.** The cloud-in-cell weights are computed once per substep and reused for
  depositing charge and current and for reading E and B back at every electron.
- **B in one transform.** B's two convolution terms are combined in frequency space before
  transforming back, saving an inverse FFT. (Fusing E and B into one batched FFT was tried and
  measured slower, so it isn't done.)
- **Drawing.** Arrow geometry is computed for all arrows at once with NumPy, then drawn with one
  call per arrow; layers at full opacity draw straight to the screen, and only translucent ones
  go through an overlay. Electrons are one pre-drawn dot sprite blitted in a single batched
  call, or single pixels written into the screen array when there are very many. On fine grids
  only every n-th cell gets a field arrow.
- **Large time step.** Drag is stable at any step and the electron motion is smooth, so a tick
  can cover 1/30 time unit; results were checked to be identical up to 1/8.

Typical frame times at the default 8× speed, all field layers on:

| Grid      | Electrons | Frame    |
|-----------|-----------|----------|
| 40 × 32   | ~1,500    | ~10 ms   |
| 100 × 80  | ~10,000   | ~27 ms   |
| 200 × 160 | ~38,000   | ~100 ms  |

On the largest grids the FFTs dominate (physics and display alike); the simulation then runs
slower than the chosen speed instead of freezing.

