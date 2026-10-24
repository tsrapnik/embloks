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
| `F`                 | Battery frequency: DC, or AC at 0.02-0.2 cycles per time unit |
| `M`                 | Field mode: instant / dynamic (induction and waves)           |
| `[` / `]` or wheel  | Smaller / larger brush                                        |
| `C`                 | Clear the canvas                                              |
| Ctrl + `0`-`9`      | Save the scene to a slot                                      |
| Shift + `0`-`9`     | Load the scene from a slot                                    |
| `F1` - `F7`         | Load an example scene (see *Example scenes*)                  |
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

- one fixed **positive ion** (red ring, drawn at the centre), with charge +4 e (`cell_charge`)
  spread evenly over the cell (see *Charge distribution*), and
- 8 **electrons** (blue dots) of charge -0.5 e each (`carriers_per_cell`), which move freely
  within their island (see *Electron motion*), so the cell is neutral overall. Splitting the
  charge finely matters: a battery moves one electron at a time, so smaller electrons mean a
  steadier current (see *Batteries*).

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
instead of each one getting stuck on an ion. How exactly ions and electrons are smeared is
described next.

### Charge distribution

The field is computed on the grid (see *Under the hood*), and the grid can only hold charge on
cell centres. Each charge is therefore described twice: a coarse version on the grid that sets
all fields at a distance, and a sharper version used only between charges less than
`electron_cutoff` (1 cell) apart.

**Ions.** The grid holds each ion's +4 e as shares on its own and its 8 neighbouring cell centres
(1), each share then smeared with a softening of 0.8 cell (`conductor_softening`) (2). The shares
are exactly what a charge spread evenly over the cell puts on the grid, and at close range the ion
*is* that evenly spread charge (3). In a wire the ions therefore form an even positive
background up to the walls, with no favoured spot inside a cell.

![A conductor cell's ion: its shares on the grid, the smeared long-range version and the
uniform close-range version](docs/images/ion.png)

**Electrons.** On the grid an electron's charge is shared over the 4 nearest cell centres in
proportion to how close it is (*cloud-in-cell*), and smeared like the ions (top row). Its shape
on the grid thus depends on where it sits: on a cell centre it looks exactly like an ion, on a
corner it is a broad, flat cloud. That cloud is far too wide to keep electrons apart: two
electrons a quarter of a cell apart barely repel through the grid. So within 1 cell, electrons
interact as compact balls with a softening of only 0.25 (`electron_softening`, bottom row): they
push each other apart, while the evenly spread ions hold them in. An electron gas filling a
conductor evenly feels no net close-range force, just as with the grid fields.

![One electron at four positions: its grid cloud (top) and its compact close-range ball
(bottom)](docs/images/electron.png)

Without the close-range part (set `electron_cutoff` to 0), electrons in a thin wire collapse
into a few tight stacks: two electrons at the same spot feel no force from each other through
the grid, and identical forces keep them together. With it, they spread evenly over the wire
(the right picture below starts from the same stacked state):

![Repulsion between two electrons versus distance, and electrons in a wire with the close-range
part off and on](docs/images/electron_spacing.png)

The batteries include the close-range part when they read their terminal voltage (averaged over
each cell, as at a single point it would jump with every electron passing by): electrons crowded
at a terminal push harder than the grid potential alone shows. The figures are drawn by
[docs/make_figures.py](docs/make_figures.py) (needs matplotlib).

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

Newly painted cells get their electrons spread evenly over the cell, matching the ion, so a fresh
conductor is nearly field-free. The outermost electrons settle a little in the first few time
units, and the electrons end up in a regular pattern (see *Charge distribution*). Conductor
surfaces keep a small field (about 0.04, fading within a few cells) where the electrons next to a
wall settle slightly inwards.

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

### Example scenes

`F1`-`F7` load ready-made scenes. Each starts from rest, so you watch it settle. Each resets the
physics settings and display layers to their defaults, then shows the layers that suit it.

- **F1 Battery and resistor**: the textbook circuit. A rectangular loop of low-resistance wire,
  the battery on the left (`+` up) and a high-resistance resistor on the right. Green S arrows show
  the energy leaving the battery and flowing *through the space* inside and around the loop into
  the resistor, while B (blue inside the loop: into the screen) shows the clockwise current.
- **F2 Screening**: a neutral conductor between a `+5` and a `-5` test charge. Electrons shift
  until the field inside the block is nearly zero, and field lines meet its surface at right
  angles.
- **F3 Batteries in series**: two identical loops, one driven by a single cell of EMF 2, the other
  by three in series. Hover over their wires: the right loop carries three times the current.
- **F4 Capacitor**: two large parallel plates, close together, each wired to one battery
  terminal. Watch them charge: electrons gather on the `-` plate's inner face and leave the `+`
  plate, until the plates are one EMF apart (about 60 time units: an RC charging curve through
  the wires). Between the plates the field is nearly uniform; at the plate ends it fringes out.
- **F5 Parallel paths**: a battery drives two thick bus bars with three resistors between them,
  all of the same length and material. The two identical ones carry equal currents; the one
  twice as wide has half the resistance and carries twice the current. Hover a resistor and
  read **Wire I** in the info panel (the current per cell, *I*, is the same in all three: the
  wide one just has twice as many cells side by side). The resistors are 2 and 4 cells wide
  rather than 1 and 2, because one-cell wires are the least accurate (see *Batteries*).
- **F6 Transformer** (dynamic fields): a big loop driven by two AC sources, and a smaller loop
  inside it with only a resistor. The inner loop's current is induced by the outer loop's
  changing magnetic flux: it lags the outer current by a quarter period (it follows -dI/dt,
  Lenz's law), and its flux visibly cancels part of the field inside it (darker colour). Press
  `M` for instant fields and the current through its resistor stops.
- **F7 Antenna** (dynamic fields): a half-wave wire antenna driven at its middle, and an
  identical receiver one wavelength away with a resistor in its middle. B colours show the waves
  spreading at light speed; the green S arrows point outwards (radiated energy). Hover the
  receiver's middle and read *Wire I*; with `M` (instant fields) it drops ~7x.

Save your own variations to a slot with Ctrl + digit.

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
  before drawing, so weak fields become visible (or strong ones stop saturating). Click the
  value to switch to **auto**: the gain then follows the field so that its strongest ~1% just
  reaches full strength. B and S start in auto: they grow with the current (S with the current
  squared), so a fixed scale would wash out the whole screen in a low-resistance circuit,
- **opacity**: −/+ steps through 10% to 100%, e.g. keep E faintly in the background while
  watching S,
- **mode**: **log** or **linear**. Log scale shows weak and strong fields side by side;
  linear scale makes arrow length (or colour strength) proportional to the field, which is
  honest about *how much* stronger one place is than another. E and S start in log, B in linear:
  on a log scale a loop's weak return field outside looks almost as strong as the field inside.

Particles can only be shown or hidden. Clicks on the panel never paint.

The E arrows (and S, which uses E) are averaged over about half a time unit of simulated time
(`display_smoothing_time`), so they don't flicker as single electrons jitter. The average
holds still while paused, and restarts from the exact field whenever you paint, so edits show
up immediately. B is drawn from the averaged current for the same reason. The physics always
uses the exact, instantaneous fields.

### Dynamic fields: induction and waves

By default fields are **instant**: Coulomb's law and the Biot-Savart law, recomputed from where
the charges are right now. That covers circuits, screening and magnetic fields of steady
currents, but a *changing* magnetic field creates no electric field, and nothing travels at a
finite speed. `M` switches to **dynamic fields**: Maxwell's equations stepped forward in time,
as in plasma physics particle-in-cell codes. You then also get:

- **Induction**: a changing magnetic flux drives currents (transformers, F6), and every loop has
  self-inductance, so AC currents lag and big loops resist fast changes.
- **Waves**: changes in the currents spread outward at the speed of light (`speed_of_light`,
  2 cells per time unit by default) and carry energy away (antennas, F7).

Use **AC sources** to see it: with the battery tool, `F` steps the frequency from DC through
0.02-0.2 cycles per time unit; the EMF then follows `EMF·sin(2πft)`. A wire whose length is half
the wavelength `speed_of_light / frequency` makes a good antenna.

This mode is genuine **2D electromagnetism**: everything is uniform out of the screen, so a
"charge" is really an infinitely long line of charge. Two consequences:

- Static fields fall off as 1/r instead of 1/r², a bit further reaching than in instant mode.
  DC circuits still carry the same currents in both modes (checked by the tests).
- A current loop in the plane is really an infinitely long **solenoid**: its magnetic field is
  uniform inside and zero outside, and its inductance is μ0 × area, which is large. A transformer
  therefore needs one loop *inside* the other (side by side they share no flux). The secondary
  voltage is set by the fraction of the primary's area it encloses. The transformer scene uses a
  faster light speed (8), which makes the inductance 16x smaller, so a real current flows round
  the loops instead of just charging them up.

Switching modes starts the dynamic fields from rest (only the electrostatic part), and the
fields themselves aren't saved: a loaded scene starts them from rest too.

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
  of N cells of resistance R has resistance N·R. In wires only one cell wide, expect up to ~20%
  deviation: each charge is smeared over about a cell, so the field can't step sharply where a
  thin wire meets a resistor. Wider wires get closer (3 cells wide: within ~4%).
- **Lumpy currents:** a small current is made of few electrons. At 0.12 e per time unit only one
  (-0.5 e) electron passes a point every ~4 time units, so readings averaged over a short time
  scatter by 10-20%. That is real shot noise of discrete charges, not an error.
- **Current reading:** the battery's current is the charge it actually moved, averaged like
  the wire currents (over `current_smoothing_time`, 2 time units).
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
  its centre (from the current field mode, shown in the status as *Fields*),
- for conductors, the **current** through it: conventional current (the direction positive
  charge would flow, opposite to the electrons), averaged over 2 time units
  (`current_smoothing_time`), like an ammeter's response time. **I** is the current through
  that one cell; **Wire I** is the total through the whole wire at that point, like a clamp
  meter: it adds up the cells across the wire, perpendicular to the flow.

Readings sit in fixed columns: x and y components, then the magnitude, with every number
right-aligned in a column wide enough for any value. Numbers use 4 decimals between 0.001 and
1000 (scientific notation outside that range), so the decimal point stays put and nothing shifts
sideways as values change.

All numbers in the panel (V, E, B, S, I, the battery's terminal voltage and current, and the
mean |E| in wires) are additionally low-pass filtered over about one *real* second
(`readout_smoothing_seconds`), so they are readable at any simulation speed instead of
flickering. When you move to another cell or edit the scene, the readings restart from the
exact value, so they never slide over from a stale one. Only the displayed numbers are
filtered, not the physics. E inside a wire still moves a little: it is small on average and
fluctuates slowly as single electrons pass.

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
  even positive background that electrons can glide over. On the grid, ions and electrons share
  the same ε, so an evenly filled conductor cancels exactly.
- **Close-range correction.** Between electrons less than `electron_cutoff` apart, the
  difference between a compact (ε = 0.25) and the grid's (ε = 0.8) Coulomb force is added
  directly, pair by pair (pairs found with a k-d tree), tapered smoothly to zero at the cutoff.
  The ions' matching close-range pull only depends on which cells around an electron conduct, so
  it is tabulated once per distinct 3 × 3 neighbourhood when you paint, and read back with one
  lookup per electron. This roughly doubles the cost of a physics step in typical scenes, and
  costs more in very large solid conductors (the number of electron pairs grows with them).
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
- **Dynamic fields (FDTD).** E and B live on a staggered (Yee) grid: Ex and Ey on cell edges,
  B on cell corners, stepped with Faraday's and Ampère's laws in a leapfrog. After every step E is
  corrected so that Gauss's law holds exactly (divergence cleaning, solved with a sine transform);
  the electrostatic part is thereby instantaneous, as in the Coulomb gauge, and only fields of
  changing currents travel at c. That also keeps the fields right when charge jumps without a
  current (painting, battery transfers). A 16-cell margin with gradually rising damping absorbs
  outgoing waves. Electrons read each field component from its own staggered position: averaging
  onto cell centres first would hide an odd/even checkerboard of charge from them, which then
  grows unchecked. Charge and current pass through a 1-2-1 smoothing filter (twice), dynamic
  mode's counterpart of the softening length. If light would cross more than ~0.6 cell per
  tick, the field step is split into substeps (the CFL stability limit).
- **Large time step.** Drag is stable at any step and the electron motion is smooth, so a tick
  can cover 1/30 time unit; results were checked to be identical up to 1/8.

Typical frame times at the default 8× speed, all field layers on:

| Grid      | Electrons | Frame    |
|-----------|-----------|----------|
| 40 × 32   | ~3,000    | ~14 ms   |
| 100 × 80  | ~19,000   | ~39 ms   |
| 200 × 160 | ~76,000   | ~160 ms  |

(30% of the canvas filled with wire, 8 electrons per cell.)

On the largest grids the FFTs dominate (physics and display alike); the simulation then runs
slower than the chosen speed instead of freezing.

