# Embloks

A grid-based electromagnetism sandbox that runs in the browser. Paint vacuum, conductors and
batteries onto a grid and watch the electric, magnetic and Poynting fields evolve. Curious how
the fields and forces are computed? See [docs/fields.md](docs/fields.md).

Embloks is written in Rust and compiled to WebAssembly; the physics runs in WebGPU compute
shaders and is drawn with WebGPU, so it needs a browser with WebGPU (recent Chrome, Edge,
Safari or Firefox).

## Running

Requires Rust with the `wasm32-unknown-unknown` target and [Trunk](https://trunkrs.dev/) (the dev
container installs both).

```bash
cd crates/embloks-web
trunk serve            # then open http://localhost:8080
trunk build            # or: a static site in dist/, to host anywhere
```

The site is static files only (no server-side code, no special headers), so any web host will
do. A few URL options help when testing:

| Option           | Effect                                                                 |
|------------------|------------------------------------------------------------------------|
| `?backend=cpu`   | Run the physics on the CPU (the double-precision reference) instead    |
| `?paused`        | Start with the physics paused                                          |
| `?offscreen`     | Draw to an offscreen texture instead of the canvas, for automated tests on software GPUs that can't present to a canvas; `wasmBindings.request_snapshot()` and `wasmBindings.snapshot()` read the frames back |

## Tests

```bash
cargo test --workspace
```

- [crates/embloks-core](crates/embloks-core/): the physics reference, scenes, saves and the app's
  behaviour, including physics checks (Gauss's law, waves at the speed of light, screening, Ohm's
  law, induction, antennas, electrons never escaping their island) and comparisons with numbers
  saved from the original Python version ([tests/fixtures](crates/embloks-core/tests/fixtures/)).
- [crates/embloks-gpu](crates/embloks-gpu/): the compute shaders against the CPU reference, the
  same physics checks on the GPU, and rendered images of example scenes (written to
  `target/renders/`). They need a GPU adapter: in the dev container the software Vulkan driver
  lavapipe provides one. Without any adapter they are skipped.
- [crates/embloks-web](crates/embloks-web/): key mapping and the panels' HTML.

## Using the app

| Input               | Action                                                        |
|---------------------|---------------------------------------------------------------|
| Left mouse (drag)   | Paint with the current tool                                   |
| Right mouse (drag)  | Erase: test charges if one is under the brush, else material  |
| `1` / `2` / `3`     | Tool: conductor / battery / test charge                       |
| Up / Down           | Tool value: resistance / EMF / charge                         |
| `R`                 | Rotate battery orientation (clockwise)                        |
| `F`                 | Battery frequency: DC, or AC at 0.02-0.2 cycles per time unit |
| `[` / `]` or wheel  | Smaller / larger brush                                        |
| `C`                 | Clear the canvas                                              |
| Ctrl + Shift + `0`-`9` | Save the scene to a slot                                   |
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
| Esc                 | Cancel the current stroke                                     |

The toolbar at the bottom offers the same (tools, values, brush, speed, example scenes, saves),
for touch screens too: there **Erase** makes a drag erase. `F1`-`F7` are taken by some browsers
or keyboards; the toolbar's *Examples* menu always works. Browsers keep Ctrl + digit for
switching tabs, hence Ctrl + **Shift** + digit for saving.

**Erasing:** right-click decides once, when you press the button: if a test charge is under the
brush, that stroke removes test charges only (dragging over wires leaves them intact); otherwise
it erases materials back to vacuum. So right-clicking a charged wire twice first removes the
charge, then the wire.

### Grid

The canvas is divided into a grid of square cells. The grid density (about how many columns fit
across the window, 4 to 200) is set by `grid_density` in
[settings.rs](crates/embloks-core/src/settings.rs) and changed live with `+`/`-`, which clears the
canvas. The example scenes always use a 40 x 32 grid, whatever the window's shape. On very fine
grids the grid lines are hidden so they don't drown out the canvas.

### Materials

Every cell is one of (erase with right-click to get vacuum back):

- **Vacuum** (dark): empty space.
- **Conductor** (copper): carries charge. Each conductor cell has a resistance in ohms, picked
  from 0.1 to 100 with Up/Down before painting. Low-resistance cells are bright copper,
  high-resistance cells dark brown. Painting over a conductor replaces its resistance.
- **Battery** (grey, red and blue stripes): a voltage source, see *Batteries* below. It conducts
  like a wire, but pushes its electrons towards its `-` side. The red stripe/`+` marks the
  positive terminal, the blue stripe/`-` the negative one. Set the EMF (1 to 10) with Up/Down
  and the direction of the `+` side with `R` before painting.

Conductor and battery cells only connect through shared edges: two cells touching only at a
corner are *not* connected. Dragging the brush always paints an edge-connected path, so a quick
diagonal stroke still gives an unbroken wire. A group of connected cells is called an *island*;
electrons can never leave their island. A battery belongs to the island of the wires it touches:
electrons flow through it.

### Charges

Every conductor cell holds:

- one fixed **positive ion** (red ring, drawn at the centre), with charge +4 e (`cell_charge`)
  spread evenly over the cell (see *Charge distribution*), and
- 8 **electrons** (blue dots) of charge -0.5 e each (`carriers_per_cell`), which move freely
  within their island (see *Electron motion*), so the cell is neutral overall. Splitting the
  charge finely matters: smaller electrons mean steadier currents and voltages.

A battery cell holds 4 times as much (`battery_density`): an ion of +16 e and 32 electrons.

The charge per cell sets how many free electrons a wire has. Batteries need to pile up some
charge on their terminals to hold their voltage, and a wire with too few electrons would simply
run empty (real metals have enormously many). Both values are in
[settings.rs](crates/embloks-core/src/settings.rs).

Painting a conductor or battery cell adds its ion and electrons. Erasing a cell removes its ion
and the same number of electrons from that island: first electrons inside the erased cell, then
the nearest ones. If more electrons than that were in the erased cell, the extras are pushed into
the nearest remaining cell of the island (or disappear if the whole island is erased). Painting a
battery over a wire keeps the wire's electrons and adds the battery's extra ones; painting a wire
over a battery takes the extras away again. Repainting a conductor with another resistance keeps
its charges.

Ion rings are only drawn when cells are large enough; on fine grids electrons become single pixels.

### Test charges

Tool `3` places fixed **test charges** (red with `+`, blue with `-`) at cell centres, anywhere on
the canvas, including on top of conductors. Pick the charge (-10 e to +10 e) with Up/Down. A larger
brush paints a whole charged plate, one charge per cell. Right-click removes test
charges (see *Erasing* above). Test charges never move; use them to poke at conductors.

### Fields: Maxwell's equations in 2D

The fields are computed from **Maxwell's equations**, stepped forward in time on the grid, as in
plasma physics particle-in-cell codes. Besides circuits, screening and the magnetic fields of
steady currents, that gives you:

- **Induction**: a changing magnetic flux drives currents (transformers, F6), and every loop has
  self-inductance, so AC currents lag and big loops resist fast changes.
- **Waves**: changes in the currents spread outward at the speed of light (`speed_of_light`,
  8 cells per time unit by default; the antenna scene slows it to 2) and carry energy away
  (antennas, F7).

The world is genuinely **two-dimensional**: everything is uniform out of the screen, so a
"charge" is really an infinitely long line of charge. Two consequences:

- Static fields fall off as 1/r instead of 1/r², so they reach further than you might expect.
- A current loop in the plane is really an infinitely long **solenoid**: its magnetic field is
  uniform inside and zero outside, and its inductance is μ0 × area, which is large. A transformer
  therefore needs one loop *inside* the other (side by side they share no flux). The secondary
  voltage is set by the fraction of the primary's area it encloses. Since μ0 = 4πk / c², the
  speed of light sets how inductive circuits are: at the default of 8 a circuit settles within
  tens of time units; at 2 a capacitor charged through its wires rings around the battery's
  voltage for hundreds.

The fields start from rest when a scene is loaded (only the static field of the charges), so
it takes a moment for magnetic fields and waves to build up. How it all works, step by step and
with pictures, is in [docs/fields.md](docs/fields.md).

### Electric field

Yellow arrows show the electric field **E** at cell centres: the force a positive charge would
feel there. It comes from all charges (ions, electrons and test charges) and from changing
magnetic fields. Both arrow length and brightness grow with the *logarithm* of the field strength
(between `e_field_min` and `e_field_max` in [settings.rs](crates/embloks-core/src/settings.rs)), because fields
near a charge are many times stronger than fields far away. On fine grids only every n-th cell
gets an arrow, so arrows don't overlap. The info panel shows the E value under the mouse. Scale,
opacity and linear/log display can be changed in the layers panel (see *Display layers*).

Each charge is slightly smeared over the grid, so fields never become infinite at a charge. See
*Charge distribution* next.

### Charge distribution

Each ion's charge is spread evenly over its cell, so neighbouring ions act like a smooth positive
background that electrons glide over instead of getting stuck on single ions. Electrons are
compact balls that push each other apart when they come within a few cells of each other
(`electron_cutoff`), so they spread evenly through a wire rather than piling up. Further away,
every charge acts as a cloud about two cells wide. The details, with pictures, are in
[docs/fields.md](docs/fields.md).

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
units, and the electrons end up in a regular pattern. Conductor
surfaces keep a small field (about 0.04, fading within a few cells) where the electrons next to a
wall settle slightly inwards.

### Magnetic field

Currents make a magnetic field **B** (Ampère's law), a changing B makes an electric field
(Faraday's law: induction), and B pushes on moving charges (the Lorentz force
`F = q(E + v × B)`). Because every charge moves within the plane, B always
points straight *out of* or *into* the screen. So it is a single signed number per cell, shown
as a colour rather than arrows:

- **red**: B points out of the screen,
- **blue**: B points into the screen,
- stronger colour: stronger field (logarithmic scale, like the E arrows).

Right-hand rule: curl the fingers of your right hand along the (conventional) current; your
thumb points along B. A loop with current flowing anticlockwise on screen is red inside; outside
it B is (nearly) zero, since in 2D a loop is a solenoid. Electrons flow through batteries too, so
the loop is closed.

The magnetic force is perpendicular to an electron's motion: it steers, never speeds up. It is
roughly (v/c)² times weaker than the electric force, where `speed_of_light` (8 cells per time
unit in [settings.rs](crates/embloks-core/src/settings.rs)) is much larger than the electrons' drift speeds, so
its effect on the motion is tiny, as in real circuits. Lowering `speed_of_light` exaggerates it
(and makes circuits more inductive, see above).
The colours are scaled by c², so they look the same whatever c is. The info panel shows B under
the mouse. `B` toggles the colours.

### Energy flow (Poynting vector)

Green arrows (toggle with `S`, off at start) show the **Poynting vector** `S = E × B / μ0`: which
way electromagnetic energy flows, and how much. Since E lies in the plane and B points out of it,
S lies in the plane again. Arrows are log-scaled like the E arrows. Show S with E hidden (`E`), as
both sit on the same cells.

The surprising textbook result shows up clearly here. Build a loop of low-resistance wire with
a short high-resistance section (a resistor) and a battery. The energy does **not** travel inside
the wires: it leaves the battery, flows through the empty space inside the loop, and pours into
the resistor sideways, where it turns into heat (the drag on the electrons). The wires only guide
it. (In 2D the loop is a solenoid with no field outside, so the energy stays inside it; the weak
flow you may see outside comes from the canvas's edge, which holds the loop's flux in like a
conducting box.)

Because the world is truly 2D, no energy escapes above or below the screen: adding up S around
the resistor gives the power it turns into heat (within about 10% in F1, the rest being the grid's
coarseness).

### Example scenes

`F1`-`F7` load ready-made scenes. Each starts from rest, so you watch it settle. Each resets the
physics settings and display layers to their defaults, then shows the layers that suit it.

- **F1 Battery and resistor**: the textbook circuit. A rectangular loop of low-resistance wire,
  the battery on the left (`+` up) and a high-resistance resistor on the right. Green S arrows
  (on a linear scale) show the energy leaving the battery and flowing *through the space* inside
  the loop into the resistor, while B (blue inside the loop: into the screen) shows the clockwise
  current.
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
- **F6 Transformer**: a big loop driven by two AC sources, and a smaller loop inside it with
  only a resistor. The inner loop's current is induced by the outer loop's changing magnetic
  flux: it lags the outer current by a quarter period (it follows -dI/dt, Lenz's law), and its
  flux visibly cancels part of the field inside it (darker colour). Erase one cell of the outer
  loop and the current through the inner resistor stops.
- **F7 Antenna**: a half-wave wire antenna driven at its middle, and an identical receiver one
  wavelength away with a resistor in its middle. B colours show the waves spreading at light
  speed; the green S arrows point outwards (radiated energy). Hover the receiver's middle and
  read *Wire I*.

Save your own variations to a slot with Ctrl + Shift + digit.

### Saving and loading

**Ctrl + Shift + a digit** saves the whole scene to one of ten slots, **Shift + the digit** loads
it back (or use the slot menu in the toolbar). Slots live in the browser's storage (IndexedDB),
per site and browser. **Export** downloads the scene as a `.embloks` file and **Import** opens
one, to keep or share scenes. A short message in the info panel confirms each save or load.

A save holds everything needed to continue exactly where you left off: the materials, every
electron with its velocity, test charges, the batteries' gains, the averaged currents,
the simulated time, the speed and the display layers. It also stores the physics settings that
shaped the scene (charge per cell, smoothing, speed of light, ...), and restores them on load,
so a scene behaves the same even after the defaults in `settings.rs` change. Loading resizes the grid to the
saved scene. The fields themselves aren't saved: a loaded scene starts them from rest (only the
static field of its charges), so waves and induced currents take a moment to build up again.

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
up immediately. B is averaged the same way. The physics always uses the exact, instantaneous
fields.

### AC sources

With the battery tool, `F` steps the frequency from DC through 0.02-0.2 cycles per time unit;
the EMF then follows `EMF·sin(2πft)`. A wire whose length is half the wavelength
`speed_of_light / frequency` makes a good antenna.

### Simulation speed

Physics advances in fixed ticks of `dt` = 1/30 time unit. Electrons respond within a few time
units, but a circuit needs tens of units to settle: the battery has to build up small surface
charges along the wires through their resistance (like charging a capacitor through a
resistor), and only then does the current flow evenly around the loop. So by default the
simulation runs at **8×**: 8 time units per real second, and a typical loop reaches its steady
current within about 10 seconds. `,` and `.` halve or double the speed (¼× to 64×); the info
panel shows it. If the computer can't keep up (large grids, high speeds) the simulation runs
slower than chosen rather than freezing; the page title shows the frame rate and simulated
time.

### Batteries

A battery is a **voltage source**: it keeps the voltage between its `+` and `-` terminals at
its EMF. The terminals are the wire cells touching its `+` and `-` faces.

A battery cell is a conductor with its own ion and electrons, and electrons flow through it like
through any wire. On top of the electric field, its electrons feel an *impressed field* of EMF
per cell pointing to the `+` side, so the force on them (they are negative) pushes them towards
the `-` side. This stands in for the chemistry of a real battery. It is the "source force" of
textbooks (EMF = ∮ f·dl) and the resistive voltage source of FDTD codes.

With nothing connected, electrons pile up at the `-` side until the electric field they make
cancels the impressed field inside the battery. That happens exactly when V = EMF. With a
circuit connected, the drag on the battery's own electrons is its internal resistance
(`battery_resistance`, 0.1 per cell), so V = EMF - I·R_internal. Charge only ever moves
continuously, so the fields see a smooth current, and nothing jumps.

The impressed field alone comes up short, though. With so few electrons per cell, the voltage
rests on the handful of electrons at the battery's faces. It falls 10-25% short of the EMF, and
it jitters as those electrons shift, which a circuit turns into a slowly wobbling current. So
the battery measures what it delivers, V + I·R_internal, and keeps adjusting how strongly it
pushes (its *gain*) until that equals the EMF. The adjustment takes about a quarter of a time
unit (`battery_trim_time`). This is a feedback loop on the force, never on the charge. For an
AC battery it learns the average shortfall over the cycle.

- **Series:** battery cells lined up end to end along their axis form one battery. Their EMFs
  add, and a cell pointing the other way subtracts. Each cell adds 0.1 internal resistance.
- **Ohm's law:** in a closed loop the current settles at `EMF / total resistance`, where a wire
  of N cells of resistance R has resistance N·R. In wires only one cell wide, expect up to ~20%
  deviation: each charge is smeared over a couple of cells, so the field can't step sharply
  where a thin wire meets a resistor. Wider wires get closer.
- **Lumpy currents:** a small current is made of few electrons. At 0.12 e per time unit only one
  (-0.5 e) electron passes a point every ~4 time units, so readings averaged over a short time
  scatter by 10-20%. That is real shot noise of discrete charges, not an error.
- **Current reading:** the battery's current is the current of its electrons, averaged over the
  battery and its two terminal cells and over time like the wire currents
  (`current_smoothing_time`, 2 time units).
- **Patience:** the surface charges that steer the current around the loop take a while to
  build up: about 60 time units for a long loop of resistance-1 wire (under 10 seconds at the
  default 8× speed), less for lower resistance. Speed up with `.` if it's still too slow.
- **Limits:** a battery whose terminals aren't both wired does nothing. Very high EMFs on
  small circuits empty the wires of electrons, which is why EMF stops at 10.

Hover over a battery to see its stack's EMF, the measured terminal voltage, its current and its
gain (typically 1.2-1.4: how much harder than the bare EMF it has to push).

### Info panel

The brush outline follows the mouse. The info panel (top-left) has up to three sections, with
labels in blue on the left: the **status** (current tool and its value, brush size, simulation
speed, speed of light, numbers of electrons, islands and test charges, mean |E| inside
conductors), the **cell under the mouse**,
and the **keys**. `H` cycles between showing all of it, only status and cell, and nothing. For
the cell under the mouse it shows:

- the material, island number and electron count (and for batteries, the stack's details),
- the electric potential **V**, field **E**, magnetic field **B** and energy flow **S** at
  its centre,
- for conductors and batteries, the **current** through it: conventional current (the direction positive
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

The page title shows the frame rate, simulation time, paused state, grid size (columns x rows)
and where the physics runs (GPU or CPU).

## Under the hood

### Structure

Three Rust crates, plus the WGSL shaders and a small HTML page:

- [embloks-core](crates/embloks-core/): everything that doesn't need a browser or a GPU: the
  world being painted (materials, electrons, batteries, islands), the physics as a
  double-precision reference implementation, the example scenes, saves, and the app's state and
  input handling (keys, strokes, the fixed time step, the info panel's content).
- [embloks-gpu](crates/embloks-gpu/): the same physics in WebGPU compute shaders
  ([shaders/](crates/embloks-gpu/src/shaders/)), and the renderer. The CPU keeps the world for
  painting; electrons, fields and battery gains live on the GPU between edits, and a frame's
  ticks are recorded without reading anything back. Painting or saving first reads the electrons
  back (about a frame); the info panel's numbers come back in a small block every frame.
- [embloks-web](crates/embloks-web/): the browser front end: canvas and WebGPU set-up, mouse,
  touch and keyboard, the info panel, layers panel and toolbar as HTML over the canvas, and save
  slots.

### Libraries

- **[wgpu](https://wgpu.rs/)**: WebGPU from Rust, in the browser and natively (the GPU tests run
  through Vulkan, Metal or DX12).
- **[wasm-bindgen](https://rustwasm.github.io/wasm-bindgen/)** and
  **[web-sys](https://rustwasm.github.io/wasm-bindgen/web-sys/)**: the browser's APIs from Rust.
- **[ndarray](https://github.com/rust-ndarray/ndarray)**: arrays for the CPU reference.
- **[rustdct](https://github.com/ejmahler/rust_dct)**: sine transforms for Gauss's law on the
  CPU (the GPU multiplies by sine matrices instead).
- **[serde](https://serde.rs/)**, **[postcard](https://github.com/jamesmunns/postcard)** and
  **[miniz_oxide](https://github.com/Frommi/miniz_oxide)**: saves.
- **[Trunk](https://trunkrs.dev/)**: builds the site (no JavaScript tooling needed).

### How the fields are computed

Summing the force of every charge on every electron would cost N² operations per step, and
would still miss induction and waves. Instead, Embloks puts the charges and currents on the grid
and steps Maxwell's equations there (a particle-in-cell method), which costs about the same per
cell however many electrons there are. The grid smooths every charge over a couple of cells, so
electrons closer than `electron_cutoff` are also paired up directly, adding the difference.
[docs/fields.md](docs/fields.md) explains it step by step, with pictures of an example scene,
along with the other techniques (stable drag, substeps, the absorbing border, single precision
on the GPU).

### Performance

Time per physics tick, with 30% of the canvas filled with wire and 8 electrons per cell:

| Grid      | Electrons | CPU reference |
|-----------|-----------|---------------|
| 40 × 32   | ~3,000    | ~2.4 ms       |
| 100 × 80  | ~19,000   | ~15 ms        |
| 200 × 160 | ~76,000   | ~59 ms        |

(One core; the default 8× speed runs 4 ticks per frame.) The GPU runs every electron and every
cell in parallel; measure it on your machine with

```bash
cargo run --release -p embloks-gpu --example bench
```

which times both on the same scenes. If the GPU can't keep up (large grids, high speeds), at
most two frames of work wait for it and the simulation runs slower than chosen instead of
freezing.
