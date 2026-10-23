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

| Input               | Action                                            |
|---------------------|---------------------------------------------------|
| Left mouse (drag)   | Paint with the current tool                       |
| Right mouse (drag)  | Erase (paint vacuum)                              |
| `1` - `4`           | Tool: vacuum / conductor / battery / test charge  |
| Up / Down           | Tool value: resistance / EMF / charge             |
| `R`                 | Rotate battery orientation (clockwise)            |
| `[` / `]` or wheel  | Smaller / larger brush                            |
| `C`                 | Clear the canvas                                  |
| `P`                 | Hide / show particles (ions and electrons)        |
| `E`                 | Hide / show electric field arrows                 |
| `H`                 | Hide / show the info panel                        |
| `+` / `-`           | Finer / coarser grid (resets the scene)           |
| Space               | Pause / resume physics                            |
| Esc                 | Quit                                              |

### Grid

The canvas is divided into a grid of square cells. The grid density (about how many columns fit
across the window, 4 to 200) is set by `grid_density` in
[src/embloks/config.py](src/embloks/config.py) and changed live with `+`/`-`, which clears the
canvas. On very fine grids the grid lines are hidden so they don't drown out the canvas.

### Materials

Every cell is one of:

- **Vacuum** (dark): empty space.
- **Conductor** (copper): carries charge. Each conductor cell has a resistance, picked from
  0.1 to 100 with Up/Down before painting. Low-resistance cells are bright copper, high-resistance
  cells dark brown. Painting over a conductor replaces its resistance.
- **Battery** (grey, red and blue stripes): an ideal voltage source. The red stripe/`+` marks the
  positive terminal, the blue stripe/`-` the negative one; only these two faces connect to
  neighbouring cells. Set the EMF (1 to 20) with Up/Down and the direction of the `+` side with `R`
  before painting. Batteries placed end to end add up like cells in series.

Cells only connect through shared edges: two cells touching only at a corner are *not*
connected, and a battery's two side faces are walls. Dragging the brush always paints an
edge-connected path, so a quick diagonal stroke still gives an unbroken wire. A group of connected
cells is called an *island*; charge can never leave its island.

### Charges

Every conductive cell (conductor or battery) holds:

- one fixed **positive ion** (red ring) at its centre, with charge +e, and
- **electrons** (blue dots) that move freely within their island (see *Electron motion*).
  By default there is one electron per cell; `carriers_per_cell` in
  [config.py](src/embloks/config.py) sets more, each carrying a fraction -e/n so cells stay neutral.

Painting a conductive cell adds its ion and electrons. Erasing a cell removes its ion and the same
number of electrons from that island: first electrons inside the erased cell, then the nearest
ones. If more electrons than that were in the erased cell, the extras are pushed into the nearest
remaining cell of the island (or disappear if the whole island is erased). Repainting a cell with
another conductive material (e.g. changing resistance) keeps its charges.

Ion rings are only drawn when cells are large enough; on fine grids electrons become single pixels.

### Test charges

Tool `4` places fixed **test charges** (red with `+`, blue with `-`) at cell centres, anywhere on
the canvas, including on top of conductors. Pick the charge (-10 e to +10 e) with Up/Down. A larger
brush paints a whole charged plate, one charge per cell. With this tool, right-click removes test
charges and leaves materials alone. Test charges never move; use them to poke at conductors.

### Electric field

Yellow arrows show the electric field **E** at cell centres: the force a positive charge would
feel there. It is the sum of the fields of all charges (ions, electrons and test charges), using
Coulomb's law. Both arrow length and brightness grow with the *logarithm* of the field strength
(between `e_field_min` and `e_field_max` in [config.py](src/embloks/config.py)), because fields
near a charge are many times stronger than fields far away. On fine grids only every n-th cell
gets an arrow, so arrows don't overlap. The info panel shows the exact E value under the mouse.

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
(vacuum, a battery's side faces, the edge of the canvas), losing part of their speed
(`wall_restitution`). Fast electrons are moved in several small substeps, so they can't skip
through a one-cell-wide wall.

What to try: place a test charge next to a conductor. Electrons flow towards a positive charge
(away from a negative one) until the field *inside* the conductor is nearly zero. This is
**screening**, and it's why field lines always meet a conductor's surface at right angles. The info
panel shows the mean |E| over all conductor cells, which drops as the electrons settle. It
never reaches exactly zero: one electron per cell is grainy, so some small arrows of random
direction remain. More `carriers_per_cell` makes it smoother.

Newly painted cells get their electrons right on top of their ions, so a fresh conductor is
neutral everywhere and has no field until something disturbs it.

Time: the simulation runs at about one time unit per real second (`dt` per tick in config.py).

The brush outline follows the mouse. The info panel (top-left) shows the current tool, its value,
the brush size, the number of electrons and islands, and what is in the cell under the mouse
(material, island number and electron count).

The window title shows the frame rate, simulation time, paused state and grid size
(columns x rows).

## Under the hood

### Libraries

- **[pygame-ce](https://pyga.me/)**: window, mouse/keyboard input and drawing.
- **[NumPy](https://numpy.org/)**: all simulation state lives in arrays (one entry per cell or
  per electron), so every step is a handful of vectorised operations instead of Python loops.
- **[SciPy](https://scipy.org/)**: FFTs for the field calculation, and connected-component
  labelling to find islands.
- **[pytest](https://pytest.org/)**: tests, including physics checks (Coulomb's law, screening,
  electrons never escaping their island).

### Techniques and optimizations

- **Charge smoothing (softening).** Each charge is treated as a small blurred cloud rather than
  a point: `E = k·q·r / (r² + ε²)^1.5`. This keeps fields finite when charges get close, makes a
  charge's field at its own centre zero (so nothing pushes itself), and smooths the ions into an
  even positive background that electrons can glide over. Ions and electrons share the same ε, so
  an electron sitting on its ion cancels it exactly.
- **Fields via FFT.** Summing every charge's field at every cell costs
  `cells × charges` operations. Instead, the charges are put on the grid and the field is
  computed as a convolution with the Coulomb kernel using FFTs, which costs about
  `cells × log(cells)` no matter how many electrons there are. The kernel's FFT is computed once
  per grid size.
- **Cloud-in-cell.** Electrons sit anywhere, not just on cell centres, so each one's charge is
  shared over the 4 nearest cell centres in proportion to how close it is. Each electron reads
  the field back with exactly the same weights. This symmetry means an electron feels no force
  from itself, and every push between two electrons is equal and opposite.
- **Caching.** Ions and test charges only change when you paint, so their field is computed once
  and reused until then. Each tick only the electrons' part is recomputed. The material image is
  likewise only redrawn after painting.
- **Stable drag.** Resistance is applied implicitly, `v ← (v + a·dt) / (1 + Γ·dt)`, which stays
  stable for any resistance; a simple explicit update would blow up at high resistance.
- **Substeps.** If electrons move fast, a tick is split into smaller steps so that nothing moves
  more than 0.4 cell at a time and can't tunnel through a one-cell wall.
- **Drawing.** On fine grids only every n-th cell gets a field arrow, and with many electrons they
  are drawn as single pixels written straight into the screen array.

The result: on a 200×160 grid with ~10,000 electrons a physics tick takes ~3 ms and a frame
~20 ms.
