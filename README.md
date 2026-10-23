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
| `1` / `2` / `3`     | Tool: vacuum / conductor / battery                |
| Up / Down           | Conductor: resistance, battery: EMF (voltage)     |
| `R`                 | Rotate battery orientation (clockwise)            |
| `[` / `]` or wheel  | Smaller / larger brush                            |
| `C`                 | Clear the canvas                                  |
| `P`                 | Hide / show particles (ions and electrons)        |
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
- **electrons** (blue dots) that will be free to move within their island once physics is
  added. By default there is one electron per cell; `carriers_per_cell` in
  [config.py](src/embloks/config.py) sets more, each carrying a fraction -e/n so cells stay neutral.

Painting a conductive cell adds its ion and electrons. Erasing a cell removes its ion and the same
number of electrons from that island: first electrons inside the erased cell, then the nearest
ones. If more electrons than that were in the erased cell, the extras are pushed into the nearest
remaining cell of the island (or disappear if the whole island is erased). Repainting a cell with
another conductive material (e.g. changing resistance) keeps its charges.

Ion rings are only drawn when cells are large enough; on fine grids electrons become single pixels.

The brush outline follows the mouse. The info panel (top-left) shows the current tool, its value,
the brush size, the number of electrons and islands, and what is in the cell under the mouse
(material, island number and electron count).

The window title shows the frame rate, simulation time, paused state and grid size
(columns x rows).
