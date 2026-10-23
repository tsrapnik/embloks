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
connected. Dragging the brush always paints an edge-connected path, so a quick diagonal stroke
still gives an unbroken wire.

The brush outline follows the mouse. The info panel (top-left) shows the current tool, its value,
the brush size, and what is in the cell under the mouse.

The window title shows the frame rate, simulation time, paused state and grid size
(columns x rows).
