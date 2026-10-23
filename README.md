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

| Key       | Action                                   |
|-----------|------------------------------------------|
| Space     | Pause / resume physics                   |
| `+` / `-` | Finer / coarser grid (resets the scene)  |
| Esc       | Quit                                     |

The canvas is divided into a grid of square cells. Hover the mouse over the grid to highlight a
cell. The grid density (about how many columns fit across the window, 4 to 200) is set by
`grid_density` in [src/embloks/config.py](src/embloks/config.py) and changed live with `+`/`-`.
On very fine grids the grid lines are hidden so they don't drown out the canvas.

The window title shows the frame rate, simulation time, paused state, grid size (columns x rows)
and the hovered cell as (column, row).
