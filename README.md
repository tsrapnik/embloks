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

Run the tests with `pytest`.

## Using the app

| Key   | Action                  |
|-------|-------------------------|
| Space | Pause / resume physics  |
| Esc   | Quit                    |

The window title shows the frame rate, the simulation time and whether the simulation is paused.
For now the canvas is empty; more features arrive with each build step.
