"""Batteries as ideal voltage sources with a small internal resistance.

A battery cell blocks electrons on all sides. Battery cells lined up along their
orientation axis form one *stack*; the conductor cells just beyond its two ends
are its terminals. Every tick the stack measures the voltage between its
terminals and moves electrons from the + terminal to the - terminal at

    current = (EMF - V) / internal_resistance

so it holds V = EMF exactly when no current flows, and V = EMF - I * R_internal
under load: a textbook voltage source. Physically this stands in for the ions
that carry charge through a real battery's electrolyte: electrons don't cross
it, charge is taken from one electrode and delivered at the other.
"""

from dataclasses import dataclass

import numpy as np

from embloks.config import Settings
from embloks.materials import Material, Materials, Orientation

# Electrons delivered at a terminal land this close to its cell centre.
_DELIVERY_JITTER = 0.02


@dataclass(frozen=True)
class Stack:
    cells: tuple[tuple[int, int], ...]
    # Terminal cells before the first and after the last battery cell along the axis.
    low: tuple[int, int]
    high: tuple[int, int]
    # Voltage the stack maintains: V(high) - V(low). Negative if it points towards low.
    emf: float
    internal_resistance: float


def find_stacks(materials: Materials, resistance_per_cell: float) -> list[Stack]:
    """Group battery cells into stacks: runs along their own (shared) axis."""
    kind, orientation, emf = materials.kind, materials.orientation, materials.emf
    horizontal = (orientation == Orientation.RIGHT) | (orientation == Orientation.LEFT)

    def in_run(cell: tuple[int, int], axis_horizontal: bool) -> bool:
        c, r = cell
        return (
            0 <= c < materials.cols
            and 0 <= r < materials.rows
            and kind[c, r] == Material.BATTERY
            and horizontal[c, r] == axis_horizontal
        )

    stacks, seen = [], set()
    for c, r in zip(*np.nonzero(kind == Material.BATTERY)):
        start = (int(c), int(r))
        if start in seen:
            continue
        axis_horizontal = bool(horizontal[start])
        step = (1, 0) if axis_horizontal else (0, 1)
        # Walk back to the first cell of the run, then collect it forwards.
        first = start
        while in_run((first[0] - step[0], first[1] - step[1]), axis_horizontal):
            first = (first[0] - step[0], first[1] - step[1])
        cells, cell = [], first
        while in_run(cell, axis_horizontal):
            cells.append(cell)
            cell = (cell[0] + step[0], cell[1] + step[1])
        seen.update(cells)
        # RIGHT and DOWN point along +step; LEFT and UP against it.
        signs = [1 if orientation[x] in (Orientation.RIGHT, Orientation.DOWN) else -1 for x in cells]
        total = float(sum(sign * emf[x] for sign, x in zip(signs, cells)))
        low = (first[0] - step[0], first[1] - step[1])
        stacks.append(Stack(tuple(cells), low, cell, total, resistance_per_cell * len(cells)))
    return stacks


def terminal_potential(world, potential: np.ndarray, cell: tuple[int, int]) -> float | None:
    """Mean potential over a terminal cell and its 3x3 neighbours in the same island.

    None if the terminal isn't a conductor. Averaging damps the noise from
    individual electrons passing right by the cell centre.
    """
    m, labels = world.materials, world.connectivity.labels
    c, r = cell
    if not (0 <= c < m.cols and 0 <= r < m.rows) or labels[c, r] < 0:
        return None
    c0, c1 = max(c - 1, 0), min(c + 2, m.cols)
    r0, r1 = max(r - 1, 0), min(r + 2, m.rows)
    same_island = labels[c0:c1, r0:r1] == labels[c, r]
    return float(potential[c0:c1, r0:r1][same_island].mean())


def pump(world, potential: np.ndarray, dt: float, settings: Settings) -> None:
    """Run every battery stack for dt, moving electrons between its terminals."""
    p = world.particles
    for stack in world.stacks:
        v_low = terminal_potential(world, potential, stack.low)
        v_high = terminal_potential(world, potential, stack.high)
        if v_low is None or v_high is None:
            world.pump_charge.pop(stack, None)
            world.pump_current.pop(stack, None)
            continue
        current = (stack.emf - (v_high - v_low)) / stack.internal_resistance
        # Positive current = conventional current low -> high inside the battery,
        # i.e. electrons taken at the high terminal and delivered at the low one.
        owed = world.pump_charge.get(stack, 0.0) + current * dt
        quantum = settings.cell_charge / settings.carriers_per_cell
        moved = 0.0
        while abs(owed) >= quantum:
            source, target = (stack.high, stack.low) if owed > 0 else (stack.low, stack.high)
            if not _move_electron(world, source, target):
                owed = 0.0  # Source side has no electrons left; don't build up a debt.
                break
            step = np.sign(owed) * quantum
            owed -= step
            moved += step
        world.pump_charge[stack] = owed
        # The battery's current is the charge it actually moved, smoothed like wire
        # currents. (The commanded rate above swings wildly between single-electron
        # transfers and is not a current.)
        alpha = min(1.0, dt / settings.current_smoothing_time)
        previous = world.pump_current.get(stack, 0.0)
        world.pump_current[stack] = previous + alpha * (moved / dt - previous)
        if len(p) == 0:
            break


def current_grid(world) -> np.ndarray:
    """The batteries' internal currents as a current density, shape (cols, rows, 2).

    Pumped electrons jump across a battery instead of moving through it, so they
    carry no velocity; this puts the (smoothed) pump current back along the
    stack's cells, closing the current loop for magnetic fields.
    """
    m = world.materials
    grid = np.zeros((m.cols, m.rows, 2))
    for stack, current in world.pump_current.items():
        # Positive current flows from the low terminal towards the high one.
        direction = np.sign(np.subtract(stack.high, stack.low))
        for cell in stack.cells:
            grid[cell] += current * direction
    return grid


def _move_electron(world, source: tuple[int, int], target: tuple[int, int]) -> bool:
    """Move the electron nearest the source terminal (along the wire) to the target."""
    p = world.particles
    if len(p) == 0:
        return False
    cols, rows = p.cells()
    steps = world.terminal_distance[source][cols, rows]
    # Path distance first; straight-line distance only breaks ties within a cell.
    straight = ((p.pos - (np.array(source) + 0.5)) ** 2).sum(axis=1)
    chosen = np.lexsort((straight, steps))[0]
    if not np.isfinite(steps[chosen]):
        return False
    p.pos[chosen] = np.array(target) + 0.5 + world.rng.uniform(-_DELIVERY_JITTER, _DELIVERY_JITTER, 2)
    p.vel[chosen] = 0.0
    return True
