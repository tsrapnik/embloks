"""Batteries as voltage sources with a small internal resistance.

A battery cell conducts like a wire, with its own ion and electrons, but it also
pushes its electrons towards its - side with an *impressed field*: a
non-electrostatic force of EMF per cell along its axis (Griffiths' source force
f_s, the "resistive voltage source" of FDTD). Electrons drift through the
battery with drag, so they keep moving until the charge they pile up on the
terminals cancels that force, which is when V(+) - V(-) = EMF. Under load the
battery cells' own drag drops part of it: V = EMF - I * R_internal, a textbook
voltage source. Charge only ever moves continuously, so the current through a
battery is carried by real electron velocities, like in any wire.

Battery cells lined up along their orientation axis form one *stack*; the wire
cells just beyond its two ends are its terminals. With so few electrons per cell
the impressed field alone is not a steady source: the voltage rests on the
handful of electrons at the battery's faces, it comes up short, and it jitters
as they shift. So each stack measures what it delivers, V + I * R_internal, and
keeps scaling its impressed field until that equals the EMF (see trim): a
feedback loop on the force, not on the charge.
"""

from dataclasses import dataclass

import numpy as np

from embloks.config import Settings
from embloks.materials import ORIENTATION_VECTORS, Material, Materials, Orientation


@dataclass(frozen=True)
class Stack:
    cells: tuple[tuple[int, int], ...]
    # Terminal cells before the first and after the last battery cell along the axis.
    low: tuple[int, int]
    high: tuple[int, int]
    # Voltage the stack maintains: V(high) - V(low). Negative if it points towards low.
    # For AC cells this is the amplitude.
    emf: float
    internal_resistance: float
    # Per cell: (signed EMF, frequency). Frequency 0 is DC.
    sources: tuple[tuple[float, float], ...] = ()

    @property
    def is_ac(self) -> bool:
        return any(frequency > 0 for _, frequency in self.sources)

    def emf_at(self, time: float) -> float:
        """V(high) - V(low) the stack drives at `time`: AC cells follow EMF * sin(2 pi f t)."""
        if not self.sources:
            return self.emf
        return sum(
            emf if frequency == 0 else emf * np.sin(2 * np.pi * frequency * time)
            for emf, frequency in self.sources
        )


def find_stacks(materials: Materials) -> list[Stack]:
    """Group battery cells into stacks: runs along their own (shared) axis."""
    kind, orientation, emf, frequency = (
        materials.kind, materials.orientation, materials.emf, materials.frequency
    )
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
        sources = tuple((float(sign * emf[x]), float(frequency[x])) for sign, x in zip(signs, cells))
        total = sum(signed for signed, _ in sources)
        low = (first[0] - step[0], first[1] - step[1])
        stacks.append(
            Stack(tuple(cells), low, cell, total,
                  float(sum(materials.resistance[x] for x in cells)), sources)
        )
    return stacks


def terminal_potential(world, potential: np.ndarray, cell: tuple[int, int]) -> float | None:
    """Mean potential over a terminal cell and its 3x3 neighbours of wire in the same island.

    None if the terminal isn't a wire. Averaging damps the noise from individual
    electrons passing right by the cell centre. Battery cells are left out: the
    potential climbs through them.
    """
    if not _wired(world, cell):
        return None
    m, labels = world.materials, world.connectivity.labels
    c, r = cell
    c0, c1 = max(c - 1, 0), min(c + 2, m.cols)
    r0, r1 = max(r - 1, 0), min(r + 2, m.rows)
    wire = (labels[c0:c1, r0:r1] == labels[c, r]) & (m.kind[c0:c1, r0:r1] == Material.CONDUCTOR)
    return float(potential[c0:c1, r0:r1][wire].mean())


def _wired(world, cell: tuple[int, int]) -> bool:
    """Whether a terminal cell is a wire (a battery only works with both wired)."""
    m = world.materials
    c, r = cell
    return 0 <= c < m.cols and 0 <= r < m.rows and m.kind[c, r] == Material.CONDUCTOR


def probe_cells(world) -> np.ndarray:
    """Mask (cols, rows) of the cells whose potential terminal_potential reads."""
    m = world.materials
    mask = np.zeros((m.cols, m.rows), dtype=bool)
    for stack in world.stacks:
        for c, r in (stack.low, stack.high):
            mask[max(c - 1, 0) : c + 2, max(r - 1, 0) : r + 2] = True
    return mask


def impressed_field(world, cols: np.ndarray, rows: np.ndarray, time: float) -> np.ndarray:
    """The batteries' impressed field at time `time` for electrons in the given cells, (N, 2).

    EMF per cell towards each battery cell's + side, so the force on an electron
    (charge < 0) points to its - side; zero outside batteries. AC cells follow
    EMF * sin(2 pi f t). Over a cell of length 1 it adds EMF to the integral of
    the field, times the stack's gain (see trim).
    """
    materials = world.materials
    gain = np.ones((materials.cols, materials.rows))
    for stack, value in world.battery_gain.items():
        gain[tuple(np.array(stack.cells).T)] = value
    emf = materials.emf[cols, rows] * gain[cols, rows]
    frequency = materials.frequency[cols, rows]
    emf = np.where(frequency > 0, emf * np.sin(2 * np.pi * frequency * time), emf)
    return emf[:, None] * ORIENTATION_VECTORS[materials.orientation[cols, rows]]


def stack_current(world, stack: Stack, current: np.ndarray) -> float:
    """Current through a stack, from low terminal to high, given currents per cell (cols, rows, 2).

    Averaged over the battery cells and its wired terminals: electrons bunch up
    where they leave the battery, so a single cell over- or under-reads.
    """
    direction = np.sign(np.subtract(stack.high, stack.low))
    cells = [*stack.cells, *(t for t in (stack.low, stack.high) if _wired(world, t))]
    return float(np.mean([current[cell] @ direction for cell in cells]))


def trim(world, potential: np.ndarray, current: np.ndarray, dt: float, settings: Settings,
         time: float = 0.0) -> None:
    """Nudge every wired stack's gain so it delivers its EMF: V + I * R_internal = EMF.

    `potential` and `current` (per cell, (cols, rows, 2)) are the present ones. The
    gain integrates the shortfall over battery_trim_time, weighted by the EMF
    itself (a least-mean-squares fit), so it works for AC sources as well: near
    a zero crossing, where the shortfall says little, it hardly moves.
    """
    for stack in world.stacks:
        v_low = terminal_potential(world, potential, stack.low)
        v_high = terminal_potential(world, potential, stack.high)
        amplitude = sum(abs(emf) for emf, _ in stack.sources)
        if v_low is None or v_high is None or amplitude == 0:
            world.battery_gain.pop(stack, None)
            continue
        target = stack.emf_at(time)
        delivered = v_high - v_low + stack_current(world, stack, current) * stack.internal_resistance
        gain = world.battery_gain.get(stack, 1.0)
        gain += dt / settings.battery_trim_time * (target - delivered) * target / amplitude**2
        world.battery_gain[stack] = float(np.clip(gain, *settings.battery_gain_range))
