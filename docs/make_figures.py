"""Draw the charge distribution figures in docs/images/ (used by README.md).

Needs matplotlib, which the app itself doesn't: pip install matplotlib
Run from the repository root: python docs/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from embloks import battery, physics, scenes
from embloks.config import Settings
from embloks.fields import CloudInCell, ElectricField, GridKernel, _close_field
from embloks.materials import Material
from embloks.world import World

OUT = Path(__file__).parent / "images"
S = Settings()
EPS = S.conductor_softening
EPS_E = S.electron_softening
# One-dimensional cloud-in-cell weights of an evenly spread cell: 1/8, 3/4, 1/8.
SPREAD = np.array([1 / 8, 3 / 4, 1 / 8])

plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white", "font.size": 10})


def plummer(x, y, cx, cy, eps):
    """In-plane slice of the density of a unit charge smeared with softening eps.

    This (Plummer) density is exactly the one whose field is k q r / (r^2 + eps^2)^1.5.
    """
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    return 3 / (4 * np.pi * eps**3) * (1 + r2 / eps**2) ** -2.5


def cell_lines(ax, lo=-1.5, hi=1.5):
    for v in np.arange(np.ceil(lo + 0.5), hi, 1.0) - 0.5:
        ax.axhline(v, color="w", lw=0.6, alpha=0.6)
        ax.axvline(v, color="w", lw=0.6, alpha=0.6)


def outline_cell(ax, color="c"):
    ax.add_patch(Rectangle((-0.5, -0.5), 1, 1, fill=False, ec=color, lw=2))


# Coordinates in cells, the cell of interest centred on (0, 0).
x, y = np.meshgrid(np.linspace(-1.5, 1.5, 301), np.linspace(-1.5, 1.5, 301), indexing="xy")
EXTENT = (-1.5, 1.5, 1.5, -1.5)


def ion_figure():
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    # (a) What the grid stores: the ion's charge per cell centre.
    stencil = np.outer(SPREAD, SPREAD)
    axs[0].imshow(stencil.T, extent=EXTENT, cmap="inferno", vmin=0, vmax=0.6)
    for (i, j), v in np.ndenumerate(stencil):
        axs[0].text(i - 1, j - 1, f"{v * 64:.0f}/64", ha="center", va="center",
                    color="k" if v > 0.3 else "w", fontsize=11)
    outline_cell(axs[0])
    axs[0].set_title("1. Grid: share of the ion's charge\nput on each cell centre")
    # (b) Long range: those shares, each smeared with conductor_softening.
    grid = sum(stencil[i, j] * plummer(x, y, i - 1, j - 1, EPS) for i in range(3) for j in range(3))
    im = axs[1].imshow(grid, extent=EXTENT, cmap="inferno", vmin=0)
    cell_lines(axs[1])
    outline_cell(axs[1])
    fig.colorbar(im, ax=axs[1], label="charge density (per unit charge)")
    axs[1].set_title(f"2. Long range (grid fields):\nshares smeared with softening {EPS}")
    # (c) Close range: a uniform square.
    square = ((np.abs(x) < 0.5) & (np.abs(y) < 0.5)).astype(float)
    axs[2].imshow(square, extent=EXTENT, cmap="inferno", vmin=0, vmax=1.6)
    cell_lines(axs[2])
    outline_cell(axs[2])
    axs[2].set_title("3. Close range (within electron_cutoff):\ncharge spread evenly over the cell")
    for ax in axs:
        ax.set_xlabel("x (cells)")
    axs[0].set_ylabel("y (cells)")
    fig.suptitle("A conductor cell's ion (+4 e)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "ion.png", dpi=80)


def electron_figure():
    positions = [((0.0, 0.0), "at the cell centre"), ((0.25, 0.0), "0.25 cell off centre"),
                 ((0.5, 0.0), "on a cell edge"), ((0.5, 0.5), "on a cell corner")]
    fig, axs = plt.subplots(2, 4, figsize=(17, 8.4))
    top = plummer(0, 0, 0, 0, EPS)
    for k, ((px, py), label) in enumerate(positions):
        # Long range: cloud-in-cell shares on the 4 nearest centres, each smeared like the ions.
        cic = CloudInCell(np.array([[px + 2.5, py + 2.5]]), 5, 5)
        grid = np.zeros_like(x)
        for index, weight in zip(cic.index[:, 0], cic.weight[:, 0]):
            c, r = divmod(int(index), 5)
            grid += weight * plummer(x, y, c - 2, r - 2, EPS)
        ax = axs[0, k]
        ax.imshow(grid, extent=EXTENT, cmap="viridis", vmin=0, vmax=top)
        cell_lines(ax)
        ax.plot(px, py, "r+", ms=14, mew=2)
        ax.set_title(f"Electron {label}")
        # Close range: a compact ball at the electron itself.
        ax = axs[1, k]
        ax.imshow(plummer(x, y, px, py, EPS_E), extent=EXTENT, cmap="viridis", vmin=0,
                  vmax=plummer(0, 0, 0, 0, EPS_E))
        cell_lines(ax)
        ax.plot(px, py, "r+", ms=14, mew=2)
    axs[0, 0].set_ylabel(f"Long range (grid fields)\ncloud-in-cell, softening {EPS}", fontsize=11)
    axs[1, 0].set_ylabel(f"Close range (within {S.electron_cutoff:g} cell)\nsoftening {EPS_E}",
                         fontsize=11)
    fig.suptitle("One electron (-0.5 e); red +: its position. The colour scale of each row is "
                 "fixed; the close-range ball is ~30x denser at its peak.", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "electron.png", dpi=80)


def pair_force(separations, centre=12.5, n=24):
    """Force between two electrons (charge -0.5 each) along x, through the grid only."""
    kernel = GridKernel(n, n, EPS)
    q = -S.cell_charge / S.carriers_per_cell
    out = []
    for d in separations:
        a, b = (centre + d / 2, 12.5), (centre - d / 2, 12.5)
        field = kernel.field(CloudInCell(np.array([b]), n, n).deposit(np.array([q])))
        out.append(q * CloudInCell(np.array([a]), n, n).gather(field)[0, 0])
    return np.array(out)


def run_scene(cutoff, seed=0):
    s = Settings(electron_cutoff=cutoff)
    w = World(40, 32, s, np.random.default_rng(seed))
    scenes.battery_and_resistor(w)
    p = w.particles
    # Start every electron on its cell centre, the worst case, to show what the physics does.
    p.pos = np.floor(p.pos) + 0.5 + w.rng.uniform(-0.02, 0.02, p.pos.shape)
    electric = ElectricField(s)
    for i in range(1500):
        battery.pump(w, electric.potential_grid(w, battery.probe_cells(w)), s.dt, s, i * s.dt)
        physics.step(w, electric, s.dt, s)
    return w


def behaviour_figure():
    q2 = (S.cell_charge / S.carriers_per_cell) ** 2
    seps = np.linspace(0, 2, 81)
    grid = pair_force(seps)
    close = q2 * _close_field(np.stack([seps, 0 * seps], -1), EPS_E, EPS, S.electron_cutoff)[:, 0]
    fig = plt.figure(figsize=(16, 5.6))
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(seps, grid, label="grid fields only")
    ax.plot(seps, grid + close, label="grid + close range (used)")
    ax.plot(seps, q2 * seps / (seps**2 + EPS_E**2) ** 1.5, ":", label=f"Coulomb, softening {EPS_E}")
    ax.set_xlabel("distance between the electrons (cells)")
    ax.set_ylabel("repulsion")
    ax.set_ylim(0, 2.2)
    ax.legend()
    ax.set_title("Force between two electrons")

    for k, (cutoff, title) in enumerate(((0.0, "Close range off (electron_cutoff = 0)"),
                                         (S.electron_cutoff, "Close range on (default)"))):
        w = run_scene(cutoff)
        ax = fig.add_subplot(1, 3, k + 2)
        for c, r in np.argwhere(w.materials.conductive):
            ax.add_patch(Rectangle((c, r), 1, 1, fc="#e8a060", ec="#7a5030", lw=0.5))
        for c, r in np.argwhere(w.materials.kind == Material.BATTERY):
            ax.add_patch(Rectangle((c, r), 1, 1, fc="#556655", ec="k", lw=0.5))
        pos = w.particles.pos
        ax.scatter(pos[:, 0], pos[:, 1], s=8, c="#1f5fd0", zorder=3)
        ax.set_xlim(9, 17)
        ax.set_ylim(17, 7)
        ax.set_aspect("equal")
        ax.set_xticks(range(9, 18))
        ax.set_yticks(range(7, 18))
        ax.grid(alpha=0.3)
        ax.set_title(title)
    fig.suptitle("Battery & resistor scene (F1), top-left corner, 50 time units after starting "
                 "with every electron on its cell centre", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "electron_spacing.png", dpi=80)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ion_figure()
    electron_figure()
    behaviour_figure()
    print(f"Wrote figures to {OUT}")
