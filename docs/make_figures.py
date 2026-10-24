"""Draw the figures in docs/images/ (used by docs/fields.md).

Needs matplotlib, which the app itself doesn't: pip install matplotlib
Run from the repository root: python docs/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree

from embloks import battery, physics, scenes
from embloks.config import Settings
from embloks.fields import (CloudInCell, ElectricField, GridKernel, _close_field, coulomb_field,
                            field_kernels, softened_inv_r3)
from embloks.materials import Material
from embloks.world import World

OUT = Path(__file__).parent / "images"
S = Settings()
EPS = S.conductor_softening
EPS_E = S.electron_softening
# One-dimensional cloud-in-cell weights of an evenly spread cell: 1/8, 3/4, 1/8.
SPREAD = np.array([1 / 8, 3 / 4, 1 / 8])

plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white", "font.size": 10})


def gaussian(x, y, cx, cy, eps):
    """In-plane slice of the density of a unit charge smeared with softening eps.

    This Gaussian density is exactly the one whose potential is k q erf(r / eps) / r.
    """
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    return np.exp(-r2 / eps**2) / (np.pi**1.5 * eps**3)


def softened_force(r, eps):
    """Repulsion between two unit charges at distance r, one smeared with softening eps."""
    return r * softened_inv_r3(r**2, eps)


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
    grid = sum(stencil[i, j] * gaussian(x, y, i - 1, j - 1, EPS) for i in range(3) for j in range(3))
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
    top = gaussian(0, 0, 0, 0, EPS)
    for k, ((px, py), label) in enumerate(positions):
        # Long range: cloud-in-cell shares on the 4 nearest centres, each smeared like the ions.
        cic = CloudInCell(np.array([[px + 2.5, py + 2.5]]), 5, 5)
        grid = np.zeros_like(x)
        for index, weight in zip(cic.index[:, 0], cic.weight[:, 0]):
            c, r = divmod(int(index), 5)
            grid += weight * gaussian(x, y, c - 2, r - 2, EPS)
        ax = axs[0, k]
        ax.imshow(grid, extent=EXTENT, cmap="viridis", vmin=0, vmax=top)
        cell_lines(ax)
        ax.plot(px, py, "r+", ms=14, mew=2)
        ax.set_title(f"Electron {label}")
        # Close range: a compact ball at the electron itself.
        ax = axs[1, k]
        ax.imshow(gaussian(x, y, px, py, EPS_E), extent=EXTENT, cmap="viridis", vmin=0,
                  vmax=gaussian(0, 0, 0, 0, EPS_E))
        cell_lines(ax)
        ax.plot(px, py, "r+", ms=14, mew=2)
    axs[0, 0].set_ylabel(f"Long range (grid fields)\ncloud-in-cell, softening {EPS}", fontsize=11)
    axs[1, 0].set_ylabel(f"Close range (within {S.electron_cutoff:g} cells)\nsoftening {EPS_E}",
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
    seps = np.linspace(0, 2.5, 101)
    grid = pair_force(seps)
    close = q2 * _close_field(np.stack([seps, 0 * seps], -1), EPS_E, EPS, S.electron_cutoff)[:, 0]
    fig = plt.figure(figsize=(16, 5.6))
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(seps, grid, label="grid fields only")
    ax.plot(seps, grid + close, label="grid + close range (used)")
    ax.plot(seps, q2 * softened_force(seps, EPS_E), ":", label=f"Coulomb, softening {EPS_E}")
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


# The P3M walkthrough (docs/fields.md): 10 electrons in vacuum on a 16 x 16 grid.
N = 16
Q = -S.cell_charge / S.carriers_per_cell
# Electron 0 is the cloud-in-cell example; electrons 1 and 2 are a close pair.
EXAMPLE = np.array([[3.8, 7.1], [10.3, 10.4], [10.75, 10.15], [2.6, 2.3], [7.4, 3.6],
                    [12.9, 2.9], [6.2, 12.8], [13.6, 7.2], [8.1, 7.6], [3.3, 13.4]])
GRID_EXTENT = (0, N, N, 0)


def grid_axes(ax, n=N, lines=True):
    """Cell coordinates with y down, as on screen; optionally the cell boundaries."""
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    if lines:
        for v in range(1, n):
            ax.axhline(v, color="0.75", lw=0.4)
            ax.axvline(v, color="0.75", lw=0.4)


# Spectra span a wide range: a square-root colour scale shows the weak parts too.
SQRT = matplotlib.colors.PowerNorm(0.5)


def spectrum(a):
    """Magnitude of a full 2D spectrum, k = 0 in the middle."""
    return np.abs(np.fft.fftshift(a))


def p3m_split_figure():
    r = np.linspace(0, 2.5, 501)
    compact = softened_force(r, EPS_E)
    smooth = softened_force(r, EPS)
    close = _close_field(np.stack([r, 0 * r], -1), EPS_E, EPS, S.electron_cutoff)[:, 0]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(r, compact, "k:", lw=2, label=f"wanted: compact electrons (softening {EPS_E})")
    ax.plot(r, smooth, lw=2, label=f"PM, far part on the grid: erf (softening {EPS})")
    ax.plot(r, close, lw=2, label="PP, near part pair by pair: erfc remainder")
    ax.plot(r, smooth + close, "--", lw=1.5, label="PM + PP (used)")
    ax.axvline(S.electron_cutoff, color="0.5", lw=1)
    ax.text(S.electron_cutoff + 0.03, 6.5, "cutoff", color="0.4")
    ax.set_xlabel("distance between two charges (cells)")
    ax.set_ylabel("repulsion (unit charges)")
    ax.set_ylim(-0.5, 7)
    ax.legend()
    ax.set_title("Splitting the Coulomb force into a far and a near part")
    fig.tight_layout()
    fig.savefig(OUT / "p3m_split.png", dpi=80)


def p3m_assign_figure(cic, rho):
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.4))
    ax = axs[0]
    grid_axes(ax)
    centres = np.arange(N) + 0.5
    cx, cy = np.meshgrid(centres, centres)
    ax.plot(cx, cy, ".", color="0.6", ms=3)
    ax.scatter(EXAMPLE[:, 0], EXAMPLE[:, 1], s=60, c="#1f5fd0", zorder=3)
    # Electron 0: its 4 nodes and weights.
    x0, y0 = EXAMPLE[0]
    for index, weight in zip(cic.index[:, 0], cic.weight[:, 0]):
        c, r = divmod(int(index), N)
        ax.plot([x0, c + 0.5], [y0, r + 0.5], color="#d03030", lw=1)
        right, below = c + 0.5 > x0, r + 0.5 > y0
        ax.text(c + 0.5 + (0.15 if right else -0.15), r + 0.5 + (0.45 if below else -0.2),
                f"{weight:.2f}", ha="left" if right else "right", color="#d03030", fontsize=11,
                weight="bold")
    for k, (px, py) in enumerate(EXAMPLE):
        ax.text(px + 0.25, py - 0.25, str(k), fontsize=9, color="#1f5fd0")
    ax.set_title("The scene: 10 electrons (blue, numbered), grid nodes at the cell centres\n"
                 "(grey). Red: electron 0 is shared over its 4 nearest nodes")
    ax = axs[1]
    lim = np.abs(rho).max()
    im = ax.imshow(rho.T, extent=GRID_EXTENT, cmap="RdBu_r", vmin=-lim, vmax=lim)
    grid_axes(ax)
    for (c, r), v in np.ndenumerate(rho):
        if abs(v) >= 0.005:
            ax.text(c + 0.5, r + 0.5, f"{v:.2f}".replace("-0.", "-."), ha="center", va="center",
                    fontsize=7, color="w" if v < -0.6 * lim else "k")
    fig.colorbar(im, ax=ax, label="charge on the node (e)", shrink=0.8)
    ax.set_title("Step 1. The charge grid ρ\n(each electron: -0.5 e, shared over 4 nodes)")
    for ax in axs:
        ax.set_xlabel("x (cells)")
        ax.set_ylabel("y (cells)")
    fig.tight_layout()
    fig.savefig(OUT / "p3m_assign.png", dpi=100)


def p3m_kernel_figure(kernel_x, shape):
    n = kernel_x.shape[0] // 2
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.4))
    lim = np.abs(kernel_x).max()
    im = axs[0].imshow(kernel_x.T, extent=(-n - 0.5, n + 0.5, n + 0.5, -n - 0.5), cmap="PuOr",
                       vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=axs[0], shrink=0.85)
    axs[0].set_title("The kernel Kx in real space: the x field of one\n"
                     f"unit charge (softening {EPS}), with the correction")
    axs[0].set_xlabel("x offset from the charge (cells)")
    axs[0].set_ylabel("y offset (cells)")
    # Along the x axis: the softened field itself, and the corrected kernel samples.
    r = np.linspace(0, 6, 301)
    offsets = np.arange(0, 7)
    axs[1].plot(r, softened_force(r, EPS), color="0.4", label=f"softened field (erf, softening {EPS})")
    axs[1].plot(offsets, kernel_x[n + offsets, n], "o-", color="#d06020",
                label="Kx with the assignment correction")
    axs[1].axhline(0, color="0.7", lw=0.8)
    axs[1].set_xlabel("x offset from the charge (cells)")
    axs[1].set_ylabel("field")
    axs[1].legend()
    axs[1].set_title("Along the x axis: the correction boosts the nearest\n"
                     "nodes to make up for cloud-in-cell smearing")
    im = axs[2].imshow(spectrum(np.fft.fft2(kernel_x, s=shape)).T, cmap="magma", norm=SQRT,
                       extent=(-shape[0] / 2, shape[0] / 2, shape[1] / 2, -shape[1] / 2))
    fig.colorbar(im, ax=axs[2], shrink=0.85)
    axs[2].set_title("Its spectrum |K̂x|: the filter.\nZero at kx = 0, strongest at low k")
    axs[2].set_xlabel("kx (cycles per padded grid)")
    axs[2].set_ylabel("ky")
    fig.tight_layout()
    fig.savefig(OUT / "p3m_kernel.png", dpi=80)


def p3m_pipeline_figure(rho, kernel_x, shape):
    padded = np.zeros(shape)
    padded[:N, :N] = rho
    rho_hat = np.fft.fft2(padded)
    product = rho_hat * np.fft.fft2(kernel_x, s=shape)
    back = np.fft.ifft2(product).real
    k_extent = (-shape[0] / 2, shape[0] / 2, shape[1] / 2, -shape[1] / 2)
    fig, axs = plt.subplots(1, 4, figsize=(21, 5.6))
    lim = np.abs(rho).max()
    axs[0].imshow(padded.T, extent=(0, shape[0], shape[1], 0), cmap="RdBu_r", vmin=-lim, vmax=lim)
    axs[0].add_patch(Rectangle((0, 0), N, N, fill=False, ec="k", lw=1.5, ls="--"))
    axs[0].set_title(f"a. Before the FFT: ρ, zero-padded\nto {shape[0]} x {shape[1]} (dashed: the scene)")
    im = axs[1].imshow(spectrum(rho_hat).T, extent=k_extent, cmap="magma", norm=SQRT)
    fig.colorbar(im, ax=axs[1], shrink=0.75)
    axs[1].set_title("b. After the FFT: |ρ̂|\n(centre, k = 0: total charge, -5 e)")
    im = axs[2].imshow(spectrum(product).T, extent=k_extent, cmap="magma", norm=SQRT)
    fig.colorbar(im, ax=axs[2], shrink=0.75)
    axs[2].set_title("c. After the filter: |ρ̂ · K̂x|\n(column kx = 0 and high k suppressed)")
    lim = np.abs(back).max()
    im = axs[3].imshow(back.T, extent=(0, shape[0], shape[1], 0), cmap="PuOr", vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=axs[3], shrink=0.75)
    axs[3].add_patch(Rectangle((N - 1, N - 1), N, N, fill=False, ec="k", lw=1.5, ls="--"))
    axs[3].set_title("d. After the inverse FFT: Ex\n(dashed: the part kept, the scene's cells)")
    for ax in (axs[0], axs[3]):
        ax.set_xlabel("x (cells)")
        ax.set_ylabel("y (cells)")
    for ax in axs[1:3]:
        ax.set_xlabel("kx (cycles per padded grid)")
        ax.set_ylabel("ky")
    fig.tight_layout()
    fig.savefig(OUT / "p3m_pipeline.png", dpi=80)
    return back[N - 1 : 2 * N - 1, N - 1 : 2 * N - 1]


def p3m_field_figure(field):
    fig, axs = plt.subplots(1, 3, figsize=(19, 6))
    lim = np.abs(field).max()
    for k, name in enumerate(("Ex", "Ey")):
        im = axs[k].imshow(field[..., k].T, extent=GRID_EXTENT, cmap="PuOr", vmin=-lim, vmax=lim)
        grid_axes(axs[k])
        axs[k].scatter(EXAMPLE[:, 0], EXAMPLE[:, 1], s=25, c="#1f5fd0", ec="w", zorder=3)
        fig.colorbar(im, ax=axs[k], shrink=0.8)
        axs[k].set_title(f"{name} on the grid nodes")
    ax = axs[2]
    grid_axes(ax)
    centres = np.arange(N) + 0.5
    strength = np.sqrt((field**2).sum(-1))
    unit = field / strength[..., None]
    cx, cy = np.meshgrid(centres, centres, indexing="ij")
    ax.quiver(cx, cy, unit[..., 0], unit[..., 1], np.log10(strength), cmap="viridis",
              angles="xy", scale_units="xy", scale=1.4, width=0.004)
    ax.scatter(EXAMPLE[:, 0], EXAMPLE[:, 1], s=60, c="#1f5fd0", zorder=3)
    ax.set_title("E as arrows (direction only; colour: log |E|).\nIt points at the (negative) electrons")
    for ax in axs:
        ax.set_xlabel("x (cells)")
        ax.set_ylabel("y (cells)")
    fig.suptitle("Step 4. The field after both inverse FFTs, cropped to the scene", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "p3m_field.png", dpi=80)


def p3m_force_figure(pm, pp, exact):
    fig, axs = plt.subplots(1, 3, figsize=(19, 6.4))
    # Arrow length grows with the square root of the force: the close pair pushes
    # 20x harder than the far field, which would otherwise hide everything else.
    for ax, force, colour, title in ((axs[0], pm, "#2a7ab0", "Step 5. PM force: E read back at each electron"),
                                     (axs[1], pp, "#d06020", f"Step 6. PP force: only pairs closer than {S.electron_cutoff:g} cells")):
        grid_axes(ax, lines=False)
        ax.grid(alpha=0.3)
        ax.scatter(EXAMPLE[:, 0], EXAMPLE[:, 1], s=50, c="#1f5fd0", zorder=3)
        for k, (px, py) in enumerate(EXAMPLE):
            ax.text(px + 0.25, py - 0.25, str(k), fontsize=9, color="#1f5fd0")
        size = np.sqrt((force**2).sum(-1))
        shown = force / np.maximum(np.sqrt(size), 1e-12)[:, None]
        ax.quiver(EXAMPLE[:, 0], EXAMPLE[:, 1], shown[:, 0], shown[:, 1], color=colour,
                  angles="xy", scale_units="xy", scale=0.25, width=0.006, zorder=4)
        ax.set_title(title + "\n(arrow length ~ square root of the force)")
        ax.set_xlabel("x (cells)")
        ax.set_ylabel("y (cells)")
    ax = axs[2]
    index = np.arange(len(EXAMPLE))
    ax.bar(index - 0.2, np.sqrt((exact**2).sum(-1)), 0.4, color="#e03030", label="brute force")
    ax.bar(index + 0.2, np.sqrt(((pm + pp) ** 2).sum(-1)), 0.4, color="k", label="P3M: PM + PP")
    ax.set_yscale("log")
    ax.set_xticks(index)
    ax.set_xlabel("electron")
    ax.set_ylabel("|force|")
    ax.legend()
    ax.set_title("Step 7. Total force on each electron,\nP3M against brute force")
    fig.tight_layout()
    fig.savefig(OUT / "p3m_forces.png", dpi=80)


def p3m_figures():
    q = np.full(len(EXAMPLE), Q)
    cic = CloudInCell(EXAMPLE, N, N)
    rho = cic.deposit(q)
    kernel = GridKernel(N, N, EPS)
    # The same real-space kernel GridKernel transforms (offsets -(N-1) .. N-1).
    kernel_x, kernel_y, _ = field_kernels(N, N, EPS)
    p3m_split_figure()
    p3m_assign_figure(cic, rho)
    p3m_kernel_figure(kernel_x, kernel.shape)
    ex = p3m_pipeline_figure(rho, kernel_x, kernel.shape)
    field = kernel.field(rho)
    assert np.allclose(ex, field[..., 0]), "the walkthrough must match GridKernel"
    padded_y = np.fft.ifft2(np.fft.fft2(rho, s=kernel.shape) * np.fft.fft2(kernel_y, s=kernel.shape))
    assert np.allclose(padded_y.real[N - 1 : 2 * N - 1, N - 1 : 2 * N - 1], field[..., 1])
    p3m_field_figure(field)
    pm = q[:, None] * cic.gather(field)
    # PP, as in CloseRange.force: pairs within the cutoff from a k-d tree.
    pp = np.zeros_like(pm)
    for i, j in cKDTree(EXAMPLE).query_pairs(S.electron_cutoff, output_type="ndarray"):
        pair = q[i] * q[j] * _close_field(EXAMPLE[i] - EXAMPLE[j], EPS_E, EPS, S.electron_cutoff)
        pp[i] += pair
        pp[j] -= pair
    exact = q[:, None] * np.stack([coulomb_field(EXAMPLE[k:k + 1], np.delete(EXAMPLE, k, 0),
                                                 np.delete(q, k), EPS_E)[0]
                                   for k in range(len(EXAMPLE))])
    p3m_force_figure(pm, pp, exact)
    error = np.sqrt(((pm + pp - exact) ** 2).sum(-1)) / np.sqrt((exact**2).sum(-1))
    print("P3M force error per electron vs brute force:", np.round(error, 3))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ion_figure()
    electron_figure()
    behaviour_figure()
    p3m_figures()
    print(f"Wrote figures to {OUT}")
