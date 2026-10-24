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

from embloks import scenes
from embloks.close_range import compact_g, grid_force, point_correction
from embloks.config import Settings
from embloks.fields import CloudInCell
from embloks.materials import Material
from embloks.maxwell import Maxwell
from embloks.simulation import Simulation
from embloks.world import World

OUT = Path(__file__).parent / "images"
S = Settings()
EPS_E = S.electron_softening
PARAMS = (S.electron_softening, S.smoothing_passes, S.electron_cutoff)
# One-dimensional cloud-in-cell weights of an evenly spread cell: 1/8, 3/4, 1/8.
SPREAD = np.array([1 / 8, 3 / 4, 1 / 8])

plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white", "font.size": 10})


def smooth_nodes(values):
    """Node values (n, n) after the app's smoothing passes (zero beyond the edges)."""
    from embloks.maxwell import _smooth

    for _ in range(S.smoothing_passes):
        values = _smooth(values)
    return values


def compact_density(x, y, cx, cy, eps):
    """Density of a unit line charge smeared as a 2D Gaussian of width eps."""
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / eps**2) / (np.pi * eps**2)


def compact_force(r, eps=EPS_E):
    """Repulsion between two unit line charges at distance r, smeared with width eps."""
    return r * compact_g(r**2, eps)


def node_image(ax, values, half, cmap, vmax, labels=False):
    """Node values (n, n), node (n // 2, n // 2) at the origin, as squares one cell wide."""
    n = values.shape[0]
    extent = (-half - 0.5, half + 0.5, half + 0.5, -half - 0.5)
    k = n // 2 - half
    view = values[k : k + 2 * half + 1, k : k + 2 * half + 1]
    ax.imshow(view.T, extent=extent, cmap=cmap, vmin=0, vmax=vmax)
    if labels:
        for (i, j), v in np.ndenumerate(view):
            if v > 0.004:
                ax.text(i - half, j - half, f"{100 * v:.0f}%" if v >= 0.01 else "<1%", ha="center",
                        va="center", fontsize=8, color="k" if v > 0.5 * vmax else "w")
    return view


def cell_lines(ax, half):
    for v in np.arange(-half, half + 1) - 0.5:
        ax.axhline(v, color="w", lw=0.6, alpha=0.6)
        ax.axvline(v, color="w", lw=0.6, alpha=0.6)


def outline_cell(ax, color="c"):
    ax.add_patch(Rectangle((-0.5, -0.5), 1, 1, fill=False, ec=color, lw=2))


def ion_figure():
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    n, half = 15, 3
    # (1) What the cloud-in-cell deposit stores: the ion's charge per cell centre.
    shares = np.zeros((n, n))
    shares[n // 2 - 1 : n // 2 + 2, n // 2 - 1 : n // 2 + 2] = np.outer(SPREAD, SPREAD)
    node_image(axs[0], shares, half, "inferno", 0.6, labels=True)
    outline_cell(axs[0])
    axs[0].set_title("1. Deposit: share of the ion's charge\nput on each cell centre")
    # (2) Long range: those shares after the smoothing passes.
    smoothed = smooth_nodes(shares)
    node_image(axs[1], smoothed, half, "inferno", 0.6, labels=True)
    outline_cell(axs[1])
    axs[1].set_title(f"2. Long range (grid fields): after {S.smoothing_passes}\n"
                     "passes of the 1-2-1 smoothing filter")
    # (3) Close range: a uniform square.
    x, y = np.meshgrid(np.linspace(-half - 0.5, half + 0.5, 401), np.linspace(-half - 0.5, half + 0.5, 401))
    square = ((np.abs(x) < 0.5) & (np.abs(y) < 0.5)).astype(float)
    axs[2].imshow(square, extent=(-half - 0.5, half + 0.5, half + 0.5, -half - 0.5), cmap="inferno",
                  vmin=0, vmax=1.6)
    cell_lines(axs[2], half)
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
    n, half = 15, 3
    fig, axs = plt.subplots(2, 4, figsize=(17, 8.6))
    x, y = np.meshgrid(np.linspace(-half - 0.5, half + 0.5, 401), np.linspace(-half - 0.5, half + 0.5, 401))
    for k, ((px, py), label) in enumerate(positions):
        # Long range: cloud-in-cell shares on the 4 nearest centres, then smoothed.
        grid = smooth_nodes(CloudInCell(np.array([[px + n // 2 + 0.5, py + n // 2 + 0.5]]), n, n)
                            .deposit(np.ones(1)))
        ax = axs[0, k]
        node_image(ax, grid, half, "viridis", 0.2)
        ax.plot(px, py, "r+", ms=14, mew=2)
        ax.set_title(f"Electron {label}")
        # Close range: a compact ball at the electron itself.
        ax = axs[1, k]
        ax.imshow(compact_density(x, y, px, py, EPS_E), extent=(-half - 0.5, half + 0.5, half + 0.5, -half - 0.5),
                  cmap="viridis", vmin=0, vmax=compact_density(0, 0, 0, 0, EPS_E))
        cell_lines(ax, half)
        ax.plot(px, py, "r+", ms=14, mew=2)
    axs[0, 0].set_ylabel(f"Long range (grid fields)\ncloud-in-cell + {S.smoothing_passes} smoothing passes",
                         fontsize=11)
    axs[1, 0].set_ylabel(f"Close range (within {S.electron_cutoff:g} cells)\nGaussian of width {EPS_E}",
                         fontsize=11)
    fig.suptitle("One electron (-0.5 e); red +: its position. Top: its share of the charge on each "
                 "cell centre (colour scale up to 20%). Bottom: the compact ball, ~40x denser at its peak.",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "electron.png", dpi=80)


def simulate(world, ticks):
    """What App.step does: batteries, then electrons and fields."""
    simulation = Simulation(world, world.settings)
    for i in range(ticks):
        simulation.step(world.settings.dt, i * world.settings.dt)
    return simulation.maxwell


def run_scene(cutoff, seed=0):
    s = Settings(electron_cutoff=cutoff)
    w = World(40, 32, s, np.random.default_rng(seed))
    scenes.battery_and_resistor(w)
    p = w.particles
    # Start every electron on its cell centre, the worst case, to show what the physics does.
    p.pos = np.floor(p.pos) + 0.5 + w.rng.uniform(-0.02, 0.02, p.pos.shape)
    simulate(w, 1500)
    return w


def behaviour_figure():
    q2 = (S.cell_charge / S.carriers_per_cell) ** 2
    r, f_grid = grid_force(S.smoothing_passes, 4.0)
    close = point_correction(np.stack([r, 0 * r], -1), *PARAMS)[:, 1]
    fig = plt.figure(figsize=(16, 5.6))
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(r, q2 * f_grid, label="grid fields only")
    ax.plot(r, q2 * (f_grid + close), label="grid + close range (used)")
    ax.plot(r, q2 * compact_force(r), ":", label=f"compact line charges (width {EPS_E})")
    ax.set_xlabel("distance between the electrons (cells)")
    ax.set_ylabel("repulsion")
    ax.set_ylim(0, 1.5)
    ax.legend()
    ax.set_title("Force between two electrons")

    for k, (cutoff, title) in enumerate(((0.0, "Close range off (electron_cutoff = 0)"),
                                         (S.electron_cutoff, "Close range on (default)"))):
        w = run_scene(cutoff)
        ax = fig.add_subplot(1, 3, k + 2)
        for c, rr in np.argwhere(w.materials.conductive):
            ax.add_patch(Rectangle((c, rr), 1, 1, fc="#e8a060", ec="#7a5030", lw=0.5))
        for c, rr in np.argwhere(w.materials.kind == Material.BATTERY):
            ax.add_patch(Rectangle((c, rr), 1, 1, fc="#556655", ec="k", lw=0.5))
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


# The walkthrough (docs/fields.md): 10 electrons in vacuum on a 16 x 16 grid.
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


def electrons(ax, numbers=True, size=60):
    ax.scatter(EXAMPLE[:, 0], EXAMPLE[:, 1], s=size, c="#1f5fd0", ec="w", zorder=3)
    if numbers:
        for k, (px, py) in enumerate(EXAMPLE):
            ax.text(px + 0.25, py - 0.25, str(k), fontsize=9, color="#1f5fd0")


def charge_image(ax, fig, rho, title, label_values=True):
    lim = np.abs(rho).max()
    im = ax.imshow(rho.T, extent=GRID_EXTENT, cmap="RdBu_r", vmin=-lim, vmax=lim)
    grid_axes(ax)
    if label_values:
        for (c, r), v in np.ndenumerate(rho):
            if abs(v) >= 0.005:
                ax.text(c + 0.5, r + 0.5, f"{v:.2f}".replace("-0.", "-."), ha="center", va="center",
                        fontsize=7, color="w" if v < -0.6 * lim else "k")
    fig.colorbar(im, ax=ax, label="charge on the node (e)", shrink=0.8)
    ax.set_title(title)


def assign_figure(cic, rho):
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.4))
    ax = axs[0]
    grid_axes(ax)
    centres = np.arange(N) + 0.5
    cx, cy = np.meshgrid(centres, centres)
    ax.plot(cx, cy, ".", color="0.6", ms=3)
    electrons(ax)
    # Electron 0: its 4 nodes and weights.
    x0, y0 = EXAMPLE[0]
    for index, weight in zip(cic.index[:, 0], cic.weight[:, 0]):
        c, r = divmod(int(index), N)
        ax.plot([x0, c + 0.5], [y0, r + 0.5], color="#d03030", lw=1)
        right, below = c + 0.5 > x0, r + 0.5 > y0
        ax.text(c + 0.5 + (0.15 if right else -0.15), r + 0.5 + (0.45 if below else -0.2),
                f"{weight:.2f}", ha="left" if right else "right", color="#d03030", fontsize=11,
                weight="bold")
    ax.set_title("The scene: 10 electrons (blue, numbered), grid nodes at the cell centres\n"
                 "(grey). Red: electron 0 is shared over its 4 nearest nodes")
    charge_image(axs[1], fig, rho, "Step 1. The charge grid ρ\n(each electron: -0.5 e, shared over 4 nodes)")
    for ax in axs:
        ax.set_xlabel("x (cells)")
        ax.set_ylabel("y (cells)")
    fig.tight_layout()
    fig.savefig(OUT / "pic_assign.png", dpi=100)


def smooth_figure(rho, smoothed):
    fig, axs = plt.subplots(1, 3, figsize=(19, 6))
    one = np.zeros((7, 7))
    one[3, 3] = 1
    weights = smooth_nodes(one)
    axs[0].imshow(weights.T, extent=(-3.5, 3.5, 3.5, -3.5), cmap="inferno", vmin=0)
    for (i, j), v in np.ndenumerate(weights):
        if v > 0:
            axs[0].text(i - 3, j - 3, f"{v * 256:.0f}", ha="center", va="center", fontsize=9,
                        color="k" if v > 0.07 else "w")
    axs[0].set_title(f"The filter: where {S.smoothing_passes} passes of 1-2-1 send the charge\n"
                     "of one node (in 256ths): (1 4 6 4 1) / 16 per axis")
    axs[0].set_xlabel("x offset (cells)")
    axs[0].set_ylabel("y offset (cells)")
    charge_image(axs[1], fig, rho, "Before: ρ from step 1", label_values=False)
    electrons(axs[1], numbers=False, size=25)
    charge_image(axs[2], fig, smoothed, "Step 2. After smoothing: every charge\nspread over about 2 cells",
                 label_values=False)
    electrons(axs[2], numbers=False, size=25)
    for ax in axs[1:]:
        ax.set_xlabel("x (cells)")
        ax.set_ylabel("y (cells)")
    fig.tight_layout()
    fig.savefig(OUT / "pic_smooth.png", dpi=80)


def yee_figure():
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    n = 3
    for v in range(n + 1):
        ax.plot([0, n], [v, v], color="0.6", lw=1)
        ax.plot([v, v], [0, n], color="0.6", lw=1)
    c = np.arange(n) + 0.5
    e = np.arange(n + 1)
    cx, cy = np.meshgrid(c, c)
    ax.plot(cx.ravel(), cy.ravel(), "o", color="#1f5fd0", ms=11, label="nodes (cell centres): ρ, φ, J")
    ex_x, ex_y = np.meshgrid(e, c)
    ax.quiver(ex_x - 0.18, ex_y, 0.36, 0, color="#d06020", angles="xy", scale_units="xy", scale=1,
              width=0.008, label="Ex: on vertical cell edges")
    ey_x, ey_y = np.meshgrid(c, e)
    ax.quiver(ey_x, ey_y - 0.18, 0, 0.36, color="#2a9a40", angles="xy", scale_units="xy", scale=1,
              width=0.008, label="Ey: on horizontal cell edges")
    bx, by = np.meshgrid(e, e)
    ax.plot(bx.ravel(), by.ravel(), "s", color="#a02090", ms=9, mfc="w", mew=2, label="Bz: on cell corners")
    ax.set_xlim(-0.4, n + 0.4)
    ax.set_ylim(n + 0.4, -0.4)
    ax.set_aspect("equal")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("y (cells)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)
    ax.set_title("Step 3. The staggered (Yee) grid: each field where\nits differences are needed")
    fig.tight_layout()
    fig.savefig(OUT / "pic_yee.png", dpi=80)


def poisson_figure(mx):
    fig, axs = plt.subplots(1, 3, figsize=(19, 6))
    m = mx.margin
    full = (-m, N + m, N + m, -m)
    lim = np.abs(mx.potential).max()
    im = axs[0].imshow(mx.potential.T, extent=full, cmap="PuOr", vmin=-lim, vmax=lim)
    axs[0].add_patch(Rectangle((0, 0), N, N, fill=False, ec="k", lw=1.5, ls="--"))
    fig.colorbar(im, ax=axs[0], shrink=0.8)
    axs[0].set_title("Step 4. The potential φ from a sine-transform\nPoisson solve, on the whole grid "
                     "(dashed: the\nscene; φ = 0 just outside the absorbing margin)")
    e, _ = mx.node_fields()
    lim = np.abs(e).max()
    for k, name in enumerate(("Ex", "Ey")):
        ax = axs[k + 1]
        im = ax.imshow(e[..., k].T, extent=GRID_EXTENT, cmap="PuOr", vmin=-lim, vmax=lim)
        grid_axes(ax)
        electrons(ax, numbers=False, size=25)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"{name} = -dφ/d{name[1]}, averaged to the nodes for display")
    for ax in axs:
        ax.set_xlabel("x (cells)")
        ax.set_ylabel("y (cells)")
    fig.tight_layout()
    fig.savefig(OUT / "pic_poisson.png", dpi=80)


def split_figure():
    r, f_grid = grid_force(S.smoothing_passes, 4.0)
    close = point_correction(np.stack([r, 0 * r], -1), *PARAMS)[:, 1]
    with np.errstate(divide="ignore"):
        point = np.where(r > 0, 2 / r, np.nan)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(r, point, color="0.7", lw=1, label="point line charges: 2 / r")
    ax.plot(r, compact_force(r), "k:", lw=2, label=f"wanted: compact electrons (width {EPS_E})")
    ax.plot(r, f_grid, lw=2, label="PM, far part: what the grid delivers (measured)")
    ax.plot(r, close, lw=2, label="PP, near part: the difference, pair by pair")
    ax.plot(r, f_grid + close, "--", lw=1.5, label="PM + PP (used)")
    ax.axvline(S.electron_cutoff, color="0.5", lw=1)
    ax.text(S.electron_cutoff + 0.03, 5.6, "cutoff", color="0.4")
    ax.set_xlabel("distance between two charges (cells)")
    ax.set_ylabel("repulsion (unit charges)")
    ax.set_ylim(-0.3, 6)
    ax.legend()
    ax.set_title("Splitting the force into a far and a near part")
    fig.tight_layout()
    fig.savefig(OUT / "pic_split.png", dpi=80)


def force_figure(pm, pp, exact):
    fig, axs = plt.subplots(1, 3, figsize=(19, 6.4))
    # Arrow length grows with the square root of the force: the close pair pushes
    # far harder than the rest, which would otherwise hide everything else.
    for ax, force, colour, title in ((axs[0], pm, "#2a7ab0", "Step 5. PM force: E read back at each electron"),
                                     (axs[1], pp, "#d06020", f"Step 6. PP force: only pairs closer than {S.electron_cutoff:g} cells")):
        grid_axes(ax, lines=False)
        ax.grid(alpha=0.3)
        electrons(ax, size=50)
        size = np.sqrt((force**2).sum(-1))
        shown = force / np.maximum(np.sqrt(size), 1e-12)[:, None]
        ax.quiver(EXAMPLE[:, 0], EXAMPLE[:, 1], shown[:, 0], shown[:, 1], color=colour,
                  angles="xy", scale_units="xy", scale=0.4, width=0.006, zorder=4)
        ax.set_title(title + "\n(arrow length ~ square root of the force)")
        ax.set_xlabel("x (cells)")
        ax.set_ylabel("y (cells)")
    ax = axs[2]
    index = np.arange(len(EXAMPLE))
    ax.bar(index - 0.2, np.sqrt((exact**2).sum(-1)), 0.4, color="#e03030", label="brute force")
    ax.bar(index + 0.2, np.sqrt(((pm + pp) ** 2).sum(-1)), 0.4, color="k", label="PM + PP")
    ax.set_yscale("log")
    ax.set_xticks(index)
    ax.set_xlabel("electron")
    ax.set_ylabel("|force|")
    ax.legend()
    ax.set_title("Step 7. Total force on each electron,\nPM + PP against brute force")
    fig.tight_layout()
    fig.savefig(OUT / "pic_forces.png", dpi=80)


def waves_figure():
    """The antenna scene (F7) after a while: waves leaving, swallowed by the margin."""
    loaded = next(sc for sc in scenes.SCENES if sc.build is scenes.antenna)
    s = Settings(**loaded.settings)
    w = World(40, 32, s, np.random.default_rng(0))
    scenes.antenna(w)
    mx = simulate(w, int(round(25 / s.dt)))
    m = mx.margin
    bz = mx.bz * s.speed_of_light**2
    lim = np.percentile(np.abs(bz), 99.5)
    fig, ax = plt.subplots(figsize=(9, 7.2))
    im = ax.imshow(bz.T, extent=(-m - 0.5, 40 + m + 0.5, 32 + m + 0.5, -m - 0.5), cmap="RdBu", vmin=-lim, vmax=lim)
    ax.add_patch(Rectangle((0, 0), 40, 32, fill=False, ec="k", lw=1.5, ls="--"))
    for c, r in np.argwhere(w.materials.conductive | (w.materials.kind == Material.BATTERY)):
        ax.add_patch(Rectangle((c, r), 1, 1, fc="#e8a060", ec="none"))
    fig.colorbar(im, ax=ax, shrink=0.8, label="c² Bz")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("y (cells)")
    ax.set_title("Antenna (F7) after 25 time units: waves of B spread at c = 2 cells per\n"
                 "time unit and fade in the absorbing margin (outside the dashed scene)")
    fig.tight_layout()
    fig.savefig(OUT / "pic_waves.png", dpi=80)


def walkthrough_figures():
    q = np.full(len(EXAMPLE), Q)
    cic = CloudInCell(EXAMPLE, N, N)
    rho = cic.deposit(q)
    mx = Maxwell(N, N, S)
    full = mx.smoothed(rho)
    assign_figure(cic, rho)
    smooth_figure(rho, mx.visible(full))
    yee_figure()
    mx.enforce_gauss(full)
    poisson_figure(mx)
    split_figure()
    pm = q[:, None] * mx.gather(EXAMPLE)[:, :2]
    # PP, as in CloseRange.force: pairs within the cutoff.
    pp = np.zeros_like(pm)
    for i, j in cKDTree(EXAMPLE).query_pairs(S.electron_cutoff, output_type="ndarray"):
        pair = q[i] * q[j] * point_correction(EXAMPLE[i] - EXAMPLE[j], *PARAMS)[1:]
        pp[i] += pair
        pp[j] -= pair
    d = EXAMPLE[:, None] - EXAMPLE[None]
    exact = (q[:, None, None] * q[None, :, None] * compact_g((d**2).sum(-1), EPS_E)[..., None] * d).sum(1)
    force_figure(pm, pp, exact)
    error = np.sqrt(((pm + pp - exact) ** 2).sum(-1)) / np.sqrt((exact**2).sum(-1))
    print("PM + PP force error per electron vs brute force:", np.round(error, 3))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ion_figure()
    electron_figure()
    behaviour_figure()
    walkthrough_figures()
    waves_figure()
    print(f"Wrote figures to {OUT}")
