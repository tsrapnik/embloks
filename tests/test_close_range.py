import numpy as np
import pytest
from scipy.spatial import cKDTree

from embloks import close_range
from embloks.close_range import CloseRange, compact_g, point_correction, square_table
from embloks.config import Settings
from embloks.materials import Material
from embloks.maxwell import Maxwell
from embloks.world import World


def reference_force(world, settings):
    """CloseRange.force the slow way: a k-d tree for the pairs, a sum over ions."""
    s, p = settings, world.particles
    params = (s.electron_softening, s.smoothing_passes, s.electron_cutoff)
    force = np.zeros_like(p.pos)
    i, j = cKDTree(p.pos).query_pairs(s.electron_cutoff, output_type="ndarray").T
    pair = s.coulomb_k * (p.charge[i] * p.charge[j])[:, None] * point_correction(
        p.pos[i] - p.pos[j], *params)[:, 1:]
    np.add.at(force, i, pair)
    np.add.at(force, j, -pair)
    square, half = square_table(*params)
    for c, r in np.argwhere(world.materials.conductive):
        d = p.pos - (np.array([c, r]) + 0.5)
        near = np.all(np.abs(d) < half, axis=1)
        index = (d[near] + half) * close_range._SQUARE_SAMPLES
        for axis in (0, 1):
            force[near, axis] += (s.coulomb_k * s.cell_charge * p.charge[near]
                                  * close_range._bilinear(square[..., 1 + axis], index))
    return force


def test_compiled_force_matches_the_reference():
    s = Settings()
    w = World(30, 20, s, np.random.default_rng(1))
    w.paint(*np.nonzero(np.random.default_rng(2).uniform(size=(30, 20)) < 0.5),
            Material.CONDUCTOR, resistance=1)
    p = w.particles
    p.pos = np.clip(p.pos + np.random.default_rng(3).uniform(-0.3, 0.3, p.pos.shape),
                    0.001, [29.999, 19.999])
    force = CloseRange(s).force(w)
    reference = reference_force(w, s)
    assert np.abs(force - reference).max() < 1e-12 * np.abs(reference).max()


def test_grid_plus_close_range_gives_the_compact_force():
    """Two electrons: the Maxwell grid's force plus the correction follows a compact
    line charge within the cutoff, on average over where they sit in their cells."""
    s = Settings()
    w = World(24, 24, s)
    w.particles.add(np.zeros((2, 2)), -1.0, 1.0)
    rng = np.random.default_rng(0)
    close = CloseRange(s)
    for r in (0.25, 0.5, 1.0, 2.0, 2.8):
        totals = []
        for _ in range(40):
            a = rng.uniform(10, 13, 2)
            u = np.array([np.cos(angle := rng.uniform(0, 2 * np.pi)), np.sin(angle)])
            w.particles.pos = np.array([a, a + r * u])
            grid = -1.0 * Maxwell.for_world(w, s).gather(w.particles.pos)[1, :2]
            totals.append((grid + close.force(w)[1]) @ u)
        compact = r * compact_g(r * r, s.electron_softening)  # q^2 = 1.
        assert np.mean(totals) == pytest.approx(compact, rel=0.05)


def test_close_range_force_pushes_stacked_electrons_apart():
    s = Settings()
    w = World(10, 10, s)
    w.paint(np.array([4, 5]), np.array([4, 4]), Material.CONDUCTOR, resistance=1)
    w.particles.pos[:] = [5.0, 4.5]
    w.particles.pos[0, 0] -= 0.05
    w.particles.pos[1, 0] += 0.05
    force = CloseRange(s).force(w)
    # The pair repels (and momentum is conserved between electrons)...
    assert force[0, 0] < -0.5 and force[1, 0] > 0.5
    # ...while the ions of the two cells draw the rest of them back symmetrically.
    assert force.sum(axis=0) == pytest.approx([0.0, 0.0], abs=1e-9)


def test_evenly_filled_conductor_feels_no_net_close_range_force():
    """Deep inside a conductor (over a cutoff from its surface), electrons on an even
    lattice and the evenly spread ions balance: no net close-range force."""
    s = Settings(carriers_per_cell=16)
    w = World(22, 18, s)
    w.paint(np.repeat(np.arange(2, 20), 14), np.tile(np.arange(2, 16), 18),
            Material.CONDUCTOR, resistance=1)
    sub = (np.arange(4) + 0.5) / 4
    pattern = np.stack(np.meshgrid(sub, sub, indexing="ij"), axis=-1).reshape(-1, 2)
    cells = np.argwhere(w.materials.conductive)
    w.particles.pos = (cells[:, None, :] + pattern[None]).reshape(-1, 2)
    force = CloseRange(s).force(w)
    inner = np.all((w.particles.pos > [6, 6]) & (w.particles.pos < [16, 12]), axis=1)
    pair = (s.cell_charge / 16) ** 2 * 0.25 * compact_g(0.25**2, s.electron_softening)
    assert np.abs(force[inner]).max() < 1e-3 * pair
