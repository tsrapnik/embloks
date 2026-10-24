import numpy as np
import pygame
import pytest

from embloks import physics
from embloks.app import App
from embloks.config import Settings
from embloks.maxwell import Maxwell, _smooth


def point_charge(mx: Maxwell, col: int, row: int, q: float = 1.0) -> np.ndarray:
    rho = np.zeros((mx.nx, mx.ny))
    rho[col + mx.margin, row + mx.margin] = q
    return rho


def test_gauss_law_holds_exactly_and_static_field_is_2d_coulomb():
    s = Settings()
    mx = Maxwell(40, 32, s)
    rho = point_charge(mx, 20, 16)
    mx.enforce_gauss(rho)
    assert np.abs(mx.divergence() - 4 * np.pi * s.coulomb_k * rho).max() < 1e-12
    e, _ = mx.node_fields()
    for r in (4, 8):  # A line charge: |E| = 2 k q / r.
        assert np.hypot(*e[20 + r, 16]) == pytest.approx(2 / r, rel=0.05)


def gaussian_pulse(mx: Maxwell, col: float, row: float, width2: float) -> None:
    x, y = np.meshgrid(np.arange(mx.nx + 1) - 0.5, np.arange(mx.ny + 1) - 0.5, indexing="ij")
    mx.bz = np.exp(-((x - col - mx.margin) ** 2 + (y - row - mx.margin) ** 2) / width2)


def test_waves_travel_at_the_speed_of_light():
    s = Settings(absorbing_margin=40, speed_of_light=2.0)  # Slow enough to stay on the grid.
    mx = Maxwell(80, 80, s)
    gaussian_pulse(mx, 40, 40, 0.5)
    rho, current = np.zeros((mx.nx, mx.ny)), np.zeros((mx.nx, mx.ny, 2))
    fronts, t = [], 0.0
    for until in (5.0, 10.0, 15.0):
        while t < until - 1e-9:
            mx.step(rho, current, s.dt)
            t += s.dt
        profile = np.abs(mx.bz[mx.margin + 40 :, mx.margin + 40])
        fronts.append(np.max(np.nonzero(profile > 1e-3)[0]))
    speeds = np.diff(fronts) / 5.0
    assert speeds == pytest.approx([s.speed_of_light] * 2, abs=0.25)


def test_absorbing_border_swallows_outgoing_waves():
    s = Settings()
    mx = Maxwell(40, 32, s)
    gaussian_pulse(mx, 20, 16, 2.0)
    rho, current = np.zeros((mx.nx, mx.ny)), np.zeros((mx.nx, mx.ny, 2))

    def energy():
        return (mx.ex**2).sum() + (mx.ey**2).sum() + s.speed_of_light**2 * (mx.bz**2).sum()

    for _ in range(int(5 / s.dt)):
        mx.step(rho, current, s.dt)
    before = energy()
    for _ in range(int(40 / s.dt)):
        mx.step(rho, current, s.dt)
    assert energy() < 1e-3 * before


def test_fast_light_is_substepped_and_stays_stable():
    s = Settings(speed_of_light=40.0)  # c dt sqrt(2) ~ 1.9: unstable in a single step.
    mx = Maxwell(30, 30, s)
    gaussian_pulse(mx, 15, 15, 2.0)
    for _ in range(300):
        mx.step(np.zeros((mx.nx, mx.ny)), np.zeros((mx.nx, mx.ny, 2)), s.dt)
    assert np.isfinite(mx.bz).all() and np.abs(mx.bz).max() < 1


def test_staggered_gather_reads_uniform_fields_exactly():
    mx = Maxwell(10, 8, Settings())
    mx.ex[:] = 0.3
    mx.ey[:] = -0.2
    mx.bz[:] = 0.7
    fields = mx.gather(np.array([[2.3, 4.6], [7.9, 1.1]]))
    assert np.allclose(fields, [[0.3, -0.2, 0.7]] * 2)


def test_smoothing_conserves_charge():
    rng = np.random.default_rng(0)
    rho = rng.normal(size=(12, 9))
    current = rng.normal(size=(12, 9, 2))
    # Charge on the outermost ring would leak out; in the app that ring lies deep inside
    # the absorbing margin, where there is never any charge.
    for array in (rho, current):
        array[[0, -1]] = 0
        array[:, [0, -1]] = 0
    assert _smooth(rho).sum() == pytest.approx(rho.sum())
    assert _smooth(current).sum(axis=(0, 1)) == pytest.approx(current.sum(axis=(0, 1)))


@pytest.fixture
def app():
    app = App(Settings())
    yield app
    pygame.quit()


def test_gauss_law_holds_after_stepping_a_scene(app):
    app.load_scene(1)  # Screening: conductor plus test charges.
    for _ in range(30):
        app.step(app.settings.dt)
    mx = app.maxwell
    rho, _ = mx.sources(app.world)
    assert np.abs(mx.divergence() - 4 * np.pi * rho).max() < 1e-9


def test_smoothing_the_visible_grid_matches_smoothing_the_whole_grid():
    s = Settings()
    mx = Maxwell(9, 7, s)
    values = np.random.default_rng(1).normal(size=(9, 7, 3))
    whole = np.zeros((mx.nx, mx.ny, 3))
    mx.visible(whole)[...] = values
    for _ in range(s.smoothing_passes):
        whole = _smooth(whole)
    np.testing.assert_allclose(mx.smoothed(values), whole, atol=1e-15)
