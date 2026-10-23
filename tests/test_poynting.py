import numpy as np
import pytest

from embloks.config import Settings
from embloks.fields import MagneticField, poynting
from embloks.world import World


def test_poynting_is_perpendicular_to_e_with_magnitude_e_b_over_mu0():
    s = Settings()
    rng = np.random.default_rng(2)
    e = rng.normal(size=(6, 5, 2))
    bz = rng.normal(size=(6, 5))
    flow = poynting(e, bz, s)
    assert np.allclose((flow * e).sum(axis=-1), 0)
    inv_mu0 = s.speed_of_light**2 / (4 * np.pi * s.coulomb_k)
    assert np.allclose(np.hypot(*flow.transpose(2, 0, 1)), inv_mu0 * np.hypot(*e.transpose(2, 0, 1)) * np.abs(bz))


def test_energy_flows_into_a_current_carrying_wire_from_both_sides():
    s = Settings()
    w = World(80, 60, s)
    current = np.zeros((80, 60, 2))
    current[:, 30, 0] = 1.0  # Wire along row 30, conventional current to the right...
    e = np.zeros((80, 60, 2))
    e[..., 0] = 0.1  # ...driven by a field pointing the same way.
    flow = poynting(e, MagneticField(s).grid(w, current), s)
    assert flow[40, 25, 1] > 0  # Above the wire (smaller y): pointing down, into it.
    assert flow[40, 35, 1] < 0  # Below the wire: pointing up, into it.
    assert flow[40, 25, 0] == pytest.approx(0) and flow[40, 35, 0] == pytest.approx(0)
