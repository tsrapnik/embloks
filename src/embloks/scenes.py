"""Built-in example scenes, loaded with F1-F4.

Each scene is built from code: it resets physics settings and display layers to
their defaults, paints materials and test charges, then sets the layers that
show its point best. Scenes start from rest, so you see them settle.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from embloks.config import Settings
from embloks.materials import Material, Orientation
from embloks.storage import PHYSICS_SETTINGS
from embloks.world import World


@dataclass(frozen=True)
class Scene:
    title: str
    # One line, shown in the info panel on load.
    hint: str
    # Grid columns; rows follow from the window.
    density: int
    build: Callable[[World], None]
    # Layer id -> attributes to set after resetting all layers to defaults.
    layers: dict[str, dict]
    sim_speed: float = 8


def outline(c0: int, c1: int, r0: int, r1: int) -> tuple[np.ndarray, np.ndarray]:
    """Columns and rows of a rectangle's border cells, corners included."""
    cells = {(c, r) for c in range(c0, c1 + 1) for r in (r0, r1)}
    cells |= {(c, r) for c in (c0, c1) for r in range(r0, r1 + 1)}
    cols, rows = np.array(sorted(cells)).T
    return cols, rows


def line(c0: int, c1: int, r0: int, r1: int) -> tuple[np.ndarray, np.ndarray]:
    """Cells of a horizontal (r0 == r1) or vertical (c0 == c1) run, ends included."""
    cols, rows = np.meshgrid(np.arange(c0, c1 + 1), np.arange(r0, r1 + 1), indexing="ij")
    return cols.ravel(), rows.ravel()


def battery_and_resistor(world: World) -> None:
    """The textbook circuit: battery on the left (+ up), resistor on the right."""
    world.paint(*outline(10, 29, 7, 24), Material.CONDUCTOR, resistance=0.2)
    world.paint(*line(29, 29, 12, 19), Material.CONDUCTOR, resistance=5)
    world.paint(*line(10, 10, 15, 15), Material.BATTERY, emf=6, orientation=Orientation.UP)


def screening(world: World) -> None:
    """A neutral conductor between a positive and a negative test charge."""
    world.paint(*np.meshgrid(np.arange(15, 25), np.arange(11, 21), indexing="ij"),
                Material.CONDUCTOR, resistance=1)
    world.place_charges(np.array([8]), np.array([16]), 5)
    world.place_charges(np.array([31]), np.array([16]), -5)


def series_batteries(world: World) -> None:
    """Two identical loops: one cell of EMF 2 on the left, three in series on the right."""
    for c0, cells in ((3, 1), (22, 3)):
        world.paint(*outline(c0, c0 + 14, 8, 23), Material.CONDUCTOR, resistance=0.5)
        middle = c0 + 7
        world.paint(*line(middle - cells // 2, middle + cells // 2, 23, 23),
                    Material.BATTERY, emf=2, orientation=Orientation.RIGHT)


def open_circuit(world: World) -> None:
    """A battery with two wires whose ends don't meet: they charge up like a capacitor."""
    world.paint(*line(20, 20, 18, 18), Material.BATTERY, emf=8, orientation=Orientation.RIGHT)
    world.paint(*line(8, 19, 18, 18), Material.CONDUCTOR, resistance=0.5)
    world.paint(*line(21, 32, 18, 18), Material.CONDUCTOR, resistance=0.5)
    world.paint(*line(8, 8, 8, 17), Material.CONDUCTOR, resistance=0.5)
    world.paint(*line(32, 32, 8, 17), Material.CONDUCTOR, resistance=0.5)
    world.paint(*line(8, 17, 8, 8), Material.CONDUCTOR, resistance=0.5)
    world.paint(*line(23, 32, 8, 8), Material.CONDUCTOR, resistance=0.5)


SCENES = [
    Scene(
        "Battery and resistor",
        "energy (green S arrows) flows from the battery through space into the resistor",
        40, battery_and_resistor,
        {"E": {"visible": False}, "S": {"visible": True}},
    ),
    Scene(
        "Screening",
        "electrons shift until E inside the conductor is (nearly) zero",
        40, screening,
        {"B": {"visible": False}},
    ),
    Scene(
        "Batteries in series",
        "3 cells in series drive 3x the current: hover the wires to compare",
        40, series_batteries,
        {"S": {"visible": False}},
    ),
    Scene(
        "Open circuit",
        "charge piles up at the wire ends until they are one EMF apart",
        40, open_circuit,
        {"B": {"visible": False}},
    ),
]


def load_scene(app, scene: Scene) -> None:
    """Replace the app's scene with `scene`, from rest, with default physics and layers."""
    defaults = Settings()
    for name in PHYSICS_SETTINGS:
        setattr(app.settings, name, getattr(defaults, name))
    app.settings.grid_density = scene.density
    app.reset_scene()
    scene.build(app.world)
    app.reset_layers()
    for key, attributes in scene.layers.items():
        for name, value in attributes.items():
            setattr(app.layers[key], name, value)
    app.sim_speed = scene.sim_speed
