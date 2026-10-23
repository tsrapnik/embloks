"""Built-in example scenes, loaded with F1-F7.

Each scene is built from code: it resets physics settings and display layers to
their defaults, paints materials and test charges, then sets the layers that
show its point best. Scenes start from rest, so you see them settle.
"""

from dataclasses import dataclass, field
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
    # Needs dynamic fields (Maxwell: induction, waves) rather than instant ones.
    dynamic: bool = False
    # Physics settings that differ from the defaults for this scene.
    settings: dict = field(default_factory=dict)


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


def capacitor(world: World) -> None:
    """Two large parallel plates, close together, each wired to one battery terminal.

    The gap (5 cells) is several smoothing lengths wide so the uniform field between
    the plates is resolved; low-resistance wires keep the charging (RC) time short.
    """
    wire = 0.1
    plate_rows = (8, 23)
    world.paint(*line(16, 16, *plate_rows), Material.CONDUCTOR, resistance=wire)
    world.paint(*line(22, 22, *plate_rows), Material.CONDUCTOR, resistance=wire)
    world.paint(*line(19, 19, 28, 28), Material.BATTERY, emf=8, orientation=Orientation.RIGHT)
    for cells in (
        line(8, 18, 28, 28), line(8, 8, 15, 27), line(9, 15, 15, 15),     # - side, to left plate
        line(20, 30, 28, 28), line(30, 30, 15, 27), line(23, 29, 15, 15),  # + side, to right plate
    ):
        world.paint(*cells, Material.CONDUCTOR, resistance=wire)


def parallel_paths(world: World) -> None:
    """Three resistors between two bus bars: two identical, one twice as wide.

    Branches are 2 and 4 cells wide rather than 1 and 2: one-cell wires resolve the
    field worst at their junctions, which would blur the 2x comparison.
    """
    bus = 0.05
    for cols, rows in (((4, 7), (9, 30)), ((33, 36), (9, 30)), ((4, 36), (28, 30))):
        world.paint(*np.meshgrid(np.arange(*cols), np.arange(*rows), indexing="ij"),
                    Material.CONDUCTOR, resistance=bus)
    world.paint(*line(19, 19, 28, 29), Material.BATTERY, emf=6, orientation=Orientation.LEFT)
    for rows in ((10, 11), (16, 17), (21, 24)):
        world.paint(*line(7, 32, *rows), Material.CONDUCTOR, resistance=bus)
        world.paint(*line(12, 27, *rows), Material.CONDUCTOR, resistance=2)


def transformer(world: World) -> None:
    """A big AC-driven loop with a smaller loop inside it, closed through a resistor.

    In 2D a current loop is a long solenoid: its field is uniform inside and zero
    outside, so a transformer's coils must be nested. The inner loop catches the
    fraction of the outer loop's changing flux equal to its share of the area.
    """
    world.paint(*outline(4, 35, 4, 27), Material.CONDUCTOR, resistance=0.1)
    # Two sources on opposite sides, both driving the same way round: the outer loop's
    # charge then swings as a quadrupole, whose electric field is weak in the middle,
    # so the inner loop is driven by induction rather than by being charged up.
    world.paint(*line(4, 4, 15, 15), Material.BATTERY, emf=5, orientation=Orientation.UP,
                frequency=0.02)
    world.paint(*line(35, 35, 16, 16), Material.BATTERY, emf=5, orientation=Orientation.DOWN,
                frequency=0.02)
    world.paint(*outline(12, 27, 10, 21), Material.CONDUCTOR, resistance=0.1)
    world.paint(*line(27, 27, 13, 18), Material.CONDUCTOR, resistance=2)


def antenna(world: World) -> None:
    """A half-wave dipole driven at its centre, and an identical receiver one wavelength away."""
    # c = 2 cells per time unit and f = 0.1 per time unit: wavelength 20 cells.
    world.paint(*line(9, 9, 11, 21), Material.CONDUCTOR, resistance=0.1)
    world.paint(*line(9, 9, 16, 16), Material.BATTERY, emf=10, orientation=Orientation.UP,
                frequency=0.1)
    world.paint(*line(29, 29, 11, 21), Material.CONDUCTOR, resistance=0.1)
    world.paint(*line(29, 29, 16, 16), Material.CONDUCTOR, resistance=2)


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
        "Capacitor",
        "the plates charge until they are one EMF apart; between them E is nearly uniform",
        40, capacitor,
        {"B": {"visible": False}},
    ),
    Scene(
        "Parallel paths",
        "hover the resistors and read 'Wire I': the two thin ones carry equal currents, "
        "the double-wide one twice as much",
        40, parallel_paths,
        {"S": {"visible": False}},
    ),
    Scene(
        "Transformer",
        "the inner loop has no source: the outer loop's changing magnetic flux drives the "
        "current through its resistor (press M for instant fields: it stops)",
        40, transformer,
        {"E": {"visible": False}},
        # A faster light speed shrinks the loops' inductance (L ~ 1/c^2 in 2D), so a real
        # current flows round them; one AC period then takes about 3 seconds at 16x.
        sim_speed=16,
        dynamic=True,
        settings={"speed_of_light": 8.0},
    ),
    Scene(
        "Antenna",
        "the left wire radiates waves at light speed; the right one, a wavelength away, "
        "picks them up (hover its middle: Wire I)",
        40, antenna,
        {"S": {"visible": True}, "E": {"visible": False}},
        dynamic=True,
    ),
]


def load_scene(app, scene: Scene) -> None:
    """Replace the app's scene with `scene`, from rest, with default physics and layers."""
    defaults = Settings()
    for name in PHYSICS_SETTINGS:
        setattr(app.settings, name, getattr(defaults, name))
    app.settings.grid_density = scene.density
    app.settings.dynamic_fields = scene.dynamic
    for name, value in scene.settings.items():
        setattr(app.settings, name, value)
    app.reset_scene()
    scene.build(app.world)
    app.reset_layers()
    for key, attributes in scene.layers.items():
        for name, value in attributes.items():
            setattr(app.layers[key], name, value)
    app.sim_speed = scene.sim_speed
