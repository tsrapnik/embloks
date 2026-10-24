"""Saving and loading the complete simulation state.

A save is a compressed .npz archive: the arrays (materials, electrons, test
charges, smoothed current) plus a JSON "meta" entry with everything else
(format version, physics settings, battery pump state, view layers, speed).
"""

import json
from pathlib import Path

import numpy as np

from embloks.world import World

FORMAT_VERSION = 1

# Settings that shape the saved state. They are restored on load, so a scene
# behaves the same even if config.py has been edited since it was saved.
PHYSICS_SETTINGS = (
    "dt",
    "cell_charge",
    "carriers_per_cell",
    "carrier_charge_to_mass",
    "drag_per_resistance",
    "battery_resistance",
    "current_smoothing_time",
    "wall_restitution",
    "max_step",
    "max_substeps",
    "coulomb_k",
    "speed_of_light",
    "electron_softening",
    "electron_cutoff",
    "battery_flow_time",
    "smoothing_passes",
    "absorbing_margin",
    "absorbing_strength",
)

LAYER_FIELDS = ("visible", "gain", "opacity", "log_scale", "auto")


def save(app, path: Path) -> None:
    """Write the app's world, view and physics settings to `path` (.npz)."""
    w, m, p = app.world, app.world.materials, app.world.particles
    meta = {
        "format": FORMAT_VERSION,
        "cols": m.cols,
        "rows": m.rows,
        "sim_time": app.sim_time,
        "sim_speed": app.sim_speed,
        "settings": {name: getattr(app.settings, name) for name in PHYSICS_SETTINGS},
        # Stacks are found in a fixed order from the materials, so a list lines up on load.
        "pump_charge": [w.pump_charge.get(stack, 0.0) for stack in w.stacks],
        "pump_current": [w.pump_current.get(stack) for stack in w.stacks],
        "pump_flow": [w.pump_flow.get(stack) for stack in w.stacks],
        "layers": {
            key: {field: getattr(layer, field) for field in LAYER_FIELDS}
            for key, layer in app.layers.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.savez_compressed(
            f,
            meta=np.array(json.dumps(meta)),
            kind=m.kind,
            resistance=m.resistance,
            emf=m.emf,
            orientation=m.orientation,
            frequency=m.frequency,
            pos=p.pos,
            vel=p.vel,
            charge=p.charge,
            mass=p.mass,
            fixed_pos=w.fixed_pos,
            fixed_charge=w.fixed_charge,
            current=app.current,
        )


def load(app, path: Path) -> None:
    """Replace the app's state with the one saved at `path`."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    meta = json.loads(str(arrays.pop("meta")))
    if meta.get("format") != FORMAT_VERSION:
        raise ValueError(f"unsupported save format {meta.get('format')!r}")

    # Settings first: the world reads them (e.g. carrier charge) while being rebuilt.
    # Mutate in place, since world, fields and panel all share this object.
    # Settings missing from older saves keep their defaults.
    defaults = type(app.settings)()
    for name in PHYSICS_SETTINGS:
        value = meta["settings"].get(name, getattr(defaults, name))
        setattr(app.settings, name, type(getattr(app.settings, name))(value))

    world = World(meta["cols"], meta["rows"], app.settings)
    m = world.materials
    m.kind[:] = arrays["kind"]
    m.resistance[:] = arrays["resistance"]
    m.emf[:] = arrays["emf"]
    m.orientation[:] = arrays["orientation"]
    if "frequency" in arrays:  # Saves from before AC sources have none.
        m.frequency[:] = arrays["frequency"]
    m.version += 1
    world._update_structure()
    p = world.particles
    p.pos, p.vel, p.charge, p.mass = arrays["pos"], arrays["vel"], arrays["charge"], arrays["mass"]
    world.fixed_pos, world.fixed_charge = arrays["fixed_pos"], arrays["fixed_charge"]
    world.fixed_version += 1
    flows = meta.get("pump_flow", [None] * len(world.stacks))
    for stack, owed, current, flow in zip(
        world.stacks, meta["pump_charge"], meta["pump_current"], flows
    ):
        world.pump_charge[stack] = owed
        if current is not None:
            world.pump_current[stack] = current
        if flow is not None:
            world.pump_flow[stack] = flow

    app.install_world(world)
    app.current = arrays["current"]
    app.sim_time = meta["sim_time"]
    app.sim_speed = meta["sim_speed"]
    for key, fields in meta["layers"].items():
        if key in app.layers:
            for field, value in fields.items():
                setattr(app.layers[key], field, value)


def slot_path(save_dir: str, slot: int) -> Path:
    return Path(save_dir) / f"slot{slot}.npz"

