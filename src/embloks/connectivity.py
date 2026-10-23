from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from embloks.materials import Material, Materials, Orientation


@dataclass(frozen=True)
class Connectivity:
    """Which neighbouring cells conduct into each other, and the resulting islands.

    Cells connect only through shared edges (never corners). A battery connects
    only through its + and - faces; its two side faces are walls.
    """

    # open_x[c, r]: cells (c, r) and (c + 1, r) connect. Shape (cols - 1, rows).
    open_x: np.ndarray
    # open_y[c, r]: cells (c, r) and (c, r + 1) connect. Shape (cols, rows - 1).
    open_y: np.ndarray
    # Island id per cell, 0..count-1, or -1 for vacuum. Shape (cols, rows).
    labels: np.ndarray
    count: int


def compute_connectivity(materials: Materials) -> Connectivity:
    kind, orientation = materials.kind, materials.orientation
    conductor = kind == Material.CONDUCTOR
    battery = kind == Material.BATTERY
    horizontal = (orientation == Orientation.RIGHT) | (orientation == Orientation.LEFT)
    # Whether a cell lets charge through its left/right faces, and its top/bottom faces.
    passes_x = conductor | (battery & horizontal)
    passes_y = conductor | (battery & ~horizontal)
    open_x = passes_x[:-1] & passes_x[1:]
    open_y = passes_y[:, :-1] & passes_y[:, 1:]

    cols, rows = kind.shape
    index = np.arange(cols * rows).reshape(cols, rows)
    a = np.concatenate([index[:-1][open_x], index[:, :-1][open_y]])
    b = np.concatenate([index[1:][open_x], index[:, 1:][open_y]])
    graph = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(cols * rows,) * 2)
    _, component = connected_components(graph, directed=False)
    component = component.reshape(cols, rows)

    # Renumber so islands are 0..count-1 and vacuum is -1.
    conductive = materials.conductive
    labels = np.full((cols, rows), -1, dtype=np.int64)
    unique, labels[conductive] = np.unique(component[conductive], return_inverse=True)
    return Connectivity(open_x, open_y, labels, len(unique))
