"""Coordinate preparation and optional scale estimation for PathResults A*."""

import numpy as np
import pandas as pd

from aequilibrae.paths.cython.a_star import EuclideanContext, HaversineContext, estimate_context_scale
from aequilibrae.paths.routing_context import make_routing_context

HEURISTICS = ("euclidean", "haversine")


def make_heuristic_context(node_ids, coordinates, lonlat, heuristic, scale):
    """Copy checked coordinates into local node order for a heuristic context."""
    if heuristic not in HEURISTICS:
        raise ValueError(f"heuristic must be one of {list(HEURISTICS)}")

    frame = coordinates if heuristic == "euclidean" else lonlat

    if frame is not None and not isinstance(frame, pd.DataFrame):
        raise TypeError("coordinates must be a DataFrame indexed by external node ID")

    columns = ["x", "y"] if heuristic == "euclidean" else ["lon", "lat"]
    if frame is None or not all(column in frame.columns for column in columns):
        raise ValueError(f"{heuristic} requires coordinate columns {columns}")

    if not frame.index.is_unique:
        raise ValueError("coordinate node IDs must be unique")

    if not frame.columns.is_unique:
        raise ValueError("coordinate column names must be unique")

    if not np.all(np.isin(node_ids, frame.index)):
        raise ValueError("coordinates must include every graph node ID")

    values = frame.loc[node_ids, columns].to_numpy(dtype=np.float64)
    context_type = EuclideanContext if heuristic == "euclidean" else HaversineContext
    return context_type(values[:, 0], values[:, 1], scale)


def estimate_heuristic_scale(graph, coordinates=None, *, heuristic="euclidean") -> float:
    """
    Calculate a conservative A* coefficient.

    :Arguments:
        **graph** (:obj:`Graph`): Prepared graph with a nonnegative cost field.

        **coordinates** (:obj:`pandas.DataFrame`, optional): Planar ``x`` and ``y`` columns indexed by external node
            ID. All nodes must be present in one common coordinate system. Only used for Euclidean distance.

        **heuristic** (:obj:`str`): ``euclidean`` (default) or ``haversine``.  Haversine uses the graph's
            ``lonlat_index`` in degrees.

    :Returns:
        :obj:`float`: The smallest finite link cost divided by its positive endpoint distance, with a small round-off
            margin. Zero is returned if there are no such bounds. A zero-cost link can also force a zero scale.

    This bound works for every destination, including nonnegative turn costs.  It ignores turn and centroid
    restrictions, which can only remove paths or increase their costs. Recalculate after changing coordinates or
    decreasing link costs.  Project geographic geometry to a suitable local CRS before supplying x/y.
    """
    context = make_routing_context(graph)
    heuristic_context = make_heuristic_context(graph.all_nodes, coordinates, graph.lonlat_index, heuristic, 1.0)
    return estimate_context_scale(context, heuristic_context)
