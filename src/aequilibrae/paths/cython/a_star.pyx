"""Single-destination A* with caller-supplied heuristic inputs."""

import operator
import numpy as np

from libc.math cimport isfinite
from aequilibrae.paths.cython.dijkstra cimport RoutingContext
from aequilibrae.paths.cython.queries cimport SearchQuery
from aequilibrae.paths.cython.search_results cimport SearchResults
from aequilibrae.paths.cython.workspaces cimport AStarWorkspace, CppAStarWorkspace
from aequilibrae.utils.cython.array_allocations cimport const_array_pointer


def validate_scale(scale):
    """Check an explicit coefficient, without estimating or clamping it."""
    if scale is None:
        raise ValueError("A* requires an explicit heuristic_scale")

    scale = float(scale)
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("heuristic_scale must be finite and nonnegative")

    return scale


def _coordinates(first, second):
    first = np.array(first, dtype=np.float64, order="C", copy=True)
    second = np.array(second, dtype=np.float64, order="C", copy=True)

    if first.ndim != 1 or second.shape != first.shape or not first.size:
        raise ValueError("coordinates must be nonempty one-dimensional arrays of equal length")

    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("coordinates must be finite")

    return first, second


cdef class EuclideanContext:
    """Copy planar x/y coordinates in local node order and an explicit scale."""

    def __init__(self, x, y, scale):
        if self.node_count:
            raise RuntimeError("heuristic contexts cannot be reinitialised")

        x, y = _coordinates(x, y)
        self.scale = validate_scale(scale)
        self.x, self.y = x, y
        self.node_count = x.size

    def update_scale(self, scale: float):
        self.scale = validate_scale(scale)

    cdef CppEuclideanContext view(self) noexcept nogil:
        cdef CppEuclideanContext context
        context.node_count = self.node_count
        context.x = const_array_pointer[double](self.x)
        context.y = const_array_pointer[double](self.y)
        context.scale = self.scale
        return context


cdef class HaversineContext:
    """Copy longitude/latitude in degrees and cache radians and cosines."""

    def __init__(self, longitudes, latitudes, scale):
        if self.node_count:
            raise RuntimeError("heuristic contexts cannot be reinitialised")

        longitudes, latitudes = _coordinates(longitudes, latitudes)
        if np.any(np.abs(latitudes) > 90) or np.any(np.abs(longitudes) > 180):
            raise ValueError("haversine coordinates must be longitude/latitude in degrees")

        self.scale = validate_scale(scale)
        self.longitudes = np.deg2rad(longitudes)
        self.latitudes = np.deg2rad(latitudes)
        self.cos_latitudes = np.cos(self.latitudes)
        self.node_count = latitudes.size

    def update_scale(self, scale: float):
        self.scale = validate_scale(scale)

    cdef CppHaversineContext view(self) noexcept nogil:
        cdef CppHaversineContext context
        context.node_count = self.node_count
        context.longitudes = const_array_pointer[double](self.longitudes)
        context.latitudes = const_array_pointer[double](self.latitudes)
        context.cos_latitudes = const_array_pointer[double](self.cos_latitudes)
        context.scale = self.scale
        return context


def a_star(
    RoutingContext context, SearchQuery query not None, destination,
    HeuristicContext heuristic, SearchResults results not None, AStarWorkspace workspace not None,
):
    """
    Replace results with a single-target search; do not check scale safety.

    A consistent heuristic gives shortest paths. Larger scales can give approximate paths. Unsettled states are cleared
    before returning.
    """
    if context is None or heuristic is None:
        raise TypeError("routing and heuristic contexts must not be None")

    if isinstance(destination, (bool, np.bool_)):
        raise TypeError("destination must be a node index, not a boolean")

    destination = operator.index(destination)
    if not 0 <= destination < context.node_count:
        raise ValueError("destination is outside the context's node range")

    if query.node_count != context.node_count or heuristic.node_count != context.node_count:
        raise ValueError("query and heuristic dimensions must match context")

    if query.target_count != 1 or not query.target_mask[destination]:
        raise ValueError("A* requires the destination as its only target")

    if (results.node_count != context.node_count or
            results.state_count != context.state_count or
            results.link_count != context.link_count):
        raise ValueError("results dimensions do not match context")

    if workspace.node_count != context.node_count or workspace.state_count != context.state_count:
        raise ValueError("workspace dimensions do not match context")

    cdef size_t target = destination
    cdef CppSearchQuery cpp_query = query.view()
    cdef CppMutableSearchResults cpp_results = results.view()
    cdef CppAStarWorkspace cpp_workspace = workspace.a_star_view()

    with nogil:
        cpp_a_star(context.view(), cpp_query, target, heuristic.view(), cpp_results, cpp_workspace)

    return results


cdef double estimate_scale(
    const CppNodeBasedContext &graph, const CppHeuristicContext &heuristic,
) noexcept nogil:
    cdef size_t node, link
    cdef double distance, candidate, scale = -1.0

    for node in range(graph.node_count):
        for link in range(graph.fs[node], graph.fs[node + 1]):
            if not isfinite(graph.costs[link]):
                continue

            distance = heuristic.distance(node, graph.heads[link])
            if distance <= 0.0:
                continue

            candidate = graph.costs[link] / distance
            if isfinite(candidate) and (scale < 0.0 or candidate < scale):
                scale = candidate

    return max(scale, 0.0)


def estimate_context_scale(RoutingContext context, HeuristicContext heuristic):
    """Bound a coefficient using link costs only, without running a search."""
    if context is None or heuristic is None:
        raise TypeError("routing and heuristic contexts must not be None")

    if context.node_count != heuristic.node_count:
        raise ValueError("heuristic dimensions must match context")

    return estimate_scale(context.graph_view(), heuristic.view())
