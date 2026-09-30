"""Translate the current Graph and matrix APIs for assignment.

This boundary can be removed when Graph uses routing contexts directly. It does
not rebuild topology or repair compression and turn tables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from aequilibrae.paths.cython.aon_context import PreparedAoN
from aequilibrae.paths.cython.context import SelectLinkContext, SkimmingContext
from aequilibrae.paths.cython.outputs import SkimmingOutputs
from aequilibrae.paths.cython.parallel_numpy import aggregate_link_costs, project_link_loads, sum_axis1
from aequilibrae.paths.routing_context import make_routing_context

if TYPE_CHECKING:
    from aequilibrae.matrix import AequilibraeMatrix
    from aequilibrae.paths.cython.outputs import AoNOutputs
    from aequilibrae.paths.graph import Graph


def assignment_demand(matrix: AequilibraeMatrix, centroids: np.ndarray) -> np.ndarray:
    """Copy a computational view into fixed, packed assignment demand."""
    if matrix.view_names is None or matrix.matrix_view is None:
        raise ValueError("Set the matrix computational view before assigning demand")
    if not np.array_equal(matrix.index, centroids):
        raise ValueError("Matrix and graph must have the same centroid order")

    demand = np.asarray(matrix.matrix_view)
    zones = len(centroids)
    classes = len(matrix.view_names)
    if demand.ndim == 2 and classes == 1:
        demand = demand[:, :, None]
    if demand.shape != (zones, zones, classes) or not zones or not classes:
        raise ValueError("Demand must have shape (zones, zones, classes)")
    if not np.issubdtype(demand.dtype, np.number) or np.iscomplexobj(demand):
        raise TypeError("Demand must contain real numbers")
    if not np.all(np.isfinite(demand)) or np.any(demand < 0):
        raise ValueError("Demand must be finite and nonnegative")

    demand = np.array(demand, dtype=np.float64, order="C", copy=True)
    if not np.all(np.isfinite(demand)):
        raise ValueError("Demand must be finite when stored as float64")
    demand.flags.writeable = False
    return demand


class AssignmentMapping:
    """Map compact assignment loads to full-network reporting order.

    This is separate from GraphMapping, whose IDs describe only context links.
    """

    def __init__(self, graph: Graph):
        self.link_count = graph.num_links
        self.compact_link_count = graph.compact_num_links
        self.centroids = np.array(graph.centroids, copy=True)
        self.link_ids = graph.graph.link_id.to_numpy(copy=True)
        self.directions = graph.graph.direction.to_numpy(copy=True)
        self.graph_ids = graph.graph.__supernet_id__.to_numpy(dtype=np.int64, copy=True)
        compact_ids = graph.graph.__compressed_id__.to_numpy(dtype=np.int64, copy=True)

        self.crosswalk = np.empty(self.link_count, dtype=np.int64)
        self.crosswalk[self.graph_ids] = compact_ids
        if np.any(self.crosswalk < 0) or np.any(self.crosswalk > self.compact_link_count):
            raise ValueError("Invalid compact link mapping")

        # Graph uses compact_link_count for links removed during compression.
        # These links contribute no routing cost and receive zero assigned flow.
        for values in (self.centroids, self.link_ids, self.directions, self.graph_ids, self.crosswalk):
            values.flags.writeable = False

    def aggregate_costs(self, full_costs: np.ndarray, compact_costs: np.ndarray) -> None:
        aggregate_link_costs(full_costs, compact_costs, self.crosswalk)

    def project_loads(
        self, compact: np.ndarray, output: np.ndarray, cores: int = 1, threading_threshold: int = 10000
    ) -> None:
        project_link_loads(output, compact, self.crosswalk, cores, threading_threshold)

    def full_loads(self, compact: np.ndarray) -> np.ndarray:
        """Make a reporting snapshot in supernetwork order."""
        output = np.empty((self.link_count, compact.shape[1]), dtype=np.float64)
        self.project_loads(compact, output)
        output.flags.writeable = False
        return output


class AssignmentState:
    """One output group and its link totals in supernetwork order.

    All values remain in demand units. Only the optimizer applies PCE.
    Refresh totals after writing or blending the output group.
    """

    def __init__(self, output: AoNOutputs, mapping: AssignmentMapping, cores: int, threading_threshold: int):
        self.output = output
        self.mapping = mapping
        self.cores = cores
        self.threading_threshold = threading_threshold
        self.compact_totals = np.zeros(mapping.compact_link_count, dtype=np.float64)
        self.total_link_loads = np.zeros(mapping.link_count, dtype=np.float64)

    def update_totals(self) -> None:
        sum_axis1(self.compact_totals, self.output.loading.link_loads, self.cores, self.threading_threshold)
        self.mapping.project_loads(
            self.compact_totals[:, None], self.total_link_loads[:, None], self.cores, self.threading_threshold
        )


class AssignmentInputs:
    """Own one class's prepared routing inputs and reusable workers."""

    def __init__(
        self,
        graph: Graph,
        matrix: AequilibraeMatrix,
        time_field: str,
        selected_links: dict[str, np.ndarray],
        cores: int,
        *,
        skim_fields: list[str] | None = None,
        cost_name: str | None = None,
        heap: str = "4ary",
    ):
        self.mapping = AssignmentMapping(graph)
        self.demand = assignment_demand(matrix, self.mapping.centroids)
        self.class_names = tuple(matrix.view_names)
        links = self.mapping.compact_link_count
        self.costs = np.zeros(links, dtype=np.float64)
        self.full_costs = np.zeros(self.mapping.link_count, dtype=np.float64)

        self.routing = make_routing_context(graph, self.costs, compact=True)

        skim_names = graph.skim_fields if skim_fields is None else skim_fields
        separate_time = cost_name is not None and cost_name != time_field
        if cost_name is None and time_field in skim_names:
            cost_name = time_field

        fields = {}
        self.time_field = time_field
        self.time_skim = None
        self.turn_cost_name = None

        for name in skim_names:
            if name == time_field and not separate_time:
                # The routing distance already includes the turn costs.
                continue

            values = np.zeros(links, dtype=np.float64)
            if name == time_field:
                self.time_skim = values
            else:
                full = np.empty(self.mapping.link_count, dtype=np.float64)
                full[self.mapping.graph_ids] = graph.graph[name].to_numpy()
                self.mapping.aggregate_costs(full, values)

            fields[name] = values

        if self.time_skim is not None and graph.has_turn_restrictions:
            self.turn_cost_name = "__turn_cost__"
            while self.turn_cost_name in fields or self.turn_cost_name == cost_name:
                self.turn_cost_name = "_" + self.turn_cost_name

        self.skimming = SkimmingContext(
            links, link_fields=fields, cost_name=cost_name, turn_cost_name=self.turn_cost_name
        )
        # A selected link removed by compression can never be used by a path.
        selections = {}
        for name, indices in selected_links.items():
            indices = np.asarray(indices)
            selections[name] = indices[indices != links]
        self.selection = SelectLinkContext(links, selections)
        self.driver = PreparedAoN(
            self.routing, self.demand, cores=cores, skimming=self.skimming, selected_links=self.selection, heap=heap
        )

    def update_costs(self, congested_time: np.ndarray, fixed_cost: np.ndarray) -> None:
        np.add(congested_time, fixed_cost, out=self.full_costs)
        self.mapping.aggregate_costs(self.full_costs, self.costs)
        self.routing.update_costs(self.costs)
        if self.time_skim is not None:
            self.mapping.aggregate_costs(congested_time, self.time_skim)

    def report_skims(self, output: SkimmingOutputs) -> SkimmingOutputs:
        """Add turn delay to a separate travel-time skim, without changing raw outputs."""
        if self.turn_cost_name is None:
            return output

        matrices = output.matrices
        turn_costs = matrices.pop(self.turn_cost_name)
        matrices[self.time_field] = matrices[self.time_field] + turn_costs

        return SkimmingOutputs.from_matrices(matrices)

    def make_state(self, cores: int, threading_threshold: int) -> AssignmentState:
        return AssignmentState(self.driver.make_outputs(), self.mapping, cores, threading_threshold)
