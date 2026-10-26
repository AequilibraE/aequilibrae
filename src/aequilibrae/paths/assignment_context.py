"""Assignment-owned cost, skim, demand and output buffers for prepared graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths.cython.aon_context import PreparedAoN
from aequilibrae.paths.cython.context import SelectLinkContext, SkimmingContext, TurnBasedContext
from aequilibrae.paths.cython.outputs import SkimmingOutputs
from aequilibrae.paths.cython.parallel_numpy import sum_axis1
from aequilibrae.paths.graph import AssignmentMapping, PreparedGraph

if TYPE_CHECKING:
    from aequilibrae.paths.cython.outputs import AoNOutputs


@dataclass(frozen=True)
class SkimLayout:
    """Public skim names and the internal turn-delay reporting adjustment."""

    time_field: str
    turn_cost_name: str | None = None

    def report(self, output):
        if output is None or self.turn_cost_name is None:
            return output
        matrices = output.matrices
        turn_costs = matrices.pop(self.turn_cost_name)
        matrices[self.time_field] = matrices[self.time_field] + turn_costs
        return SkimmingOutputs.from_matrices(matrices)


def validate_assignment_demand(matrix: AequilibraeMatrix, centroids: np.ndarray) -> None:
    """Validate named demand matrices without allocating an assignment copy."""
    if not isinstance(matrix, AequilibraeMatrix):
        raise TypeError("Demand must be an AequilibraeMatrix")
    if not np.array_equal(matrix.index, centroids):
        raise ValueError("Matrix and graph must have the same centroid order")
    zones = len(centroids)
    if not zones or not len(matrix):
        raise ValueError("Demand must have shape (zones, zones, classes)")
    for values in matrix.values():
        if values.shape != (zones, zones):
            raise ValueError("Demand must have shape (zones, zones, classes)")
        if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
            raise TypeError("Demand must contain real numbers")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Demand must be finite and nonnegative")


def assignment_demand(matrix: AequilibraeMatrix, centroids: np.ndarray) -> np.ndarray:
    """Copy validated named matrices into packed, read-only assignment storage."""
    validate_assignment_demand(matrix, centroids)
    demand = np.stack(list(matrix.values()), axis=2, dtype=np.float64)
    if not np.all(np.isfinite(demand)):
        raise ValueError("Demand must be finite when stored as float64")
    demand.flags.writeable = False
    return demand


class AssignmentState:
    """One output group and its link totals in supernetwork order.

    All values remain in demand units. Only the optimizer applies PCE.
    Refresh totals after writing or blending the output group.
    """

    def __init__(
        self, output: AoNOutputs, mapping: AssignmentMapping, cores: int, threading_threshold: int, skim_layout=None
    ):
        self.skim_layout = skim_layout
        self._skim_cache = None
        self.output = output
        self.mapping = mapping
        self.cores = cores
        self.threading_threshold = threading_threshold
        self.compact_totals = np.zeros(mapping.compact_link_count, dtype=np.float64)
        self.total_link_loads = np.zeros(mapping.link_count, dtype=np.float64)

    def update_totals(self) -> None:
        self._skim_cache = None
        sum_axis1(self.compact_totals, self.output.loading.link_loads, self.cores, self.threading_threshold)
        self.mapping.project_loads(
            self.compact_totals[:, None], self.total_link_loads[:, None], self.cores, self.threading_threshold
        )

    @property
    def skims(self):
        if self.skim_layout is None:
            return self.output.skimming
        if self._skim_cache is None:
            self._skim_cache = self.skim_layout.report(self.output.skimming)
        return self._skim_cache

    def reset(self):
        if self._skim_cache is not None and self._skim_cache is not self.output.skimming:
            self._skim_cache.reset()
        self.output.reset()
        self.update_totals()


class AssignmentInputs:
    """Own one class's costs and workers while sharing immutable graph arrays."""

    def __init__(
        self,
        graph: PreparedGraph,
        matrix: AequilibraeMatrix,
        time_field: str,
        selected_links: dict[str, np.ndarray],
        cores: int,
        *,
        field_values,
        skim_fields=(),
        cost_name=None,
        heap="4ary",
        select_link_od=True,
    ):
        self.mapping = graph.mapping
        self.demand = assignment_demand(matrix, self.mapping.centroids)
        self.class_names = tuple(matrix)
        links = self.mapping.compact_link_count
        self.costs = np.zeros(links, dtype=np.float64)
        self.full_costs = np.zeros(self.mapping.link_count, dtype=np.float64)
        self.routing = graph.routing.with_costs(self.costs)
        fields = {}
        self.time_skim = None
        # Travel time has one meaning, independent of which cost skims are requested.
        cost_label = cost_name if cost_name in skim_fields and cost_name != time_field else None
        for name in skim_fields:
            if name == cost_label:
                continue
            values = np.zeros(links, dtype=np.float64)
            if name == time_field:
                self.time_skim = values
            else:
                self.mapping.aggregate_costs(field_values(name), values)
            fields[name] = values
        turn_cost_name = None
        if self.time_skim is not None and isinstance(self.routing, TurnBasedContext):
            turn_cost_name = "__turn_cost__"
            while turn_cost_name in fields or turn_cost_name == cost_label:
                turn_cost_name = "_" + turn_cost_name
        self.skim_layout = SkimLayout(time_field, turn_cost_name)
        self.skimming = SkimmingContext(links, link_fields=fields, cost_name=cost_label, turn_cost_name=turn_cost_name)
        selections = {
            name: np.asarray(indices)[np.asarray(indices) != links] for name, indices in selected_links.items()
        }
        self.selection = SelectLinkContext(links, selections)
        self.driver = PreparedAoN(
            self.routing,
            self.demand,
            cores=cores,
            skimming=self.skimming,
            selected_links=self.selection,
            heap=heap,
            select_link_od=select_link_od,
        )

    def update_costs(self, congested_time: np.ndarray, fixed_cost: np.ndarray) -> None:
        np.add(congested_time, fixed_cost, out=self.full_costs)
        self.mapping.aggregate_costs(self.full_costs, self.costs)
        self.routing.update_costs(self.costs)
        if self.time_skim is not None:
            self.mapping.aggregate_costs(congested_time, self.time_skim)

    def report_skims(self, output):
        return self.skim_layout.report(output)

    def make_state(self, cores: int, threading_threshold: int) -> AssignmentState:
        return AssignmentState(self.driver.make_outputs(), self.mapping, cores, threading_threshold, self.skim_layout)


class PreparedTrafficClass:
    """Per-class state over an execution's shared network and compact graph."""

    def __init__(self, configuration, network):
        from aequilibrae.paths.results import AssignmentResults

        self.configuration = configuration
        self.network = network
        self.graph = network.graph(configuration)
        self.results = AssignmentResults()
        self._aon_results = AssignmentResults()
        self.results.set_heap(configuration.heap)
        self._aon_results.set_heap(configuration.heap)
        self.fixed_cost = np.zeros(self.graph.mapping.link_count, dtype=np.float64)
        self.congested_time = np.array([])
        self.congested_skims = None
        self._selected_links = self.graph.selected_links(configuration.select_link_sets)

    def skim_congested(self, skim_fields=None):
        """Compute additional skims without rebuilding routing graph."""
        if self.congested_time.size == 0:
            raise RuntimeError("Run the assignment before skimming congested costs")
        fields = [skim_fields] if isinstance(skim_fields, str) else list(skim_fields or [])
        inputs = AssignmentInputs(
            self.graph,
            self.configuration.demand,
            "__congested_time__",
            {},
            self.results.cores,
            field_values=self.network.field_values,
            skim_fields=fields + ["__congested_time__", "__assignment_cost__"],
            cost_name="__assignment_cost__",
            heap=self.configuration.heap,
        )
        inputs.update_costs(self.congested_time, self.fixed_cost)
        output = inputs.driver.run(inputs.driver.make_outputs()).skimming
        self.congested_skims = inputs.report_skims(output)
        return self.congested_skims
