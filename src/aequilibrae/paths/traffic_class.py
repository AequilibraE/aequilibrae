import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from aequilibrae.matrix import AequilibraeMatrix, AequilibraeSparseMatrix
from aequilibrae.paths.assignment_context import validate_assignment_demand
from aequilibrae.paths.cython.dijkstra import available_heaps
from aequilibrae.paths.graph import TransitGraph
from aequilibrae.paths.transit_costing import TransitCosting
from aequilibrae.paths.results import TransitAssignmentResults


@dataclass(frozen=True, kw_only=True)
class TrafficClass:
    """Define a traffic class independently of a network or prepared graph.

    ``cost`` is the free-flow generalized cost in time units. Congestion adds
    the change in travel time. Fixed costs are multiplied by
    ``fixed_cost_multiplier`` and divided by ``vot``.

    Demand matrices must contain float64 values. ``name`` defaults to the
    mode. Classes with the same mode need distinct names.

    .. code-block:: python

        >>> from aequilibrae.paths import TrafficClass

        >>> project = create_example(project_path)
        >>> proj_matrices = project.matrices

        >>> demand = proj_matrices.get_matrix("demand.omx", subset=["matrix"])

        >>> tc = TrafficClass(
        ...     name="car", mode="c", cost='free_flow_time', demand=demand, pce=1.3,
        ...     skimming=['free_flow_time', 'distance'], block_centroid_flows=False,
        ... )

        >>> project.close()
    """

    mode: str
    cost: str
    demand: AequilibraeMatrix | AequilibraeSparseMatrix
    name: str | None = None
    pce: float = 1.0
    vot: float = 1.0
    skimming: tuple[str, ...] = ()
    fixed_cost_field: str | None = None
    fixed_cost_multiplier: float = 1.0
    select_link_sets: Mapping[str, tuple] = field(default_factory=dict)
    select_link_od: bool = True
    block_centroid_flows: bool = True
    heap: str = "4ary"

    def __post_init__(self):
        for label, value in (
            ("mode", self.mode),
            ("cost", self.cost),
            ("name", self.mode if self.name is None else self.name),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a nonempty string")
        object.__setattr__(self, "name", self.mode if self.name is None else self.name)
        for label, value in (("PCE", self.pce), ("Value of time", self.vot)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be a positive finite number")
        if (
            isinstance(self.fixed_cost_multiplier, bool)
            or not isinstance(self.fixed_cost_multiplier, (int, float))
            or not np.isfinite(self.fixed_cost_multiplier)
            or self.fixed_cost_multiplier < 0
        ):
            raise ValueError("Fixed cost multiplier must be nonnegative and finite")
        if self.fixed_cost_field is not None and (
            not isinstance(self.fixed_cost_field, str) or not self.fixed_cost_field
        ):
            raise ValueError("fixed_cost_field must be a nonempty string")
        if self.heap not in available_heaps():
            raise ValueError(f"heap must be one of {available_heaps()}")
        if isinstance(self.skimming, str):
            raise ValueError("skimming must contain field names")
        object.__setattr__(self, "skimming", tuple(self.skimming))
        if any(not isinstance(name, str) or not name for name in self.skimming):
            raise ValueError("skimming must contain field names")
        if not isinstance(self.select_link_od, bool) or not isinstance(self.block_centroid_flows, bool):
            raise TypeError("select_link_od and block_centroid_flows must be booleans")
        selections = {}
        for name, members in self.select_link_sets.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Select-link set names must be nonempty strings")
            if " " in name:
                warnings.warn("Input string name has a space in it. Replacing with _", stacklevel=2)
                name = "_".join(name.split(" "))
            if name in selections:
                raise ValueError(f"Duplicate select-link set name after replacing spaces: {name}")
            links = []
            for member in members:
                link, direction = (member, 0) if isinstance(member, (int, np.integer)) else member
                if (
                    isinstance(link, (bool, np.bool_))
                    or not isinstance(link, (int, np.integer))
                    or isinstance(direction, (bool, np.bool_))
                    or not isinstance(direction, (int, np.integer))
                    or direction not in (-1, 0, 1)
                ):
                    raise ValueError("Selected links require an integer link ID and a direction of -1, 0 or 1")
                links.append((int(link), direction))
            selections[name] = tuple(links)
        object.__setattr__(self, "select_link_sets", MappingProxyType(selections))
        if not isinstance(self.demand, (AequilibraeMatrix, AequilibraeSparseMatrix)):
            raise TypeError("Demand must be an AequilibraeMatrix or AequilibraeSparseMatrix")
        if any(values.dtype != np.float64 for values in self.demand.values()):
            raise TypeError("Demand matrices require float64 values")
        validate_assignment_demand(self.demand, self.demand.index)

    @property
    def info(self):
        return {
            "network mode": self.mode,
            "cost": self.cost,
            "PCE": self.pce,
            "Value-of-time": self.vot,
            "Fixed cost field": self.fixed_cost_field,
            "Fixed cost multiplier": self.fixed_cost_multiplier,
            "matrix_totals": {name: float(values.sum()) for name, values in self.demand.items()},
        }


class TransitClass:
    """A transit demand class and the costing it is assigned with.

    On a graph from ``TransitGraphBuilder``, the class evaluates ``costing`` into edge costs and frequencies, and
    writes them to the graph's time and frequency fields (``trav_time`` and ``freq``). Classes sharing a graph each
    keep their own costing. Other graphs must provide those fields, and wait a full headway unless a costing gives
    the headway fraction and wait weight.

    .. code-block:: python

        >>> from aequilibrae.paths import TransitClass, TransitCosting

        >>> project = create_example(project_path, "coquimbo")
        >>> builder = project.scenario.transit.create_graph(with_walking_edges=False)
        >>> graph = builder.to_transit_graph()

        >>> demand = AequilibraeMatrix(index=graph.centroids)
        >>> demand["pt"] = np.ones((graph.num_zones, graph.num_zones))

        >>> pt = TransitClass("pt", graph, demand, costing=TransitCosting(wait_weight=2.0))

        >>> project.close()

    :Arguments:
        **name** (:obj:`str`): Unique class name.

        **graph** (:obj:`TransitGraph`): Graph to assign on.

        **matrix** (:obj:`AequilibraeMatrix` or :obj:`AequilibraeSparseMatrix`): Demand, as exactly one matrix of
        finite, nonnegative float64 values over the graph's centroids. Use ``matrix.subset(["pt"])`` to select one
        from a matrix that holds more.

        **costing** (:obj:`TransitCosting`, optional): How the class perceives the graph. Defaults to
        ``TransitCosting()`` on graphs from ``TransitGraphBuilder``.
    """

    def __init__(
        self,
        name: str,
        graph: TransitGraph,
        matrix: AequilibraeMatrix | AequilibraeSparseMatrix,
        costing: TransitCosting | None = None,
    ):
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a nonempty string")
        if not isinstance(matrix, (AequilibraeMatrix, AequilibraeSparseMatrix)):
            raise TypeError("Matrix must be an AequilibraeMatrix or AequilibraeSparseMatrix")
        if not np.array_equal(matrix.index, graph.centroids):
            raise ValueError("Matrix and graph do not have compatible sets of centroids.")
        if any(values.dtype != np.float64 for values in matrix.values()):
            raise TypeError("Matrix values need to be of type np.float64")
        if len(matrix) != 1:
            raise ValueError("Transit assignment requires exactly one demand matrix per class")
        validate_assignment_demand(matrix, graph.centroids)

        self.graph = graph
        self.matrix = matrix
        self._id = name
        mat_config = {
            "Source": matrix.metadata.get("source", ""),
            "Number of centroids": len(matrix.index),
            "Matrix cores": tuple(matrix),
            "Matrix totals": matrix.totals(),
        }
        self._config = {"Graph": str(graph._config), "Matrix": str(mat_config)}
        self.results = TransitAssignmentResults()

        self.set_costing(costing)

    @property
    def info(self) -> dict:
        return {self._id: deepcopy(self._config)}

    def set_costing(self, costing: TransitCosting | None = None) -> None:
        """Set how the class perceives the graph, and write the resulting costs to the graph's fields.

        :Arguments:
            **costing** (:obj:`TransitCosting`, optional): Costing to use. Defaults to the one set through the
            graph builder's deprecated cost attributes, or to ``TransitCosting()``.
        """
        if costing is None:
            costing = self.graph.costing
        if costing is None and TransitCosting.applies_to(self.graph.graph):
            costing = TransitCosting()
        if costing is not None and not isinstance(costing, TransitCosting):
            raise TypeError("costing must be a TransitCosting")

        self.costing = costing
        if costing is not None and TransitCosting.applies_to(self.graph.graph):
            cost, freq = costing.edge_costs(self.graph.graph)
            self.graph.graph[self.graph.time_field] = cost
            self.graph.graph[self.graph.frequency_field] = freq
        self._config["Costing"] = None if costing is None else costing.to_dict()

    def edge_costs(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the cost and frequency of every edge of the graph, as this class assigns them."""
        edges = self.graph.graph
        if self.costing is not None and TransitCosting.applies_to(edges):
            return self.costing.edge_costs(edges)
        return (
            edges[self.graph.time_field].to_numpy(dtype=np.float64),
            edges[self.graph.frequency_field].to_numpy(dtype=np.float64),
        )

    @property
    def wait_parameters(self) -> tuple[float, float]:
        """Headway fraction and wait weight of the class."""
        if self.costing is None:
            return 1.0, 1.0
        return self.costing.headway_fraction, self.costing.wait_weight

    @property
    def provenance(self) -> dict:
        """The graph and costing a result of this class comes from."""
        return {**self.graph.provenance, "costing": self._config["Costing"]}

    def apply_costs(self):
        """Give the graph's kernel this class's costs and waits, and return the kernel."""
        cost, freq = self.edge_costs()
        headway_fraction, wait_weight = self.wait_parameters
        context = self.graph.context
        context.update_costs(cost, freq, headway_fraction=headway_fraction, wait_weight=wait_weight)
        return context

    def strategy(self, origin: int, destination: int) -> pd.DataFrame:
        """Return the optimal strategy of one trip between two zones.

        Lists every edge the trip may use, with its expected ``volume`` and its ``split``, the share of the flow
        through the edge's tail that takes it. When the graph's edges have geometry (see
        ``TransitGraphBuilder.create_line_geometry``), the result is a GeoDataFrame in ``EPSG:4326``.

        .. code-block:: python

            >>> project = create_example(project_path, "coquimbo")
            >>> builder = project.scenario.transit.create_graph(with_walking_edges=False)
            >>> builder.create_line_geometry()
            >>> graph = builder.to_transit_graph()

            >>> demand = AequilibraeMatrix(index=graph.centroids)
            >>> demand["pt"] = np.ones((graph.num_zones, graph.num_zones))

            >>> strategy = TransitClass("pt", graph, demand).strategy(1, 50)

            >>> project.close()

        :Arguments:
            **origin** (:obj:`int`): Origin zone.

            **destination** (:obj:`int`): Destination zone.

        :Returns:
            **strategy** (:obj:`pd.DataFrame`): One row per edge, with its ``link_id``.
        """
        mapping = self.graph.od_node_mapping.set_index("taz_id")
        o_key, d_key = ("node_id", "node_id") if "node_id" in mapping.columns else ("o_node_id", "d_node_id")
        for zone in (origin, destination):
            if zone not in mapping.index:
                raise ValueError(f"zone {zone} is not in the graph")

        context = self.apply_costs()
        used = context.strategy(int(mapping.at[origin, o_key]), int(mapping.at[destination, d_key]))

        edges = self.graph.graph.iloc[used.index.to_numpy()]
        columns = [c for c in ("link_id", "link_type", "line_id", "stop_id", "segment_id") if c in edges.columns]
        frame = edges[columns].reset_index(drop=True)
        frame["a_node"] = self.graph.all_nodes[used["tail"].to_numpy()]
        frame["b_node"] = self.graph.all_nodes[used["head"].to_numpy()]
        frame["volume"] = used["volume"].to_numpy()
        frame["split"] = used["split"].to_numpy()

        if "geometry" not in edges.columns:
            return frame

        import geopandas as gpd

        geometry = gpd.GeoSeries.from_wkb(edges.geometry.to_numpy(), crs="EPSG:4326")
        return gpd.GeoDataFrame(frame, geometry=geometry.values, crs="EPSG:4326")
