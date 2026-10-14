from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths.assignment_context import validate_assignment_demand
from aequilibrae.paths.cython.dijkstra import available_heaps
from aequilibrae.paths.graph import GraphBase, TransitGraph
from aequilibrae.paths.results import TransitAssignmentResults


class TransportClassBase(ABC):  # noqa: B024
    def __init__(self, name: str, graph: GraphBase, matrix: AequilibraeMatrix) -> None:
        """
        Instantiates the class

        :Arguments:
            **name** (:obj:`str`): UNIQUE class name.

            **graph** (:obj:`GraphBase`): Class-specific graph

            **matrix** (:obj:`AequilibraeMatrix`): Class/mode-specific matrix. Supports multiple user classes
        """

        if not isinstance(matrix, AequilibraeMatrix):
            raise TypeError("Matrix must be an AequilibraeMatrix")
        if not np.array_equal(matrix.index, graph.centroids):
            raise ValueError("Matrix and graph do not have compatible sets of centroids.")
        if any(values.dtype != np.float64 for values in matrix.values()):
            raise TypeError("Matrix values need to be of type np.float64")
        self._config = {}
        self.graph = graph
        self._config["Graph"] = str(
            {
                "Mode": graph.mode,
                "Block through centroids": graph.block_centroid_flows,
                "Number of centroids": graph.num_zones,
                "Links": graph.num_links,
                "Nodes": graph.num_nodes,
            }
        )
        self.matrix = matrix
        self._id = name

        matrix_names = tuple(matrix)
        mat_config = {
            "Source": matrix.metadata.get("source", ""),
            "Number of centroids": len(matrix.index),
            "Matrix cores": matrix_names,
        }
        # Totals go into the config as plain floats, as NumPy scalars do not serialize cleanly
        mat_config["Matrix totals"] = {name: float(np.nan_to_num(matrix[name]).sum()) for name in matrix_names}
        self._config["Matrix"] = str(mat_config)

    @property
    def info(self) -> dict:
        config = deepcopy(self._config)
        return {self._id: config}


@dataclass(frozen=True, kw_only=True)
class TrafficClass:
    """Define a traffic class independently of a network or prepared graph.

    ``cost`` is the free-flow generalized cost in time units. Congestion adds
    the change in travel time. Fixed costs are multiplied by
    ``fixed_cost_multiplier`` and divided by ``vot``.

    Demand matrices must contain float64 values. ``name`` defaults to the
    mode. Classes with the same mode need distinct names.
    """

    mode: str
    cost: str
    demand: AequilibraeMatrix
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
        if not isinstance(self.demand, AequilibraeMatrix):
            raise TypeError("Demand must be an AequilibraeMatrix")
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


class TransitClass(TransportClassBase):
    def __init__(self, name: str, graph: TransitGraph, matrix: AequilibraeMatrix):
        super().__init__(name, graph, matrix)
        self._config["Graph"] = str(graph._config)
        self.results = TransitAssignmentResults()

        if len(matrix) == 1:
            self.matrix_core = next(iter(matrix))
        else:
            self.matrix_core = None

    def set_demand_matrix_core(self, core: str):
        """
        Set the matrix core to use for demand.

        :Arguments:
            **core** (:obj:`str`):"""
        if core not in self.matrix:
            raise KeyError(f"'{core}' is not present in the matrix")
        self.matrix_core = core
