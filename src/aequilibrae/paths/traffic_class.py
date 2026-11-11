import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from aequilibrae.matrix import AequilibraeMatrix, AequilibraeSparseMatrix
from aequilibrae.paths.assignment_context import validate_assignment_demand
from aequilibrae.paths.cython.dijkstra import available_heaps
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
    graph_options: Mapping[str, bool] = field(default_factory=dict)
    heap: str = "4ary"

    def __post_init__(self) -> None:
        name = self.mode if self.name is None else self.name
        for label, value in (("mode", self.mode), ("cost", self.cost), ("name", name)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a nonempty string")

        self._validate_cost_parameters(self.pce, self.vot, self.fixed_cost_multiplier, self.fixed_cost_field)
        if self.heap not in available_heaps():
            raise ValueError(f"heap must be one of {available_heaps()}")

        skimming = self._normalise_skimming(self.skimming)
        if not isinstance(self.select_link_od, bool):
            raise TypeError("select_link_od must be a boolean")

        graph_options = self._normalise_graph_options(self.graph_options)
        selections = self._normalise_select_link_sets(self.select_link_sets)
        self._validate_demand(self.demand)

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "skimming", skimming)
        object.__setattr__(self, "graph_options", MappingProxyType(graph_options))
        object.__setattr__(self, "select_link_sets", MappingProxyType(selections))

    @staticmethod
    def _validate_cost_parameters(
        pce: float, vot: float, fixed_cost_multiplier: float, fixed_cost_field: str | None
    ) -> None:
        for label, value in (("PCE", pce), ("Value of time", vot)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be a positive finite number")

        if (
            isinstance(fixed_cost_multiplier, bool)
            or not isinstance(fixed_cost_multiplier, (int, float))
            or not np.isfinite(fixed_cost_multiplier)
            or fixed_cost_multiplier < 0
        ):
            raise ValueError("Fixed cost multiplier must be nonnegative and finite")

        if fixed_cost_field is not None and (not isinstance(fixed_cost_field, str) or not fixed_cost_field):
            raise ValueError("fixed_cost_field must be a nonempty string")

    @staticmethod
    def _normalise_skimming(skimming: Iterable[str]) -> tuple[str, ...]:
        if isinstance(skimming, str):
            raise ValueError("skimming must contain field names")

        skimming = tuple(skimming)
        if any(not isinstance(name, str) or not name for name in skimming):
            raise ValueError("skimming must contain field names")

        return skimming

    @staticmethod
    def _normalise_graph_options(options: Mapping[str, bool]) -> dict[str, bool]:
        if not isinstance(options, Mapping):
            raise TypeError("graph_options must be a mapping")

        graph_options = {"compact": True, "remove_dead_ends": True, "block_centroid_flows": True}
        if unknown := options.keys() - graph_options.keys():
            raise ValueError(f"Unknown graph options: {list(unknown)}")

        graph_options.update(options)
        for name, value in graph_options.items():
            if not isinstance(value, bool):
                raise TypeError(f"Graph option '{name}' must be a boolean")

        return graph_options

    @staticmethod
    def _normalise_select_link_sets(select_link_sets: Mapping[str, tuple]) -> dict[str, tuple]:
        selections = {}
        for name, members in select_link_sets.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Select-link set names must be nonempty strings")

            if " " in name:
                warnings.warn("Input string name has a space in it. Replacing with _", stacklevel=3)
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

        return selections

    @staticmethod
    def _validate_demand(demand: AequilibraeMatrix | AequilibraeSparseMatrix) -> None:
        if not isinstance(demand, (AequilibraeMatrix, AequilibraeSparseMatrix)):
            raise TypeError("Demand must be an AequilibraeMatrix or AequilibraeSparseMatrix")

        if any(values.dtype != np.float64 for values in demand.values()):
            raise TypeError("Demand matrices require float64 values")

        validate_assignment_demand(demand, demand.index)

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


@dataclass(frozen=True, kw_only=True)
class TransitClass:
    """Transit demand bound to a transit graph for hyperpath assignment."""

    name: str
    matrix: AequilibraeMatrix | AequilibraeSparseMatrix
    graph: object = field(default=None, repr=False, compare=False)
    results: TransitAssignmentResults = field(default_factory=TransitAssignmentResults, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a nonempty string")
        if not isinstance(self.matrix, (AequilibraeMatrix, AequilibraeSparseMatrix)):
            raise TypeError("Matrix must be an AequilibraeMatrix or AequilibraeSparseMatrix")
        if any(values.dtype != np.float64 for values in self.matrix.values()):
            raise TypeError("Matrix values need to be of type np.float64")

    @property
    def _id(self):
        return self.name

    @property
    def info(self):
        return {
            "name": self.name,
            "matrix_totals": {name: float(values.sum()) for name, values in self.matrix.items()},
        }
