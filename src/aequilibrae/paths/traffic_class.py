import warnings
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from aequilibrae.matrix import AequilibraeMatrix
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

        >>> demand = proj_matrices.get_matrix("demand_omx", subset=["matrix"])

        >>> tc = TrafficClass(
        ...     name="car", mode="c", cost='free_flow_time', demand=demand, pce=1.3,
        ...     skimming=['free_flow_time', 'distance'], block_centroid_flows=False,
        ... )

        >>> project.close()
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


@dataclass(frozen=True, kw_only=True)
class TransitClass:
    """Transit demand and its legacy transit-graph binding.

    Transit hyperpath assignment has not yet adopted the independent network
    workflow used by road TrafficClass.
    """

    name: str
    matrix: AequilibraeMatrix
    graph: object = field(default=None, repr=False, compare=False)
    matrix_core: str | None = None
    results: TransitAssignmentResults = field(default_factory=TransitAssignmentResults, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a nonempty string")
        if not isinstance(self.matrix, AequilibraeMatrix):
            raise TypeError("Matrix must be an AequilibraeMatrix")
        if any(values.dtype != np.float64 for values in self.matrix.values()):
            raise TypeError("Matrix values need to be of type np.float64")

    @property
    def _id(self):
        return self.name

    def set_demand_matrix_core(self, core):
        object.__setattr__(self, "matrix_core", core)

    @property
    def info(self):
        return {
            "name": self.name,
            "matrix_totals": {name: float(values.sum()) for name, values in self.matrix.items()},
        }
