"""How a transit class perceives the physical graph that ``TransitGraphBuilder`` builds."""

from dataclasses import asdict, dataclass, fields

import numpy as np
import pandas as pd

# Unweighted edge attributes of a built transit graph. Saved graphs store these and no costs.
PHYSICAL_EDGE_FIELDS = (
    "run_time",
    "dwell_time",
    "walk_distance",
    "walk_time",
    "headway",
    "trip_count",
    "line_freq",
    "capacity",
)

# Edges that walk between stops
STOP_WALK_LINK_TYPES = ("walking", "inner_transfer", "outer_transfer")


@dataclass(frozen=True)
class TransitCosting:
    """Costs and waits of a transit class, evaluated on the physical graph when the class is assigned.

    Edge costs are in seconds. A stop's expected wait is ``headway_fraction`` times the combined headway of the
    lines its strategy boards: ``0.5`` for regular services and ``1.0`` for random arrivals. The class perceives
    ``wait_weight`` times that wait, while reported waits are never weighted.

    The defaults reproduce the costing of AequilibraE 1.x, except the headway fraction, which was ``1.0``.

    .. code-block:: python

        >>> from aequilibrae.paths import TransitCosting

        >>> costing = TransitCosting(headway_fraction=0.5, wait_weight=2.0, alighting_penalty=300.0)

    :Arguments:
        **headway_fraction** (:obj:`float`): Expected wait as a fraction of the headway. Defaults to ``0.5``.

        **wait_weight** (:obj:`float`): Perception weight of waiting time. Defaults to ``1.0``.

        **alighting_penalty** (:obj:`float`): Seconds added to every alighting. Defaults to ``480``.

        **boarding_dwell_share** (:obj:`float`): Share of a stop's scheduled dwell perceived when boarding there.
        Defaults to ``0.5``.

        **alighting_dwell_share** (:obj:`float`): Share of a stop's scheduled dwell perceived when alighting there.
        Defaults to ``0.5``.

        **access_weight** (:obj:`float`): Perception weight of walking from a zone to a stop. Defaults to ``1.0``.

        **egress_weight** (:obj:`float`): Perception weight of walking from a stop to a zone. Defaults to ``1.0``.

        **walk_weight** (:obj:`float`): Perception weight of walking between stops. Defaults to ``1.0``.
    """

    headway_fraction: float = 0.5
    wait_weight: float = 1.0
    alighting_penalty: float = 480.0
    boarding_dwell_share: float = 0.5
    alighting_dwell_share: float = 0.5
    access_weight: float = 1.0
    egress_weight: float = 1.0
    walk_weight: float = 1.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{field.name} must be finite and nonnegative")

    @staticmethod
    def applies_to(edges: pd.DataFrame) -> bool:
        """Whether the edges carry the physical attributes this costing reads."""
        return {"link_type", *PHYSICAL_EDGE_FIELDS} <= set(edges.columns)

    def edge_costs(self, edges: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return the cost and frequency of every edge of a physical transit graph."""
        link_type = edges["link_type"].to_numpy()
        run, dwell, walk = (edges[name].to_numpy(dtype=np.float64) for name in ("run_time", "dwell_time", "walk_time"))

        conditions = {
            "on-board": run,
            "dwell": dwell,
            "boarding": self.boarding_dwell_share * dwell,
            "alighting": self.alighting_dwell_share * dwell + self.alighting_penalty,
            "access_connector": self.access_weight * walk,
            "egress_connector": self.egress_weight * walk,
        }
        conditions.update(dict.fromkeys(STOP_WALK_LINK_TYPES, self.walk_weight * walk))

        unknown = set(np.unique(link_type)) - conditions.keys()
        if unknown:
            raise ValueError(f"cannot cost edges of link types {sorted(unknown)}")

        cost = np.select([link_type == name for name in conditions], list(conditions.values()))
        return cost, edges["line_freq"].to_numpy(dtype=np.float64, copy=True)

    def to_dict(self) -> dict:
        return asdict(self)
