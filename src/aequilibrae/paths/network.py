"""Source tables for routing and traffic assignment."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from aequilibrae.paths.mapping import NetworkMapping

if TYPE_CHECKING:
    from aequilibrae.paths.graph import Graph
    from aequilibrae.paths.traffic_class import TrafficClass


@dataclass(frozen=True, eq=False)
class Network:
    """
    A transportation network.

    Stores link attributes, travel modes, demand centroids, node coordinates
    and turn restrictions. The same network can support graphs with different
    modes, costs, skim fields and centroid-flow policies, while retaining a
    consistent directed-link order for costs and reported results.
    Modes are single-character mode codes, not mode names.
    """

    links: pd.DataFrame
    modes: str | Iterable[str] | None = field(default=None, kw_only=True)
    centroids: ArrayLike | None = field(default=None, kw_only=True)
    turn_restrictions: pd.DataFrame | None = field(default=None, kw_only=True)
    allow_uturns: bool = field(default=False, kw_only=True)
    lonlat: pd.DataFrame | None = field(default=None, kw_only=True)

    mapping: NetworkMapping = field(init=False, repr=False)
    _link_rows: NDArray[np.intp] = field(init=False, repr=False)
    _graphs: dict[str, Graph] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._validate_links(self.links)
        modes = self._normalise_modes(self.modes)

        if not isinstance(self.allow_uturns, bool):
            raise ValueError("allow_uturns must be a boolean")

        selected_links = self._modes_mask(self.links, modes)
        if not selected_links.any():
            raise ValueError("No links are available for the requested modes")

        links = self.links.loc[selected_links]
        node_ids = np.union1d(links["a_node"], links["b_node"])

        if self.centroids is not None:
            self._validate_centroid_nodes(self.centroids, node_ids)

        if self.turn_restrictions is not None:
            self._validate_turn_table(self.turn_restrictions, links, modes=modes)

        if self.lonlat is not None:
            self._validate_lonlat(self.lonlat, node_ids)

        available_modes = None
        if "modes" in links:
            available_modes = tuple(sorted(set("".join(links["modes"].fillna("").astype(str)))))

        # All constructor arguments are valid; take ownership of the source data.
        links = links.copy(deep=True)
        turns = self.turn_restrictions
        if turns is not None:
            turns = turns.loc[self._modes_mask(turns, modes)].copy(deep=True)

        centroids = None if self.centroids is None else np.array(self.centroids, copy=True)
        lonlat = None if self.lonlat is None else self.lonlat.copy(deep=True)

        mapping = NetworkMapping(links)

        # We map each directed entry to its positional row in links (not its
        # index label). NetworkMapping follows physical row order, emitting +1
        # before -1 for bidirectional links. Such rows therefore appear twice.
        # Field extraction uses these repeated row positions to read the same
        # source row for both directions, selecting its _ab or _ba column as
        # appropriate, without worrying about bi-directional links.
        direction = links["direction"].to_numpy()
        counts = (direction != -1).astype(np.intp) + (direction != 1)
        rows = np.repeat(np.arange(len(links), dtype=np.intp), counts)

        for values in (rows, centroids):
            if values is not None:
                values.flags.writeable = False

        object.__setattr__(self, "links", links)
        object.__setattr__(self, "modes", available_modes if modes is None else modes)
        object.__setattr__(self, "centroids", centroids)
        object.__setattr__(self, "turn_restrictions", turns)
        object.__setattr__(self, "lonlat", lonlat)
        object.__setattr__(self, "mapping", mapping)
        object.__setattr__(self, "_link_rows", rows)

    @staticmethod
    def _validate_links(links: pd.DataFrame) -> None:
        if not isinstance(links, pd.DataFrame):
            raise ValueError("links must be a DataFrame")

        if not links.columns.is_unique or any(not isinstance(column, str) for column in links.columns):
            raise ValueError("Link columns must have unique string names")

        required = {"link_id", "a_node", "b_node", "direction"}
        if missing := required - set(links.columns):
            raise ValueError(f"Network is missing required fields: {sorted(missing)}")

        if links.empty or links["link_id"].duplicated().any():
            raise ValueError("Network must contain links with unique IDs")

        for column in ("link_id", "a_node", "b_node"):
            if not pd.api.types.is_integer_dtype(links[column]) or links[column].isna().any():
                raise ValueError(f"{column} must contain integer IDs without missing values")

            if (links[column] < 0).any():
                raise ValueError(f"{column} must contain nonnegative IDs")

        if not pd.api.types.is_integer_dtype(links["direction"]) or not links["direction"].isin([-1, 0, 1]).all():
            raise ValueError("Link directions must be integers: -1, 0 or 1")

    @staticmethod
    def _normalise_modes(modes: str | Iterable[str] | None) -> tuple[str, ...] | None:
        if modes is None:
            return None

        try:
            values = (modes,) if isinstance(modes, str) else tuple(modes)
        except TypeError as exc:
            raise ValueError("modes must be a string or an iterable of strings") from exc

        if not values or any(not isinstance(mode, str) or not mode for mode in values):
            raise ValueError("modes must contain nonempty strings")

        return values

    @staticmethod
    def _modes_mask(table: pd.DataFrame, modes: tuple[str, ...] | None) -> NDArray[np.bool_]:
        if modes is None or "modes" not in table:
            return np.ones(len(table), dtype=bool)

        if not modes:
            return np.zeros(len(table), dtype=bool)

        values = table["modes"].fillna("").astype(str)
        pattern = modes[0] if len(modes) == 1 else f"[{re.escape(''.join(modes))}]"
        return values.str.contains(pattern, regex=len(modes) > 1).to_numpy(dtype=bool)

    @staticmethod
    def _validate_centroid_nodes(centroids: ArrayLike, nodes: ArrayLike) -> None:
        values = np.asarray(centroids)
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError("Centroids must be a one-dimensional array of integers")

        if not len(values) or np.any(values <= 0) or len(np.unique(values)) != len(values):
            raise ValueError("Centroids must be nonempty, unique and positive")

        if not np.isin(values, nodes).all():
            raise ValueError("Demand centroids are not present in the network")

    @staticmethod
    def _validate_lonlat(lonlat: pd.DataFrame, nodes: NDArray[np.int64]) -> None:
        if not isinstance(lonlat, pd.DataFrame):
            raise ValueError("lonlat must be a DataFrame")

        if not lonlat.columns.is_unique or not {"lon", "lat"}.issubset(lonlat.columns):
            raise ValueError("lonlat must contain unique lon and lat columns")

        if not lonlat.index.is_unique or not np.isin(nodes, lonlat.index).all():
            raise ValueError("lonlat must have a unique index containing every network node")

        try:
            coordinates = lonlat[["lon", "lat"]].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Longitude and latitude must be numeric and finite") from exc

        if not np.isfinite(coordinates).all():
            raise ValueError("Longitude and latitude must be numeric and finite")

    @staticmethod
    def _validate_turn_table(turns: pd.DataFrame, links: pd.DataFrame, *, modes: tuple[str, ...] | None = None) -> None:
        from aequilibrae.paths.graph_building import normalise_turn_penalty

        if not isinstance(turns, pd.DataFrame):
            raise ValueError("turn_restrictions must be a DataFrame")

        if not turns.columns.is_unique:
            raise ValueError("Turn restriction columns must be unique")

        required = {"from_node", "via_node", "to_node", "penalty"}
        if missing := required - set(turns.columns):
            raise ValueError(f"Turn restrictions DataFrame missing required columns: {sorted(missing)}")

        if modes is not None:
            turns = turns.loc[Network._modes_mask(turns, modes)]

        if turns.empty:
            return

        turns["penalty"].apply(normalise_turn_penalty)

        forward = links.loc[links["direction"] >= 0, ["a_node", "b_node"]]
        backward = links.loc[links["direction"] <= 0, ["b_node", "a_node"]].set_axis(["a_node", "b_node"], axis=1)
        edges = pd.MultiIndex.from_frame(pd.concat([forward, backward]))

        for first, second in (("from_node", "via_node"), ("via_node", "to_node")):
            legs = pd.MultiIndex.from_frame(turns[[first, second]])
            if not legs.isin(edges).all():
                raise ValueError(f"Turn restrictions have missing directed legs {first} -> {second}")

    @property
    def fields(self) -> list[str]:
        columns = set(self.links.columns)
        return sorted(
            {column for column in columns if not column.endswith(("_ab", "_ba"))}
            | {column[:-3] for column in columns if column.endswith("_ab") and column[:-3] + "_ba" in columns}
        )

    @property
    def link_ids(self) -> NDArray[np.uint64]:
        return self.mapping.link_ids

    @property
    def directions(self) -> NDArray[np.int8]:
        return self.mapping.directions

    def field_values(self, field: str) -> NDArray[np.float64]:
        """Return numeric values in the shared mapping's directed network order."""
        return self._field_arrays([field])[field]

    def _field_arrays(self, fields: Iterable[str]) -> dict[str, NDArray[np.float64]]:
        """Read numeric fields in the network's directed reporting order."""
        names = tuple(dict.fromkeys(fields))
        available = self.fields
        for name in names:
            if name not in available:
                raise ValueError(f"'{name}' not in network")

        arrays = {}
        for name in names:
            values = np.empty(len(self._link_rows), dtype=np.float64)

            for direction, suffix in ((1, "_ab"), (-1, "_ba")):
                column = name + suffix if name + suffix in self.links else name
                mask = self.directions == direction
                rows = self._link_rows[mask]
                try:
                    values[mask] = pd.to_numeric(self.links[column].iloc[rows], errors="raise").to_numpy(
                        dtype=np.float64
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Field {name} must contain numeric values") from exc

            values.flags.writeable = False
            arrays[name] = values

        return arrays

    def validate_mode(self, mode: str) -> None:
        if not isinstance(mode, str) or not mode or (self.modes is not None and mode not in self.modes):
            raise ValueError(f"Mode '{mode}' is not available in this network")

    def mode_mask(self, mode: str) -> NDArray[np.bool_]:
        self.validate_mode(mode)
        return self._modes_mask(self.links, (mode,))

    def validate_centroids(self, centroids: ArrayLike) -> None:
        self._validate_centroid_nodes(centroids, self.mapping.node_ids)

    def _validate_turns(self, mode: str) -> None:
        """Check mode-specific turns without preparing routing topology."""
        self.validate_mode(mode)
        if self.turn_restrictions is None:
            return

        turns = self.turn_restrictions.loc[self._modes_mask(self.turn_restrictions, (mode,))]
        self._validate_turn_table(turns, self.links.loc[self.mode_mask(mode)])

    def build_graph(
        self,
        mode: str,
        cost: str | None = None,
        *,
        centroids: ArrayLike | None = None,
        skimming: str | Iterable[str] = (),
        compact: bool = False,
        remove_dead_ends: bool = False,
        block_centroid_flows: bool = True,
        turn_restrictions: pd.DataFrame | None = None,
        allow_uturns: bool | None = None,
    ) -> Graph:
        """Prepare one routing graph from the stored source tables.

        ``mode`` is a single-character mode code, not a mode name.
        Costs use the named field. ``Graph.set_costs()`` accepts replacement
        arrays in directed network order, including infinity for link closures.
        Link contraction (``compact``) and dead-end removal are independent,
        opt-in reductions. Both retain centroids and require them to be supplied.
        Mappings expand every retained graph link back to its source links.
        """
        from aequilibrae.paths.cython.context import SkimmingContext
        from aequilibrae.paths.graph import Graph
        from aequilibrae.paths.graph_building import build_graph_mapping
        from aequilibrae.paths.routing_context import make_routing_context

        self.validate_mode(mode)
        for name, value in (
            ("compact", compact),
            ("remove_dead_ends", remove_dead_ends),
            ("block_centroid_flows", block_centroid_flows),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")

        if allow_uturns is not None and not isinstance(allow_uturns, bool):
            raise ValueError("allow_uturns must be a boolean")

        centroids = self.centroids if centroids is None else centroids
        if centroids is not None:
            self.validate_centroids(centroids)
            centroids = np.asarray(centroids)

        if (compact or remove_dead_ends) and centroids is None:
            raise ValueError("Link contraction and dead-end removal require centroids")

        if cost is not None and (not isinstance(cost, str) or not cost):
            raise ValueError("cost must be a nonempty field name")

        try:
            skim_names = (skimming,) if isinstance(skimming, str) else tuple(skimming)
        except TypeError as exc:
            raise ValueError("skimming must contain field names") from exc

        if any(not isinstance(name, str) or not name for name in skim_names):
            raise ValueError("skimming must contain nonempty field names")

        selected_links = self.mode_mask(mode)
        if not selected_links.any():
            raise ValueError(f"No links are available for mode '{mode}'")

        links = self.links.loc[selected_links]
        if centroids is not None:
            nodes = np.union1d(links["a_node"], links["b_node"])
            self._validate_centroid_nodes(centroids, nodes)

        turns = self.turn_restrictions if turn_restrictions is None else turn_restrictions
        if turns is not None:
            self._validate_turn_table(turns, links, modes=(mode,))
            turns = turns.loc[self._modes_mask(turns, (mode,))]

        field_names = skim_names if cost is None else (*skim_names, cost)
        arrays = self._field_arrays(field_names)
        fields = {name: arrays[name] for name in skim_names}
        costs = np.zeros(len(self._link_rows)) if cost is None else arrays[cost]
        if np.isnan(costs).any() or np.any(costs < 0):
            raise ValueError("Graph costs must be nonnegative and cannot contain NaN")

        # Mode-excluded links belong in reporting mappings, not routing topology.
        # All input validation precedes topology construction and owned snapshots.
        allow_uturns = self.allow_uturns if allow_uturns is None else allow_uturns
        mapping = build_graph_mapping(
            self.mapping,
            links,
            centroids=centroids,
            turn_restrictions=turns,
            compact=compact,
            remove_dead_ends=remove_dead_ends,
            allow_uturns=allow_uturns,
        )
        centroid_count = 0 if centroids is None else len(centroids)
        routing = make_routing_context(
            mapping,
            costs,
            turn_restrictions=turns,
            centroid_count=centroid_count,
            block_centroid_flows=block_centroid_flows,
            allow_uturns=allow_uturns,
        )

        skim_values = {}
        for name, values in fields.items():
            if name != cost:
                values = mapping.aggregate_costs(values)
                values.flags.writeable = False
                skim_values[name] = values

        skimming = SkimmingContext(
            mapping.link_count,
            link_fields=skim_values,
            cost_name=cost if cost in fields else None,
        )
        return Graph(
            routing=routing,
            mapping=mapping,
            skimming=skimming,
            mode=mode,
            cost_field=cost,
            centroid_count=centroid_count,
            block_centroid_flows=block_centroid_flows,
        )

    def _assignment_graph(self, configuration: TrafficClass) -> Graph:
        """Share assignment topology for classes with the same mode."""
        mode = configuration.mode
        if mode not in self._graphs:
            self._graphs[mode] = self.build_graph(
                configuration.mode,
                centroids=configuration.demand.index,
                **configuration.graph_options,
            )

        return self._graphs[mode]
