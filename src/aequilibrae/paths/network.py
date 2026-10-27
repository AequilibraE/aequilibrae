"""Source tables for routing and traffic assignment."""

from __future__ import annotations

import numpy as np
import pandas as pd


class Network:
    """Store link attributes, modes, centroids and turn restrictions."""

    def __init__(self, links, *, modes=None, centroids=None, turn_restrictions=None, allow_uturns=False, lonlat=None):
        links = links.copy(deep=True)
        links.columns = links.columns.str.lower()
        if modes is not None:
            modes = (modes,) if isinstance(modes, str) else tuple(modes)
        required = {"link_id", "a_node", "b_node", "direction"}
        if missing := required - set(links.columns):
            raise ValueError(f"Network is missing required fields: {sorted(missing)}")
        if links.empty or links.link_id.duplicated().any():
            raise ValueError("Network must contain links with unique IDs")
        if not links.direction.isin([-1, 0, 1]).all():
            raise ValueError("Link directions must be -1, 0 or 1")
        self._node_ids = np.union1d(links.a_node, links.b_node)
        self._all_modes = "modes" not in links
        available = set() if self._all_modes else set("".join(links.modes.fillna("").astype(str)))
        self.modes = None if modes is None and self._all_modes else tuple(sorted(available) if modes is None else modes)
        if modes is not None and not self._all_modes:
            selected = links.modes.fillna("").astype(str).map(lambda value: any(m in value for m in modes))
            links = links.loc[selected].copy()
            if links.empty:
                raise ValueError("No links are available for the requested modes")
        self.links = links
        self.centroids = None if centroids is None else np.array(centroids, copy=True)
        self.turn_restrictions = None if turn_restrictions is None else turn_restrictions.copy(deep=True)
        if modes is not None and self.turn_restrictions is not None and "modes" in self.turn_restrictions:
            selected = (
                self.turn_restrictions.modes.fillna("").astype(str).map(lambda value: any(m in value for m in modes))
            )
            self.turn_restrictions = self.turn_restrictions.loc[selected].copy()
        self.allow_uturns = allow_uturns
        self.lonlat = None if lonlat is None else lonlat.copy(deep=True)
        self._graphs = {}

    def copy(self):
        """Copy source data without retaining prepared graphs."""
        network = Network(
            self.links,
            modes=self.modes,
            centroids=self.centroids,
            turn_restrictions=self.turn_restrictions,
            allow_uturns=self.allow_uturns,
            lonlat=self.lonlat,
        )
        network._node_ids = np.union1d(self._node_ids, network._node_ids)
        return network

    @property
    def fields(self):
        columns = set(self.links.columns)
        return sorted(
            {c for c in columns if not c.endswith(("_ab", "_ba"))}
            | {c[:-3] for c in columns if c.endswith("_ab") and c[:-3] + "_ba" in columns}
        )

    def _directed_indices(self):
        direction = self.links.direction.to_numpy()
        ab = np.flatnonzero(direction != -1)
        ba = np.flatnonzero(direction != 1)
        rows = np.concatenate([ab, ba])
        directions = np.concatenate([np.ones(len(ab), dtype=np.int8), -np.ones(len(ba), dtype=np.int8)])
        order = np.lexsort((directions, self.links.link_id.to_numpy(dtype=np.int64)[rows]))
        return rows[order], directions[order]

    @property
    def link_ids(self):
        rows, _ = self._directed_indices()
        return self.links.link_id.to_numpy(dtype=np.int64)[rows]

    @property
    def directions(self):
        return self._directed_indices()[1]

    def field_values(self, field):
        """Return numeric values in ascending link-ID and direction order."""
        if field not in self.fields:
            raise ValueError(f"'{field}' not in network")
        rows, directions = self._directed_indices()
        values = np.empty(len(rows), dtype=np.float64)
        for direction, suffix in ((1, "_ab"), (-1, "_ba")):
            column = field + suffix if field + suffix in self.links else field
            mask = directions == direction
            try:
                values[mask] = pd.to_numeric(self.links.iloc[rows[mask]][column], errors="raise").to_numpy(
                    dtype=np.float64
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Field {field} must contain numeric values") from exc
        values.flags.writeable = False
        return values

    def validate_mode(self, mode):
        if not isinstance(mode, str) or not mode or (self.modes is not None and mode not in self.modes):
            raise ValueError(f"Mode '{mode}' is not available in this network")

    def mode_mask(self, mode):
        self.validate_mode(mode)
        if "modes" not in self.links:
            return np.ones(len(self.links), dtype=bool)
        return self.links.modes.fillna("").astype(str).str.contains(mode, regex=False).to_numpy()

    def validate_centroids(self, centroids):
        values = np.asarray(centroids)
        nodes = np.union1d(self._node_ids, np.union1d(self.links.a_node, self.links.b_node))
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError("Centroids must be a one-dimensional array of integers")
        if not len(values) or np.any(values <= 0) or len(np.unique(values)) != len(values):
            raise ValueError("Centroids must be nonempty, unique and positive")
        if not np.isin(values, nodes).all():
            raise ValueError("Demand centroids are not present in the network")

    def _validate_turns(self, mode):
        """Check mode-specific turns without preparing routing topology."""
        from aequilibrae.paths.graph import GraphBase

        turns = self.turn_restrictions
        if turns is None:
            return
        required = {"from_node", "via_node", "to_node", "penalty"}
        if missing := required - set(turns.columns):
            raise ValueError(f"Turn restrictions DataFrame missing required columns: {missing}")
        if "modes" in turns:
            turns = turns.loc[turns.modes.fillna("").astype(str).str.contains(mode, regex=False)]
        if turns.empty:
            return
        turns.penalty.apply(GraphBase._normalise_turn_penalty)
        links = self.links.loc[self.mode_mask(mode)]
        forward = links.loc[links.direction >= 0, ["a_node", "b_node"]]
        backward = links.loc[links.direction <= 0, ["b_node", "a_node"]].set_axis(["a_node", "b_node"], axis=1)
        edges = pd.MultiIndex.from_frame(pd.concat([forward, backward]))
        for first, second in (("from_node", "via_node"), ("via_node", "to_node")):
            legs = pd.MultiIndex.from_frame(turns[[first, second]])
            if not legs.isin(edges).all():
                raise ValueError(f"Turn restrictions have missing directed legs {first} -> {second} for mode '{mode}'")

    def build_graph(
        self,
        mode,
        cost=None,
        *,
        centroids=None,
        skimming=(),
        compact=False,
        block_centroid_flows=True,
        turn_restrictions=None,
        allow_uturns=None,
    ):
        """Prepare one full or compact graph from the current source tables.

        Costs use the named field. ``Graph.set_costs()`` accepts replacement
        arrays in directed network order, including infinity for link closures.
        Compact graphs retain centroids and expand paths back to source links.
        """
        from aequilibrae.paths.graph import GraphBase, _prepare_network_graph, freeze_graph

        self.validate_mode(mode)
        centroids = self.centroids if centroids is None else np.asarray(centroids)
        if centroids is not None:
            self.validate_centroids(centroids)
        if compact and centroids is None:
            raise ValueError("Compact graphs require centroids")
        if isinstance(skimming, str):
            skimming = [skimming]
        fields = {name: self.field_values(name) for name in dict.fromkeys(skimming)}
        costs = np.zeros(len(self.link_ids)) if cost is None else self.field_values(cost)
        if np.isnan(costs).any() or np.any(costs < 0):
            raise ValueError("Graph costs must be nonnegative and cannot contain NaN")
        builder = GraphBase()
        # Mode-excluded links belong in reporting mappings, not routing topology.
        builder.network = self.links.loc[self.mode_mask(mode), ["link_id", "a_node", "b_node", "direction"]].copy()
        builder.network["modes"] = mode
        if builder.network.empty:
            raise ValueError(f"No links are available for mode '{mode}'")
        _prepare_network_graph(
            builder,
            mode,
            centroids,
            self.turn_restrictions if turn_restrictions is None else turn_restrictions,
            block_centroid_flows,
            self.allow_uturns if allow_uturns is None else allow_uturns,
        )
        return freeze_graph(builder, compact=compact, network=self, costs=costs, fields=fields, cost_name=cost)

    def _assignment_graph(self, configuration):
        """Share assignment topology for classes with the same routing policy."""
        key = (configuration.mode, tuple(configuration.demand.index), configuration.block_centroid_flows)
        if key not in self._graphs:
            self._graphs[key] = self.build_graph(
                configuration.mode,
                centroids=configuration.demand.index,
                compact=True,
                block_centroid_flows=configuration.block_centroid_flows,
            )
        return self._graphs[key]
