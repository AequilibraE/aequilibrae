"""In-memory network data shared by routing and assignment configurations."""

import numpy as np
import pandas as pd

from aequilibrae.paths.graph import Graph, _prepare_network_graph


class NetworkData:
    """Own link data and turn restrictions without preparing routing graphs.

    ``links`` uses the project schema, including paired ``field_ab`` and
    ``field_ba`` columns for directional attributes. Graphs built from this
    snapshot own their data and can be configured independently.
    """

    def __init__(self, links, modes=None, centroids=None, turn_restrictions=None, allow_uturns=False, lonlat=None):
        required = {"link_id", "a_node", "b_node", "direction", "modes"}
        missing = required - set(links.columns)
        if missing:
            raise ValueError(f"Network is missing required fields: {sorted(missing)}")
        self.links = links.copy(deep=True)
        if self.links.empty or self.links.link_id.duplicated().any():
            raise ValueError("Network must contain links with unique IDs")
        if not self.links.direction.isin([-1, 0, 1]).all():
            raise ValueError("Link directions must be -1, 0 or 1")
        available = set("".join(self.links.modes.fillna("").astype(str)))
        self.modes = tuple(sorted(available) if modes is None else ([modes] if isinstance(modes, str) else modes))
        self.centroids = None if centroids is None else np.array(centroids, copy=True)
        self.turn_restrictions = None if turn_restrictions is None else turn_restrictions.copy(deep=True)
        self.allow_uturns = allow_uturns
        self.lonlat = None if lonlat is None else lonlat.copy(deep=True)

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

    def field_values(self, field):
        """Return numeric values in ascending link-ID and direction order."""
        return self._field_values(field, *self._directed_indices())

    def _field_values(self, field, rows, directions):
        if field not in self.fields:
            raise ValueError(f"'{field}' not in network")
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
        return values

    def _prepare(self):
        return _PreparedNetwork(self)

    def validate_mode(self, mode):
        if mode not in self.modes:
            raise ValueError(f"Mode '{mode}' is not available in this network")

    def validate_centroids(self, centroids):
        values = np.asarray(centroids)
        nodes = np.union1d(self.links.a_node, self.links.b_node)
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError("Centroids must be a one-dimensional array of integers")
        if not len(values) or np.any(values <= 0) or len(np.unique(values)) != len(values):
            raise ValueError("Centroids must be nonempty, unique and positive")
        if not np.isin(values, nodes).all():
            raise ValueError("Demand centroids are not present in the network")

    def build_graph(self, mode, cost, *, centroids=None, skimming=None, block_centroid_flows=True):
        """Prepare an independent graph for a mode, cost and centroid order."""
        self.validate_mode(mode)
        rows, directions = self._directed_indices()
        for field in dict.fromkeys([cost, *(skimming or [])]):
            self._field_values(field, rows, directions)
        centroids = self.centroids if centroids is None else centroids
        if centroids is not None:
            self.validate_centroids(centroids)
        graph = Graph()
        graph.network = self.links.copy(deep=True)
        _prepare_network_graph(
            graph, mode, centroids, self.turn_restrictions, block_centroid_flows, self.allow_uturns
        )
        if self.lonlat is not None:
            graph.lonlat_index = self.lonlat.loc[graph.all_nodes]
        graph.set_graph(cost)
        graph.set_skimming(list(skimming or []))
        return graph


class _PreparedNetwork:
    """One execution's attribute snapshot and shared graph cache."""

    def __init__(self, network):
        self.data = NetworkData(
            network.links,
            modes=network.modes,
            centroids=network.centroids,
            turn_restrictions=network.turn_restrictions,
            allow_uturns=network.allow_uturns,
        )
        self._rows, self.directions = self.data._directed_indices()
        self.link_ids = self.data.links.link_id.to_numpy(dtype=np.int64)[self._rows].copy()
        self._fields = {}
        self._graphs = {}

    def field_values(self, field):
        if field not in self._fields:
            values = self.data._field_values(field, self._rows, self.directions)
            values.flags.writeable = False
            self._fields[field] = values
        return self._fields[field]

    def graph(self, configuration):
        from aequilibrae.paths.graph import prepare_assignment_graph

        # Turn restrictions and U-turn policy belong to this execution's snapshot.
        key = (configuration.mode, tuple(configuration.demand.index), configuration.block_centroid_flows)
        if key not in self._graphs:
            self._graphs[key] = prepare_assignment_graph(
                self.data.links,
                configuration.mode,
                configuration.demand.index,
                self.data.turn_restrictions,
                configuration.block_centroid_flows,
                self.data.allow_uturns,
            )
        return self._graphs[key]
