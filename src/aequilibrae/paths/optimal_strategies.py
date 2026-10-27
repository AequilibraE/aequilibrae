import logging

import numpy as np
from scipy import sparse


logger = logging.getLogger(__name__)


class OptimalStrategies:
    def __init__(self, assig_spec):
        self.__assig_spec = assig_spec  # type: TransitAssignment

    def execute(self):
        self.__classes = {}
        self.__results = {}
        self.__demand_cols = {}

        for cls in self.__assig_spec.classes:
            cls.results.prepare(cls.graph, cls.matrix)

            self.__results[cls._id] = cls.results
            try:
                # Converts the named matrix to a 0-based COO matrix; the custom zone IDs are mapped below.
                demand = sparse.coo_matrix(cls.matrix[cls.matrix_core], dtype=np.float64)
            except KeyError as e:
                raise ValueError(f"matrix core {cls.matrix_core} is not present in the matrix") from e

            hypergraph = cls.graph.context

            # Matrix rows and columns follow the graph's centroids, so the zone vertices must too
            if not np.array_equal(hypergraph._o_vert_ids, cls.graph.centroids):
                raise ValueError(f"the zone vertices of the graph for '{cls._id}' are not in centroid order")

            # The HyperpathGenerating maps taz_id, taz_id -> O, D, we then take the COO matrix and index into that
            self.__demand_cols[cls._id] = {
                "origin_column": hypergraph._o_vert_ids[demand.row].astype(np.uint32),
                "destination_column": hypergraph._d_vert_ids[demand.col].astype(np.uint32),
                "demand_column": demand.data,
            }

            self.__classes[cls._id] = hypergraph

        for cls in self.__assig_spec.classes:
            # Classes may share a graph, so each sets its own costs right before it is assigned
            hyperpath = cls.apply_costs()

            logger.info(f"Executing S&F assignment  for {cls._id}")

            hyperpath.assign(**self.__demand_cols[cls._id], threads=self.__assig_spec.cores)
            results = self.__results[cls._id]
            results.link_loads = hyperpath._edges["volume"].to_numpy(copy=True)
            results.weighted_waiting_time = hyperpath.weighted_waiting_time
            results.unassigned_demand = hyperpath.unassigned_demand
            results.unassigned_od = self.__zone_pairs(cls, hyperpath.unassigned_od)
            results.skims = hyperpath.skim_matrix if hyperpath._skimming else None

    @staticmethod
    def __zone_pairs(cls, unassigned):
        """Translate vertex IDs of unassigned demand into zone IDs."""
        mapping = cls.graph.od_node_mapping
        o_key, d_key = ("node_id", "node_id") if "node_id" in mapping.columns else ("o_node_id", "d_node_id")
        origins = mapping.set_index(o_key).taz_id
        destinations = mapping.set_index(d_key).taz_id
        return unassigned.assign(
            origin=origins.reindex(unassigned.origin).to_numpy(),
            destination=destinations.reindex(unassigned.destination).to_numpy(),
        )
