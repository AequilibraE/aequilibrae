import logging

import numpy as np

from aequilibrae.paths.cython.demand import assignment_demand


logger = logging.getLogger(__name__)


class OptimalStrategies:
    def __init__(self, assig_spec):
        self.__assig_spec = assig_spec  # type: TransitAssignment

    def execute(self):
        demand = {}
        for cls in self.__assig_spec.classes:
            if len(cls.matrix) != 1:
                raise ValueError("Transit assignment requires exactly one demand matrix per class")

            # Matrix rows and columns follow the graph's centroids, so the zone vertices must too
            if not np.array_equal(cls.graph.context._o_vert_ids, cls.graph.centroids):
                raise ValueError(f"the zone vertices of the graph for '{cls._id}' are not in centroid order")

            demand[cls._id] = assignment_demand(cls.matrix, cls.graph.centroids, transpose=True)
            cls.results.prepare(cls.graph, cls.matrix)

        for cls in self.__assig_spec.classes:
            # Classes may share a graph, so each sets its own costs right before it is assigned
            hyperpath = cls.apply_costs()

            logger.info(f"Executing S&F assignment  for {cls._id}")

            hyperpath.assign(demand[cls._id], threads=self.__assig_spec.cores)
            results = cls.results
            results.link_loads = hyperpath._edges["volume"].to_numpy(copy=True)
            results.weighted_waiting_time = hyperpath.weighted_waiting_time
            results.unassigned_demand = hyperpath.unassigned_demand
            results.unassigned_od = self.__zone_pairs(cls, hyperpath.unassigned_od)
            # Skims share the kernel's packed dense buffer and use the demand's zone IDs
            results.skims = None
            if hyperpath._skimming:
                results.skims = hyperpath.skim_matrix.to_matrix(cls.matrix.index, cls.matrix.index_name)

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
