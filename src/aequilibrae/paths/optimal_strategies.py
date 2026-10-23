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

            # The HyperpathGenerating maps taz_id, taz_id -> O, D, we then take the COO matrix and index into that
            self.__demand_cols[cls._id] = {
                "origin_column": hypergraph._o_vert_ids[demand.row].astype(np.uint32),
                "destination_column": hypergraph._d_vert_ids[demand.col].astype(np.uint32),
                "demand_column": demand.data,
            }

            self.__classes[cls._id] = hypergraph

        for cls in self.__assig_spec.classes:
            hyperpath = self.__classes[cls._id]

            logger.info(f"Executing S&F assignment  for {cls._id}")

            hyperpath.assign(**self.__demand_cols[cls._id], threads=self.__assig_spec.cores)
            self.__results[cls._id].link_loads = hyperpath._edges["volume"].values
            if hyperpath._skimming:
                skim = hyperpath.skim_matrix
                # skim.index = cls.graph.centroids[:]
                self.__results[cls._id].skims = skim
