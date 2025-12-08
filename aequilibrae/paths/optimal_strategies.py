import logging
from scipy import sparse
import numpy as np
from aequilibrae.paths.public_transport import HyperpathGenerating


class OptimalStrategies:
    def __init__(self, assig_spec):
        from aequilibrae.paths import TransitAssignment

        self.__assig_spec = assig_spec  # type: TransitAssignment
        self.__logger = assig_spec.logger

    def execute(self):
        self.__classes = {}
        self.__results = {}
        self.__demand_cols = {}

        for cls in self.__assig_spec.classes:
            cls.results.prepare(cls.graph, cls.matrix)

            self.__results[cls._id] = cls.results
            try:
                # converts 0 based array with custom index to COO matrix, ignores custom index
                idx = cls.matrix.view_names.index(cls.matrix_core)
                demand = sparse.coo_matrix(
                    (
                        cls.matrix.matrix_view[:, :, idx]
                        if len(cls.matrix.view_names) > 1
                        else cls.matrix.matrix_view[:, :]
                    ),
                    dtype=np.float64,
                )
            except ValueError as e:
                raise ValueError(
                    f"matrix core {cls.matrix_core} not found in matrix view. Ensure the matrix is prepared and the core exists"
                ) from e

            # Take the COO matrix and lookup the index values (taz_id)
            taz_row = cls.matrix.index[demand.row]
            taz_col = cls.matrix.index[demand.col]
            # Since the aeq matrix indexes based on centroids, and the transit graph can make the distinction between
            # origins and destinations, We need to translate the index of the cols in to the destination node_ids for
            # the assignment
            od_node_mapping = cls.graph.od_node_mapping.copy()
            od_node_mapping["idx"] = od_node_mapping.index
            od_node_mapping = od_node_mapping.set_index("taz_id")

            o_key, d_key = (
                ("node_id", "node_id") if len(cls.graph.od_node_mapping.columns) == 2 else ("o_node_id", "d_node_id")
            )

            # map taz_id, taz_id -> O, D, demand value triplet
            self.__demand_cols[cls._id] = {
                "origin_column": od_node_mapping.loc[taz_row, o_key].to_numpy().astype(np.uint32),
                "destination_column": od_node_mapping.loc[taz_col, d_key].values.astype(np.uint32),
                "demand_column": demand.data,
            }

            self.__classes[cls._id] = HyperpathGenerating(
                cls.graph.graph,
                head="a_node",
                tail="b_node",
                trav_time=self.__assig_spec._config["Time field"],
                freq=self.__assig_spec._config["Frequency field"],
                skim_cols=self.__assig_spec._config["Skimming Fields"],
                o_vert_ids=od_node_mapping[o_key].to_numpy(),  # taz_id
                d_vert_ids=od_node_mapping[d_key].to_numpy(),  # node_id for destination in the above taz_id
                nodes_to_indices=cls.graph.nodes_to_indices,
            )

        for cls in self.__assig_spec.classes:
            hyperpath = self.__classes[cls._id]

            self.__logger.info(f"Executing S&F assignment  for {cls._id}")

            hyperpath.assign(**self.__demand_cols[cls._id], threads=self.__assig_spec.cores)
            self.__results[cls._id].link_loads = hyperpath._edges["volume"].values
            if hyperpath._skimming:
                skim = hyperpath.skim_matrix
                # skim.index = cls.graph.centroids[:]
                self.__results[cls._id].skims = skim
