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
            self.__classes[cls._id] = HyperpathGenerating(
                cls.graph.graph,
                head="a_node",
                tail="b_node",
                trav_time=self.__assig_spec._config["Time field"],
                freq=self.__assig_spec._config["Frequency field"],
                skim_cols=self.__assig_spec._config["Skimming Fields"],
                centroids=cls.graph.centroids,
            )

            try:
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

            # Since the aeq matrix indexes based on centroids, and the transit graph can make the distinction between
            # origins and destinations, We need to translate the index of the cols in to the destination node_ids for
            # the assignment
            if len(cls.graph.od_node_mapping.columns) == 2:
                o_vert_ids = cls.graph.od_node_mapping.iloc[demand.row]["node_id"].values.astype(np.uint32)
                d_vert_ids = cls.graph.od_node_mapping.iloc[demand.col]["node_id"].values.astype(np.uint32)
            else:
                o_vert_ids = cls.graph.od_node_mapping.iloc[demand.row]["o_node_id"].values.astype(np.uint32)
                d_vert_ids = cls.graph.od_node_mapping.iloc[demand.col]["d_node_id"].values.astype(np.uint32)

            self.__demand_cols[cls._id] = {
                "origin_column": o_vert_ids,
                "destination_column": d_vert_ids,
                "demand_column": demand.data,
            }

        for cls_id, hyperpath in self.__classes.items():
            self.__logger.info(f"Executing S&F assignment  for {cls_id}")

            hyperpath.assign(**self.__demand_cols[cls_id], threads=self.__assig_spec.cores)
            self.__results[cls_id].link_loads = hyperpath._edges["volume"].values
            if hyperpath._skimming:
                self.__results[cls_id].skims = hyperpath.skim_matrix
