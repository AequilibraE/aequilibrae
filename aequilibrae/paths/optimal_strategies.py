import logging
from scipy import sparse
import numpy as np
from aequilibrae.paths.public_transport import HyperpathGenerating


class OptimalStrategies:
    def __init__(self, assig_spec):
        from aequilibrae.paths import TransitAssignment

        self.__assig_spec = assig_spec  # type: TransitAssignment
        self.__logger = assig_spec.logger
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
                trav_time=assig_spec._config["Time field"],
                freq=assig_spec._config["Frequency field"],
            )

            demand = sparse.coo_matrix(cls.matrix.matrix[cls.matrix_core], dtype=np.float64)

            # Since the aeq matrix indexes based on centroids, and the transit graph can make the destinction between origins and destinations,
            # We need to translate the index of the cols in to the destination node_ids for the assignment
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

    def execute(self):
        for cls_id, hyperpath in self.__classes.items():
            self.__logger.info(f"Executing S&F assignment  for {cls_id}")

            hyperpath.assign(**self.__demand_cols[cls_id], threads=self.__assig_spec.cores)
            self.__results[cls_id].link_loads = hyperpath._edges["volume"].values

    # def run(self, origin=None, destination=None, volume=None):
    #     for cls_id, hyperpath in self.__classes.items():
    #         self.__logger.info(f"Executing S&F single run for {cls_id}")

    #         hyperpath.run(origin, destination, volume)
    #         self.__results[cls_id].link_loads.data["volume"] = hyperpath._edges["volume"].values
