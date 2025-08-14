from abc import ABC
from copy import deepcopy

import numpy as np

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths.graph import GraphBase


class TransportClassBase(ABC):  # noqa: B024
    def __init__(self, name: str, graph: GraphBase, matrix: AequilibraeMatrix) -> None:
        """
        Instantiates the class

        :Arguments:
            **name** (:obj:`str`): UNIQUE class name.

            **graph** (:obj:`Graph`): Class/mode-specific graph

            **matrix** (:obj:`AequilibraeMatrix`): Class/mode-specific matrix. Supports multiple user classes
        """
        if not np.array_equal(matrix.index, graph.centroids):
            raise ValueError("Matrix and graph do not have compatible sets of centroids.")

        if matrix.matrix_view.dtype != graph.default_types("float"):
            raise TypeError("Matrix's computational view need to be of type np.float64")
        self._config = {}
        self.graph = graph
        self.logger = graph.logger
        self.matrix = matrix
        self._id = name

        graph_config = {
            "Mode": graph.mode,
            "Block through centroids": graph.block_centroid_flows,
            "Number of centroids": graph.num_zones,
            "Links": graph.num_links,
            "Nodes": graph.num_nodes,
        }
        self._config["Graph"] = str(graph_config)

        mat_config = {
            "Source": matrix.file_path or "",
            "Number of centroids": matrix.zones,
            "Matrix cores": matrix.view_names,
        }
        if len(matrix.view_names) == 1:
            mat_config["Matrix totals"] = {
                nm: float(np.sum(np.nan_to_num(matrix.matrix_view)[:, :])) for nm in matrix.view_names
            }
        else:
            mat_config["Matrix totals"] = {
                nm: float(np.sum(np.nan_to_num(matrix.matrix_view)[:, :, i])) for i, nm in enumerate(matrix.view_names)
            }
        self._config["Matrix"] = str(mat_config)

    @property
    def info(self) -> dict:
        config = deepcopy(self._config)
        return {self._id: config}

