import itertools
import logging
import pathlib
import socket
from typing import List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
from aequilibrae.context import get_active_project
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths.graph import Graph, _get_graph_to_network_mapping
from aequilibrae.paths.route_choice_set import RouteChoiceSet


class RouteChoice:
    all_algorithms = ["bfsle", "lp", "link-penalisation", "link-penalization"]
    default_paramaters = {
        "beta": 1.0,
        "theta": 1.0,
        "penalty": 1.1,
        "seed": 0,
        "max_routes": 0,
        "max_depth": 0,
    }

    def __init__(self, graph: Graph, matrix: AequilibraeMatrix, project=None):
        self.paramaters = self.default_paramaters.copy()
        self.procedure_id = uuid4().hex

        proj = project or get_active_project(must_exist=False)
        self.project = proj

        self.logger = proj.logger if proj else logging.getLogger("aequilibrae")

        self.cores: int = 0
        self.graph = graph
        self.matrix = matrix
        self.__rc = None

        self.schema = RouteChoiceSet.schema
        self.psl_schema = RouteChoiceSet.psl_schema

        self.compact_link_loads: Optional[np.array] = None
        self.link_loads: Optional[np.array] = None
        self.results: Optional[pa.Table] = None
        self.where: Optional[pathlib.Path] = None
        self.save_path_files: bool = False

        self.nodes: Optional[Union[List[int], List[Tuple[int, int]]]] = None

        self._config = {}

    def set_algorithm(self, algorithm: str):
        """
        Chooses the assignment algorithm.
        Options are, 'bfsle' for breadth first search with link removal, or 'link-penalisation'/'link-penalization'.

        BFSLE implemenation based on "Route choice sets for very high-resolution data" by Nadine Rieser-Schüssler,
        Michael Balmer & Kay W. Axhausen (2013).
        https://doi.org/10.1080/18128602.2012.671383

        'lp' is also accepted as an alternative to 'link-penalisation'

        :Arguments:
            **algorithm** (:obj:`str`): Algorithm to be used
        """
        algo_dict = {i: i for i in self.all_algorithms}
        algo_dict["lp"] = "link-penalisation"
        algo_dict["link-penalization"] = "link-penalisation"
        algo = algo_dict.get(algorithm.lower())

        if algo is None:
            raise AttributeError(f"Assignment algorithm not available. Choose from: {','.join(self.all_algorithms)}")

        self.algorithm = algo
        self._config["Algorithm"] = algo

    def set_cores(self, cores: int) -> None:
        """Allows one to set the number of cores to be used

            Inherited from :obj:`AssignmentResultsBase`

        :Arguments:
            **cores** (:obj:`int`): Number of CPU cores to use
        """
        self.cores = cores

    def set_paramaters(self, **kwargs):
        """
        Sets the parameters for the route choice.

        "beta", "theta", and "seed" are BFSLE specific parameters and will have no effect on link penalisation.
        "penalty" is a link penalisation specific parameter and will have no effect on BFSLE.

        Setting `max_depth`, while not required, is strongly recommended to prevent runaway algorithms.

        - When using BFSLE `max_depth` corresponds to the maximum height of the graph of graphs. It's value is
            largely dependent on the size of the paths within the network. For very small networks a value of 10
            is a recommended starting point. For large networks a good starting value is 5. Increase the value
            until the number of desired routes is being consistently returned.

        - When using LP, `max_depth` corresponds to the maximum number of iterations performed. While not enforced,
            it should be higher than `max_routes`. It's value is dependent on the magnitude of the cost field,
            specifically it's related to the log base `penalty` of the ratio of costs between two alternative routes.

        :Arguments:
            **kwargs** (:obj:`dict`): Dictionary with all parameters for the algorithm
        """

        if any(key not in self.default_paramaters for key in kwargs.keys()):
            raise ValueError("Invalid parameter provided")

        self.paramaters = self.default_paramaters | kwargs

    def set_save_path_files(self, save_it: bool) -> None:
        """Turn path saving on or off.

        :Arguments:
            **save_it** (:obj:`bool`): Boolean to indicate whether paths should be saved
        """
        self.save_path_files = save_it
        raise NotImplementedError()

    def set_save_routes(self, where: Optional[str] = None) -> None:
        """
        Set save path for route choice resutls. Provide ``None`` to disable.

        **warning** enabling route saving will disable in memory results. Viewing the results will read the results
        from disk first.

        :Arguments:
            **save_it** (:obj:`bool`): Boolean to indicate whether routes should be saved
        """
        self.where = pathlib.Path(where) if where is not None else None

    def prepare(self, nodes: Union[List[int], List[Tuple[int, int]]]):
        """
        Prepare OD pairs for batch computation.

        :Arguments:
            **nodes** (:obj:`Union[list[int], list[tuple[int, int]]]`): List of node IDs to operate on. If a 1D list is
                provided, OD pairs are taken to be all pair permutations of the list. If a list of pairs is provided
                OD pairs are taken as is. All node IDs must be present in the compressed graph. To make a node ID
                always appear in the compressed graph add it as a centroid. Duplicates will be dropped on execution.
        """
        if len(nodes) == 0:
            raise ValueError("`nodes` list-like empty.")

        if isinstance(nodes[0], tuple):
            # Selection of OD pairs
            if any(len(x) != 2 for x in nodes):
                raise ValueError("`nodes` list contains non-pair elements")
            self.nodes = nodes

        elif isinstance(nodes[0], int):
            self.nodes = list(itertools.permutations(nodes, r=2))

    def execute_single(self, origin: int, destination: int, path_size_logit: bool = False):
        if self.__rc is None:
            self.__rc = RouteChoiceSet(self.graph)

        self.results = None
        return self.__rc.run(
            origin,
            destination,
            bfsle=self.algorithm == "bfsle",
            path_size_logit=path_size_logit,
            cores=self.cores,
            **self.paramaters,
        )

    def execute(self, path_size_logit: bool = False):
        if self.__rc is None:
            self.__rc = RouteChoiceSet(self.graph)

        self.results = None
        return self.__rc.batched(
            self.nodes,
            bfsle=self.algorithm == "bfsle",
            path_size_logit=path_size_logit,
            cores=self.cores,
            **self.paramaters,
        )

    def info(self) -> dict:
        """Returns information for the transit assignment procedure

        Dictionary contains keys  'Algorithm', 'Matrix totals', 'Computer name', 'Procedure ID'.

        The classes key is also a dictionary with all the user classes per transit class and their respective
        matrix totals

        :Returns:
            **info** (:obj:`dict`): Dictionary with summary information
        """

        matrix_totals = {nm: np.sum(self.matrix.matrix_view[:, :, i]) for i, nm in enumerate(self.matrix.view_names)}

        info = {
            "Algorithm": self.algorithm,
            "Matrix totals": matrix_totals,
            "Computer name": socket.gethostname(),
            "Procedure ID": self.procedure_id,
            "Parameters": self.paramaters,
        }
        return info

    def log_specification(self):
        self.logger.info("Route Choice specification")
        self.logger.info(self._config)

    def get_results(self):
        """Returns the results of the route choice procedure

        Returns a table of OD pairs to lists of link IDs for each OD pair provided (as columns). Represents paths from ``origin`` to ``destination``.

        :Returns:
            **results** (:obj:`pa.Table`): Table with the results of the route choice procedure

        """
        if self.results is None:
            try:
                self.results = self.__rc.get_results()
            except RuntimeError as err:
                if self.where is None:
                    raise ValueError("Route choice results not computed and read/save path not specificed") from err
                self.results = pa.dataset.dataset(
                    self.where, format="parquet", partitioning=pa.dataset.HivePartitioning(self.schema)
                )

        return self.results

    def get_load_results(
        self,
        which: str = "uncompressed",
        clamp: bool = True,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame]]:
        """
        Translates the link loading results from the graph format into the network format.

        :Returns:
            **dataset** (:obj:`tuple[pd.DataFrame]`): Tuple of uncompressed and compressed AequilibraE data with the link loading results.
        """

        if not isinstance(which, str) or which not in ["uncompressed", "compressed", "both"]:
            raise ValueError("`which` argument must be one of ['uncompressed', 'compressed', 'both']")

        compressed = which == "both" or which == "compressed"
        uncompressed = which == "both" or which == "uncompressed"

        fields = self.matrix.names

        tmp = self.__rc.link_loading(self.matrix, self.save_path_files)
        if isinstance(tmp, dict):
            self.link_loads = {k: v[0] for k, v in tmp.items()}
            self.compact_link_loads = {k: v[1] for k, v in tmp.items()}
        else:
            self.link_loads = {fields[0]: tmp[0]}
            self.compact_link_loads = {fields[0]: tmp[1]}

        if clamp:
            for v in itertools.chain(self.link_loads.values(), self.compact_link_loads.values()):
                v[(v < 1e-15)] = 0.0

        # Get a mapping from the compressed graph to/from the network graph
        m = _get_graph_to_network_mapping(self.graph.graph.link_id.values, self.graph.graph.direction.values)
        m_compact = _get_graph_to_network_mapping(
            self.graph.compact_graph.link_id.values, self.graph.compact_graph.direction.values
        )

        lids = np.unique(self.graph.graph.link_id.values)
        compact_lids = np.unique(self.graph.compact_graph.link_id.values)
        # Create a data store with a row for each uncompressed link
        if uncompressed:
            uncompressed_df = pd.DataFrame(
                {"link_id": lids}
                | {k + dir: np.zeros(lids.shape) for k in self.link_loads.keys() for dir in ["_ab", "_ba"]}
            )
            for k, v in self.link_loads.items():
                # Directional Flows
                uncompressed_df[k + "_ab"].values[m.network_ab_idx] = np.nan_to_num(v[m.graph_ab_idx])
                uncompressed_df[k + "_ba"].values[m.network_ba_idx] = np.nan_to_num(v[m.graph_ba_idx])

                # Tot Flow
                uncompressed_df[k + "_tot"] = np.nan_to_num(uncompressed_df[k + "_ab"].values) + np.nan_to_num(
                    uncompressed_df[k + "_ba"].values
                )

        if compressed:
            compressed_df = pd.DataFrame(
                {"link_id": compact_lids}
                | {
                    k + dir: np.zeros(compact_lids.shape)
                    for k in self.compact_link_loads.keys()
                    for dir in ["_ab", "_ba"]
                }
            )
            for k, v in self.compact_link_loads.items():
                compressed_df[k + "_ab"].values[m_compact.network_ab_idx] = np.nan_to_num(v[m_compact.graph_ab_idx])
                compressed_df[k + "_ba"].values[m_compact.network_ba_idx] = np.nan_to_num(v[m_compact.graph_ba_idx])

                # Tot Flow
                compressed_df[k + "_tot"] = np.nan_to_num(compressed_df[k + "_ab"].values) + np.nan_to_num(
                    compressed_df[k + "_ba"].values
                )

        if uncompressed and not compressed:
            return uncompressed_df
        elif not uncompressed and compressed:
            return compressed_df
        else:
            return uncompressed_df, compressed_df

    def get_select_link_results(self) -> pd.DataFrame:
        raise NotImplementedError()
