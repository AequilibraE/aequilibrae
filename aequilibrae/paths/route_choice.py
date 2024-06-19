import itertools
import warnings
import logging
import pathlib
import socket
import sqlite3
from datetime import datetime
from typing import List, Optional, Tuple, Union, Dict
from uuid import uuid4
from functools import cached_property

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
        "generic": {"seed": 0, "max_routes": 0, "max_depth": 0, "max_misses": 100, "penalty": 1.01, "cutoff_prob": 0.0},
        "link-penalisation": {},
        "bfsle": {"penalty": 1.0},
    }

    def __init__(self, graph: Graph, matrix: Optional[AequilibraeMatrix] = None, project=None):
        self.parameters = self.default_paramaters.copy()
        self.procedure_id = None
        self.procedure_date = None

        proj = project or get_active_project(must_exist=False)
        self.project = proj

        self.logger = proj.logger if proj else logging.getLogger("aequilibrae")

        self.cores: int = 0
        self.graph = graph
        self.matrix = matrix

        self.schema = RouteChoiceSet.schema
        self.psl_schema = RouteChoiceSet.psl_schema

        self.compact_link_loads: Optional[Dict[str, np.array]] = None
        self.link_loads: Optional[Dict[str, np.array]] = None

        self.sl_compact_link_loads: Optional[Dict[str, np.array]] = None
        self.sl_link_loads: Optional[Dict[str, np.array]] = None

        self.results: Optional[pa.Table] = None
        self.where: Optional[pathlib.Path] = None
        self.save_path_files: bool = False

        self.nodes: Optional[Union[List[int], List[Tuple[int, int]]]] = None

        self._config = {}
        self._selected_links = {}

    @cached_property
    def __rc(self) -> RouteChoiceSet:
        return RouteChoiceSet(self.graph)

    def set_choice_set_generation(self, /, algorithm: str, **kwargs) -> None:
        """Chooses the assignment algorithm and set parameters.
        Options for algorithm are, 'bfsle' for breadth first search with link removal, or 'link-penalisation'/'link-penalization'.

        BFSLE implementation based on "Route choice sets for very high-resolution data" by Nadine Rieser-Schüssler,
        Michael Balmer & Kay W. Axhausen (2013).
        https://doi.org/10.1080/18128602.2012.671383

        'lp' is also accepted as an alternative to 'link-penalisation'

        Setting the parameters for the route choice:

        `seed` is a BFSLE specific parameters.

        Setting `max_depth` or `max_misses`, while not required, is strongly recommended to prevent runaway algorithms.
        `max_misses` is the maximum amount of duplicate routes found per OD pair. If it is exceeded then the route set
        if returned with fewer than `max_routes`. It has a default value of `100`.

        - When using BFSLE `max_depth` corresponds to the maximum height of the graph of graphs. It's value is
            largely dependent on the size of the paths within the network. For very small networks a value of 10
            is a recommended starting point. For large networks a good starting value is 5. Increase the value
            until the number of desired routes is being consistently returned. If it is exceeded then the route set
            if returned with fewer than `max_routes`.

        - When using LP, `max_depth` corresponds to the maximum number of iterations performed. While not enforced,
            it should be higher than `max_routes`. It's value is dependent on the magnitude of the cost field,
            specifically it's related to the log base `penalty` of the ratio of costs between two alternative routes.
            If it is exceeded then the route set if returned with fewer than `max_routes`.

        Additionally BFSLE has the option to incorporate link penalisation. Every link in all routes found at a depth
        are penalised with the `penalty` factor for the next depth. So at a depth of 0 no links are penalised nor
        removed. At depth 1, all links found at depth 0 are penalised, then the links marked for removal are removed.
        All links in the routes found at depth 1 are then penalised for the next depth. The penalisation compounds.
        Pass set `penalty=1.0` to disable.

        When performing an assignment, `cutoff_prob` can be provided to exclude routes from the path-sized logit model.
        The `cutoff_prob` is used to compute an inverse binary logit and obtain a max difference in utilities. If a
        paths total cost is greater than the minimum cost path in the route set plus the max difference, the route is
        excluded from the PSL calculations. The route is still returned, but with a probability of 0.0.

        The `cutoff_prob` should be in the range [0, 1]. It is then rescaled internally to [0.5, 1] as probabilities
        below 0.5 produce negative differences in utilities because the choice is between two routes only, one of
        which is the shortest path. A higher `cutoff_prob` includes less routes. A value of `1.0` will only include
        the minimum cost route. A value of `0.0` includes all routes.

        :Arguments:
            **algorithm** (:obj:`str`): Algorithm to be used
            **kwargs** (:obj:`dict`): Dictionary with all parameters for the algorithm
        """
        algo_dict = {i: i for i in self.all_algorithms}
        algo_dict["lp"] = "link-penalisation"
        algo_dict["link-penalization"] = "link-penalisation"
        algo = algo_dict.get(algorithm.lower())

        if algo is None:
            raise AttributeError(f"Assignment algorithm not available. Choose from: {','.join(self.all_algorithms)}")

        defaults = self.default_paramaters["generic"] | self.default_paramaters[algo]
        for key in kwargs.keys():
            if key not in defaults:
                raise ValueError(f"Invalid parameter `{key}` provided for algorithm `{algo}`")

        self.algorithm = algo
        self._config["Algorithm"] = algo

        self.parameters = defaults | kwargs

    def set_cores(self, cores: int) -> None:
        """Allows one to set the number of cores to be used

            Inherited from :obj:`AssignmentResultsBase`

        :Arguments:
            **cores** (:obj:`int`): Number of CPU cores to use
        """
        self.cores = cores

    def set_save_path_files(self, save_it: bool) -> None:
        """turn path saving on or off.

        :Arguments:
            **save_it** (:obj:`bool`): Boolean to indicate whether paths should be saved
        """
        self.save_path_files = save_it
        raise NotImplementedError()

    def set_save_routes(self, where: Optional[str] = None) -> None:
        """
        Set save path for route choice results. Provide ``None`` to disable.

        **warning** enabling route saving will disable in memory results. Viewing the results will read the results
        from disk first.

        :Arguments:
            **save_it** (:obj:`bool`): Boolean to indicate whether routes should be saved
        """
        if where is not None:
            where = pathlib.Path(where)
            if not where.exists():
                raise ValueError(f"Path does not exist `{where}`")
        self.where = where

    def prepare(self, nodes: Union[List[int], List[Tuple[int, int]]]) -> None:
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

        if all(
            isinstance(pair, tuple)
            and len(pair) == 2
            and isinstance(pair[0], (int, np.integer))
            and isinstance(pair[1], (int, np.integer))
            for pair in nodes
        ):
            self.nodes = nodes

        elif len(nodes) > 1 and all(isinstance(x, (int, np.unsignedinteger)) for x in nodes):
            self.nodes = list(itertools.permutations(nodes, r=2))
        else:
            raise ValueError(f"{type(nodes)} or {type(nodes[0])} for not valid types for the `prepare` method")

    def execute_single(self, origin: int, destination: int, perform_assignment: bool = False) -> List[Tuple[int]]:
        """
        Generate route choice sets between `origin` and `destination`, potentially performing an assignment.

        Does not require preparation.

        Node IDs must be present in the compressed graph. To make a node ID always appear in the compressed
        graph add it as a centroid.

        :Arguments:
            **origin** (:obj:`int`): Origin node ID.
            **destination** (:obj:`int`): Destination node ID.
            **perform_assignment** (:obj:`bool`): Whether or not to perform an assignment. Default `False`.

        :Returns:
            ***route set** (:obj:`List[Tuple[int]]`): A list of routes as tuples of link IDs.
        """
        self.procedure_id = uuid4().hex
        self.procedure_date = str(datetime.today())

        self.results = None
        return self.__rc.run(
            origin,
            destination,
            bfsle=self.algorithm == "bfsle",
            path_size_logit=perform_assignment,
            cores=self.cores,
            where=str(self.where) if self.where is not None else None,
            **self.parameters,
        )

    def execute(self, perform_assignment: bool = True) -> None:
        """
        Generate route choice sets between the previously supplied nodes, potentially performing an assignment.

        Node IDs must be present in the compressed graph. To make a node ID always appear in the compressed
        graph add it as a centroid.

        To access results see `RouteChoice.get_results()`.

        :Arguments:
            **perform_assignment** (:obj:`bool`): Whether or not to perform an assignment. Default `False`.
        """
        if self.nodes is None:
            raise ValueError(
                "to perform batch route choice generation you must first prepare with the selected nodes. See `RouteChoice.prepare()`"
            )

        self.procedure_date = str(datetime.today())

        self.results = None
        self.__rc.batched(
            self.nodes,
            bfsle=self.algorithm == "bfsle",
            path_size_logit=perform_assignment,
            cores=self.cores,
            where=str(self.where) if self.where is not None else None,
            **self.parameters,
        )

    def info(self) -> dict:
        """Returns information for the transit assignment procedure

        Dictionary contains keys  'Algorithm', 'Matrix totals', 'Computer name', 'Procedure ID', 'Parameters', and
        'Select links'.

        The classes key is also a dictionary with all the user classes per transit class and their respective
        matrix totals

        :Returns:
            **info** (:obj:`dict`): Dictionary with summary information
        """

        if self.matrix is None:
            matrix_totals = {}
        elif len(self.matrix.view_names) == 1:
            matrix_totals = {self.matrix.view_names[0]: np.sum(self.matrix.matrix_view[:, :])}
        else:
            matrix_totals = {
                nm: np.sum(self.matrix.matrix_view[:, :, i]) for i, nm in enumerate(self.matrix.view_names)
            }

        info = {
            "Algorithm": self.algorithm,
            "Matrix totals": matrix_totals,
            "Computer name": socket.gethostname(),
            "Procedure ID": self.procedure_id,
            "Parameters": self.parameters,
            "Select links": self._selected_links,
        }
        return info

    def log_specification(self):
        self.logger.info("Route Choice specification")
        self.logger.info(self._config)

    def get_results(self) -> Union[pa.Table, pa.dataset.Dataset]:
        """Returns the results of the route choice procedure

        Returns a table of OD pairs to lists of link IDs for each OD pair provided (as columns).
        Represents paths from ``origin`` to ``destination``.

        If `save_routes` was specified then a Pyarrow dataset is returned. The caller is responsible for reading this dataset.

        :Returns:
            **results** (:obj:`pa.Table`): Table with the results of the route choice procedure
        """
        if self.results is None:
            try:
                self.results = self.__rc.get_results()
            except RuntimeError as err:
                if self.where is None:
                    raise ValueError("Route choice results not computed and read/save path not specified") from err
                self.results = pa.dataset.dataset(
                    self.where, format="parquet", partitioning=pa.dataset.HivePartitioning(self.schema)
                )

        return self.results

    def get_load_results(
        self, compressed_graph_results=False
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Translates the link loading results from the graph format into the network format.

        :Arguments:
            **compressed_graph_results** (:obj:`bool`): Whether we should return assignment results for the
            compressed graph. Only use this option if you are SURE you know what you are doing. Default `False`.

        :Returns:
            **dataset** (:obj:`Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]`):
                A tuple of uncompressed and compressed link loading results as DataFrames.
                Columns are the matrix name concatenated direction.
        """

        if self.matrix is None:
            raise ValueError(
                "AequilibraE matrix was not initially provided. To perform link loading set the `RouteChoice.matrix` attribute."
            )

        tmp = self.__rc.link_loading(self.matrix, self.save_path_files)

        if not compressed_graph_results:
            self.link_loads = {k: v[0] for k, v in tmp.items()}
            # Create a data store with a row for each uncompressed link
            m = _get_graph_to_network_mapping(self.graph.graph.link_id.values, self.graph.graph.direction.values)
            lids = np.unique(self.graph.graph.link_id.values)
            uncompressed_df = self.__link_loads_to_df(m, lids, self.link_loads)

            return uncompressed_df
        else:
            self.compact_link_loads = {k: v[1] for k, v in tmp.items()}
            m_compact = _get_graph_to_network_mapping(
                self.graph.compact_graph.link_id.values, self.graph.compact_graph.direction.values
            )
            compact_lids = np.unique(self.graph.compact_graph.link_id.values)
            compressed_df = self.__link_loads_to_df(m_compact, compact_lids, self.compact_link_loads)

            return compressed_df

    def __link_loads_to_df(self, mapping, lids, link_loads):
        df = pd.DataFrame(
            {"link_id": lids} | {k + dir: np.zeros(lids.shape) for k in link_loads.keys() for dir in ["_ab", "_ba"]}
        )
        for k, v in link_loads.items():
            # Directional Flows
            df.iloc[mapping.network_ab_idx, df.columns.get_loc(k + "_ab")] = np.nan_to_num(v[mapping.graph_ab_idx])
            df.iloc[mapping.network_ba_idx, df.columns.get_loc(k + "_ba")] = np.nan_to_num(v[mapping.graph_ba_idx])

            # Tot Flow
            df[k + "_tot"] = df[k + "_ab"] + df[k + "_ba"]

        return df

    def set_select_links(self, links: Dict[str, List[Tuple[int, int]]]):
        """
        Set the selected links. Checks if the links and directions are valid. Translates `links=None` and
        direction into unique link ID used in compact graph.

        Supply `links=None` to disable select link analysis.

        :Arguments:
            **links** (:obj:`Union[None, Dict[str, List[Tuple[int, int]]]]`): name of link set and
             Link IDs and directions to be used in select link analysis.
        """
        self._selected_links = {}

        if links is None:
            del self._config["select_links"]
            return

        max_id = self.graph.compact_graph.id.max() + 1

        for name, link_set in links.items():
            if " " in name:
                warnings.warn("Input string name has a space in it. Replacing with _")
                name = str.join("_", name.split(" "))

            or_set = set()
            for link_ids in link_set:
                # Allow a single int to represent a bidirectional single link AND set. For compatibility, a tuple of
                # length 2 is passed, we assume that's a single (link, dir) pair
                if isinstance(link_ids, int) or (isinstance(link_ids, tuple) and len(link_ids) == 2):
                    link_ids = (link_ids,)

                and_set = set()
                for link in link_ids:
                    # Let a single int in place of an (link, dir) tuple represent a bidirectional link
                    link, dir = (link, 0) if isinstance(link, int) else link

                    if dir == 0:
                        query = (self.graph.graph["link_id"] == link) & (
                            (self.graph.graph["direction"] == -1) | (self.graph.graph["direction"] == 1)
                        )
                    else:
                        query = (self.graph.graph["link_id"] == link) & (self.graph.graph["direction"] == dir)

                    if not query.any():
                        raise ValueError(f"link_id or direction {(link, dir)} is not present within graph.")

                    for comp_id in self.graph.graph[query]["__compressed_id__"].values:
                        # Check for duplicate compressed link ids in the current link set
                        if comp_id == max_id:
                            raise ValueError(
                                f"link ID {link} and direction {dir} is not present in compressed graph. "
                                "It may have been removed during dead-end removal."
                            )
                        elif comp_id in and_set:
                            warnings.warn(
                                "Two input links map to the same compressed link in the network"
                                f", removing superfluous link {link} and direction {dir} with compressed id {comp_id}"
                            )
                        else:
                            and_set.add(comp_id)

                or_set.add(frozenset(and_set))
            self._selected_links[name] = frozenset(or_set)
        self._config["select_links"] = str(links)

    def get_select_link_results(self, compressed_graph_results=False) -> pd.DataFrame:
        """
        Get the select link loading results.

        :Returns:
            **dataset** (:obj:`Tuple[pd.DataFrame, pd.DataFrame]`):
                A tuple of uncompressed and compressed select link loading results as DataFrames.
                Columns are the matrix name concatenated with the select link set and direction.
        """

        if self.matrix is None:
            raise ValueError(
                "AequilibraE matrix was not initially provided. To perform link loading set the `RouteChoice.matrix` attribute."
            )

        tmp = self.__rc.select_link_loading(self.matrix, self._selected_links)

        self.sl_link_loads = {}
        self.sl_compact_link_loads = {}
        self.sl_od_matrix = {}
        for name, sl_res in tmp.items():
            for sl_name, res in sl_res.items():
                mat, (u, c) = res
                self.sl_od_matrix[name + "_" + sl_name] = mat
                self.sl_link_loads[name + "_" + sl_name] = u
                self.sl_compact_link_loads[name + "_" + sl_name] = c

        if not compressed_graph_results:
            # Create a data store with a row for each uncompressed link
            m = _get_graph_to_network_mapping(self.graph.graph.link_id.values, self.graph.graph.direction.values)
            lids = np.unique(self.graph.graph.link_id.values)
            uncompressed_df = self.__link_loads_to_df(m, lids, self.sl_link_loads)

            return uncompressed_df
        else:
            m_compact = _get_graph_to_network_mapping(
                self.graph.compact_graph.link_id.values, self.graph.compact_graph.direction.values
            )
            compact_lids = np.unique(self.graph.compact_graph.link_id.values)
            compressed_df = self.__link_loads_to_df(m_compact, compact_lids, self.sl_compact_link_loads)

            return compressed_df

    def __save_dataframe(self, df, method_name: str, description: str, table_name: str, report: dict, project) -> None:
        self.procedure_id = uuid4().hex
        data = [
            table_name,
            "select link",
            self.procedure_id,
            str(report),
            self.procedure_date,
            description,
        ]

        # sqlite3 context managers only commit, they don't close, oh well
        conn = sqlite3.connect(pathlib.Path(project.project_base_path) / "results_database.sqlite")
        with conn:
            df.to_sql(table_name, conn, index=False)
        conn.close()

        conn = project.connect()
        with conn:
            conn.execute(
                """Insert into results(table_name, procedure, procedure_id, procedure_report, timestamp,
                                                description) Values(?,?,?,?,?,?)""",
                data,
            )
        conn.close()

    def save_link_flows(self, table_name: str, project=None) -> None:
        """
        Saves the link link flows for all classes into the results database.

        :Arguments:
            **table_name** (:obj:`str`): Name of the table being inserted to.
            **project** (:obj:`Project`, `Optional`): Project we want to save the results to.
            Defaults to the active project
        """
        if not project:
            project = self.project or get_active_project()

        df = self.get_load_results()
        info = self.info()
        self.__save_dataframe(
            df,
            "Link loading",
            "Uncompressed link loading results",
            table_name + "_uncompressed",
            info,
            project=project,
        )

    def save_select_link_flows(self, table_name: str, project=None) -> None:
        """
        Saves the select link link flows for all classes into the results database. Additionally, it exports
        the OD matrices into OMX format.

        :Arguments:
            **table_name** (:obj:`str`): Name of the table being inserted to and the name of the
            OpenMatrix file used for OD matrices.
            **project** (:obj:`Project`, `Optional`): Project we want to save the results to.
            Defaults to the active project
        """
        if not project:
            project = self.project or get_active_project()

        u, c = self.get_select_link_results()
        info = self.info()
        self.__save_dataframe(
            u,
            "Select link analysis",
            "Uncompressed select link analysis results",
            table_name + "_uncompressed",
            info,
            project=project,
        )

        self.__save_dataframe(
            c,
            "Select link analysis",
            "Compressed select link analysis results",
            table_name + "_compressed",
            info,
            project=project,
        )

        for k, v in self.sl_od_matrix.items():
            v.to_disk((pathlib.Path(project.project_base_path) / "matrices" / table_name).with_suffix(".omx"), k)
