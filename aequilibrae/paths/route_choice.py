import itertools
import warnings
import logging
import pathlib
import socket
import sqlite3
from datetime import datetime
from typing import List, Optional, Tuple, Union, Dict
from collections.abc import Hashable
from os import path
from uuid import uuid4
from functools import cached_property

import numpy as np
import pandas as pd
import openmatrix as omx
import scipy
from aequilibrae.context import get_active_project
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths.graph import Graph, _get_graph_to_network_mapping
from aequilibrae.paths.cython.route_choice_set import RouteChoiceSet
from aequilibrae.paths.cython.route_choice_set_results import RouteChoiceSetResults
from aequilibrae.matrix.coo_demand import GeneralisedCOODemand
from aequilibrae.utils.db_utils import commit_and_close


class RouteChoice:
    all_algorithms = ["bfsle", "lp", "link-penalisation", "link-penalization"]

    default_parameters = {
        "generic": {
            "seed": 0,
            "max_routes": 0,
            "max_depth": 0,
            "max_misses": 100,
            "penalty": 1.01,
            "cutoff_prob": 0.0,
            "beta": 1.0,
            "store_results": True,
        },
        "link-penalisation": {},
        "bfsle": {"penalty": 1.0},
    }

    demand_index_names = ["origin id", "destination id"]

    def __init__(self, graph: Graph, project=None):
        self.parameters = {}
        self.procedure_id = None
        self.procedure_date = None

        proj = project or get_active_project(must_exist=False)
        self.project = proj

        self.logger = proj.logger if proj else logging.getLogger("aequilibrae")

        self.cores: int = 0
        self.graph = graph
        self.demand = self.__init_demand()

        self.sl_compact_link_loads: Optional[Dict[str, np.array]] = None
        self.sl_link_loads: Optional[Dict[str, np.array]] = None

        self.where: Optional[pathlib.Path] = None
        self.index_name = "route_choice_sl_index"

        self._config = {}
        self._selected_links = {}
        self.sl_link_loading = True

    @cached_property
    def __rc(self) -> RouteChoiceSet:
        return RouteChoiceSet(self.graph)

    def __init_demand(self):
        d = GeneralisedCOODemand(
            *self.demand_index_names, self.graph.nodes_to_indices, shape=(self.graph.num_zones, self.graph.num_zones)
        )
        return d

    def set_choice_set_generation(self, /, algorithm: str = None, **kwargs) -> None:
        """
        Chooses the assignment algorithm and set its parameters.

        Options for algorithm are 'bfsle' for breadth first search with link removal, or
        'link-penalisation'/'link-penalization'. 'lp' is also accepted as an alternative to
        'link-penalisation'. If ``algorithm`` is ``None``, none will be set, but the parameters
        will be updated. This is useful when assigning from path-files.

        BFSLE implementation based on "Route choice sets for very high-resolution data" by
        Nadine Rieser-Schüssler, Michael Balmer & Kay W. Axhausen (2013).
        `DOI: 10.1080/18128602.2012.671383 <https://doi.org/10.1080/18128602.2012.671383>`_.

        Setting the parameters for the route choice:

        * ``seed`` is a BFSLE specific parameters.

        * Although not required, setting ``max_depth`` or ``max_misses``, is strongly
          recommended to prevent runaway algorithms.

        * ``max_misses`` is the maximum amount of duplicate routes found per OD pair. If a set
          of routes is returned in a case where ``max_misses`` is exceeded, the number of routes
          may be fewer than ``max_routes``. Assumes a default value of 100.

        * When using **BFSLE** ``max_depth`` corresponds to the maximum height of the graph.
          It's value is largely dependent on the size of the paths within the network. For
          very small networks a value of 10 is a recommended starting point. For large networks
          a good starting value is 5. Increase the value until the number of desired routes is
          being consistently returned. If a set of routes is returned in a case where
          ``max_depth`` is exceeded, the number of routes may be fewer than ``max_routes``.

        * When using **LP**, ``max_depth`` corresponds to the maximum number of iterations
          performed. While not enforced, it should be higher than ``max_routes``. It's value is
          dependent on the magnitude of the cost field, specifically if it's related to the log
          base ``penalty`` of the ratio of costs between two alternative routes. If a set of
          routes is returned in a case where ``max_depth`` is exceeded, the number of routes
          may be fewer than ``max_routes``.

        * Additionally BFSLE has the option to incorporate link penalisation. Every link in all
          routes found at a depth are penalised with the penalty factor for the next depth. So at
          a depth of 0 no links are penalised nor removed. At depth 1, all links found at depth 0
          are penalised, then the links marked for removal are removed. All links in the routes
          found at depth 1 are then penalised for the next depth. The penalisation compounds.
          Set ``penalty=1.0`` to disable.

        * When performing an assignment, ``cutoff_prob`` can be provided to exclude routes from
          the path-sized logit model. The ``cutoff_prob`` is used to compute an inverse binary
          logit and obtain a max difference in utilities. If a paths total cost is greater than
          the minimum cost path in the route set plus the max difference, the route is excluded
          from the PSL calculations. The route is still returned, but with a probability of 0.0.

        * The ``cutoff_prob`` should be in the range :math:`[0, 1]`. It is then rescaled
          internally to :math:`[0.5, 1]` as probabilities below 0.5 produce negative differences
          in utilities because the choice is between two routes only, one of which is the
          shortest path. A higher ``cutoff_prob`` includes less routes. A value of 1.0 will only
          include the minimum cost route. A value of 0.0 includes all routes.

        :Arguments:
            **algorithm** (:obj:`str`): Algorithm to be used

            **kwargs** (:obj:`dict`): Dictionary with all parameters for the algorithm
        """
        if algorithm is not None:
            algo_dict = {i: i for i in self.all_algorithms}
            algo_dict["lp"] = "link-penalisation"
            algo_dict["link-penalization"] = "link-penalisation"

            sentinel = object()
            algorithm = algo_dict.get(algorithm.lower(), sentinel)

            if algorithm is sentinel:
                raise AttributeError(
                    f"Assignment algorithm not available. Choose from: {','.join(self.all_algorithms)}"
                )

            defaults = self.default_parameters["generic"] | self.default_parameters[algorithm]
            for key in kwargs.keys():
                if key not in defaults:
                    raise ValueError(f"Invalid parameter '{key}' provided for algorithm `{algorithm}`")
        else:
            defaults = self.default_parameters["generic"]
            for key in kwargs.keys():
                if key not in defaults:
                    raise ValueError(f"Invalid or non-generic parameter '{key}' provided")

        self.algorithm = algorithm
        self._config["Algorithm"] = algorithm

        self.parameters = defaults | kwargs

    def set_cores(self, cores: int) -> None:
        """Allows one to set the number of cores to be used

            Inherited from :obj:`AssignmentResultsBase`

        :Arguments:
            **cores** (:obj:`int`): Number of CPU cores to use
        """
        self.cores = cores

    def set_save_routes(self, where: Optional[str] = None) -> None:
        """
        Set save path for route choice results. Provide ``None`` to disable.

        .. warning:

            Enabling route saving will disable in memory results. Viewing the results will read
            the results from disk first.

        :Arguments:
            **save_it** (:obj:`bool`): Boolean to indicate whether routes should be saved
        """

        if where is not None:
            where = pathlib.Path(where)
            if not where.exists():
                raise ValueError(f"Path does not exist `{where}`")
        self.where = where

    def add_demand(self, demand, fill: float = 0.0):
        """
        Add demand DataFrame or matrix for the assignment.

        :Arguments:
            **demand** (:obj:`Union[pd.DataFrame, AequilibraeMatrix]`): Demand to add to
            assignment. If the supplied demand is a DataFrame, it should have a 2-level
            MultiIndex of Origin and Destination node IDs. If an AequilibraE Matrix is
            supplied node IDs will be inferred from the index. Demand values should be either
            ``float32``s or ``float64``s.

            **fill** (:obj:`float`): Value to fill any ``NaN``s with.
        """
        if isinstance(demand, pd.DataFrame):
            self.demand.add_df(demand, fill=fill)
        elif isinstance(demand, AequilibraeMatrix):
            self.demand.add_matrix(demand, fill=fill)
        else:
            raise TypeError(f"unknown argument type '{(type(demand).__name__)}'")

    def prepare(self, nodes: Union[List[int], List[Tuple[int, int]], None] = None) -> None:
        """
        Prepare OD pairs for batch computation.

        :Arguments:
            **nodes** (:obj:`Union[list[int], list[tuple[int, int]]]`): List of node IDs to
            operate on. If a 1D list is provided, OD pairs are taken to be all pair
            permutations of the list. If a list of pairs is provided OD pairs are taken as is.
            All node IDs must be present in the compressed graph. To make a node ID always appear
            in the compressed graph add it as a centroid. Duplicates will be dropped on execution.
            If ``None`` is provided, all OD pairs with non-zero flows will be used.
        """
        if nodes is not None and not self.demand.no_demand():
            raise ValueError("provide either `nodes` or set a `demand` matrix, not both")
        elif nodes is None:
            return
        elif len(nodes) == 0:
            raise ValueError("`nodes` list-like empty.")

        self.demand = self.__init_demand()
        df = pd.DataFrame()
        if all(
            isinstance(pair, tuple)
            and len(pair) == 2
            and isinstance(pair[0], (int, np.integer))
            and isinstance(pair[1], (int, np.integer))
            for pair in nodes
        ):
            df.index = pd.MultiIndex.from_tuples(nodes, name=self.demand_index_names)
        elif len(nodes) > 1 and all(isinstance(x, (int, np.unsignedinteger)) for x in nodes):
            df.index = pd.MultiIndex.from_tuples(itertools.permutations(nodes, r=2), names=self.demand_index_names)
        else:
            raise ValueError(f"{type(nodes)} or {type(nodes[0])} for not valid types for the `prepare` method")

        self.demand.add_df(df)

    def execute_single(self, origin: int, destination: int, demand: float = 0.0) -> List[Tuple[int]]:
        """
        Generate route choice sets between origin and destination, potentially performing an
        assignment.

        Does not require preparation.

        Node IDs must be present in the compressed graph. To make a node ID always appear in
        the compressed graph add it as a centroid.

        :Arguments:
            **origin** (:obj:`int`): Origin node ID.

            **destination** (:obj:`int`): Destination node ID.

            **demand** (:obj:`float`): If provided an assignment will be performed with this demand.

        :Returns:
            **route set** (:obj:`List[Tuple[int]]`): A list of routes as tuples of link IDs.
        """
        self.procedure_id = uuid4().hex
        self.procedure_date = str(datetime.today())

        return self.__rc.run(
            origin,
            destination,
            self.demand.shape,
            demand=demand,
            bfsle=self.algorithm == "bfsle",
            path_size_logit=bool(demand),
            cores=self.cores,
            where=str(self.where) if self.where is not None else None,
            sl_link_loading=self.sl_link_loading,
            **self.parameters,
        )

    def execute(self, perform_assignment: bool = True) -> None:
        """
        Generate route choice sets between the previously supplied nodes, potentially performing
        an assignment.

        To access results see ``RouteChoice.get_results()``.

        :Arguments:
            **perform_assignment** (:obj:`bool`): Whether or not to perform an assignment.
            Defaults to ``False``.
        """
        if self.demand.df.index.empty:
            logging.warning("There is no demand or pairs of OD pairs to compute Route choice for.")
            return

        self.procedure_date = str(datetime.today())

        self.__rc.batched(
            self.demand,
            self._selected_links,
            bfsle=self.algorithm == "bfsle",
            path_size_logit=perform_assignment,
            cores=self.cores,
            where=str(self.where) if self.where is not None else None,
            sl_link_loading=self.sl_link_loading,
            **self.parameters,
        )

    def execute_from_path_files(self, path_files: Union[pathlib.Path, str], recompute_psl: bool = False) -> None:
        """
        Perform an assignment from an existing set of path-files.

        This method expects the path-files to be written by the ``self.save_path_files()`` method,
        however any parquet hive dataset with the correct structure is accepted. This allows the
        use of AequilibraE's path-sized logit, link loading, select link analysis, and assignment
        while using externally generated routes.
        """

        # Read the dataset schema and make sure it conforms to what we want
        df = RouteChoiceSetResults.read_dataset(path_files)
        required_fields = ["origin id", "destination id", "route set"] + [] if recompute_psl else ["probability"]
        schema = {
            "destination id": "uint32",
            "route set": "object",
            "cost": "float64",
            "mask": "bool",
            "path overlap": "float64",
            "probability": "float64",
            "origin id": "uint32",
        }
        dtypes = df.dtypes

        try:
            for field in required_fields:
                if schema[field] != dtypes.loc[field]:
                    raise TypeError(
                        f"schema of required field '{field}' does not match. "
                        f"Expected {schema[field]}, "
                        f"found {dtypes.loc[field]}"
                    )
        except KeyError as e:
            raise KeyError(f"Column '{field}' does not exist in the path-files") from e

        self.execute_from_pandas(df=df, recompute_psl=recompute_psl)

    def execute_from_pandas(self, df: pd.DataFrame, recompute_psl: bool = False) -> None:
        """
        Perform an assignment using route sets from a Pandas DataFrame.

        Requires the DataFrame contains the ``origin id``, ``destination id`` and ``route set``
        columns. The route sets must be a list of links IDs stored as integers with the direction
        encoded as the sign.  Additionally, when ``recompute_psl`` is ``False``, the
        ``probability`` column must also be present.

        When ``recompute_psl`` is ``True``, the path-sized logit is recomputed for each route with
        respect to the graphs current cost field and the ``beta`` and ``cutoff_prob`` parameters.

        All origin and destination IDs within the DataFrame must exist within the demand matrix.

        All link IDs and directions must exist within the graph. Links must also be present within
        the compressed graph.

        If ``recompute_psl`` is ``False`` the table returned from ``self.get_results()`` will have
        all zeros for the cost and path overlap fields, and all True for the mask field. If
        ``recompute_psl`` is ``True`` these fields will be recalculated as required.
        """

        required_columns = ["origin id", "destination id", "route set"] + [] if recompute_psl else ["probability"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"provided DataFrame is missing required column '{col}'")

        self.__rc.assign_from_df(
            self.graph.graph,
            df,
            self.demand,
            select_links=self._selected_links,
            recompute_psl=recompute_psl,
            sl_link_loading=self.sl_link_loading,
            store_results=self.parameters["store_results"],
            beta=self.parameters["beta"],
            cutoff_prob=self.parameters["cutoff_prob"],
        )

    def info(self) -> dict:
        """Returns information for the transit assignment procedure

        Dictionary contains keys:

        * Algorithm,
        * Matrix totals
        * Computer name
        * Procedure ID
        * Parameters
        * Select links

        The classes key is also a dictionary with all the user classes per transit class and
        their respective matrix totals.

        :Returns:
            **info** (:obj:`dict`): Dictionary with summary information
        """

        matrix_totals = self.demand.df.sum().to_dict()

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

    def get_results(self) -> pd.DataFrame:
        """
        Returns the results of the route choice procedure

        Returns a table of OD pairs to lists of link IDs for each OD pair provided (as columns).  Represents paths from
        ``origin`` to ``destination``. When the link id in the route set is positive it represents the ab direction,
        while negative represents the ba direction.

        :Returns:
            **results** (:obj:`pd.DataFrame`): Table with the results of the route choice procedure
        """
        if self.where is None:
            results = self.__rc.get_results()
        else:
            results = RouteChoiceSetResults.read_dataset(self.where)

        return results

    def save_path_files(self, where: Optional[pathlib.Path] = None):
        """
        Save path-files to the directory specific.

        Files will be saved as a parquet hive dataset partitioned by the origin ID. Existing path-files will not be
        removed to allow incremental route choice set generation.

        :Arguments:
            **where** (:obj:`Optional[pathlib.Path]`): Directory to save the dataset to.
        """
        where = where if where is not None else self.where
        if where is None:
            raise ValueError("either the 'where' argument or 'self.where' property must not None")

        self.__rc.write_path_files(where)

    def get_load_results(self) -> pd.DataFrame:
        """
        Translates the link loading results from the graph format into the network format.

        :Returns:
            **dataset** (:obj:`Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]`): A tuple
            of link loading results as DataFrames. Columns are the matrix name concatenated
            direction.
        """

        if self.demand.no_demand():
            warnings.warn("No demand was provided. To perform link loading add a demand matrix or data frame")
            return pd.DataFrame([])

        ll = self.__rc.get_link_loading(cores=self.cores)
        ll = {(k,): v for k, v in ll.items()}

        # Create a data store with a row for each uncompressed link
        m = _get_graph_to_network_mapping(self.graph.graph.link_id.values, self.graph.graph.direction.values)
        lids = np.unique(self.graph.graph.link_id.values)
        df = self.__link_loads_to_df(m, lids, ll)

        return df

    def __link_loads_to_df(self, mapping, lids, link_loads):
        df = pd.DataFrame(
            {"link_id": lids} | {(*k, dir): np.zeros(lids.shape) for k in link_loads.keys() for dir in ["ab", "ba"]}
        )
        added_dfs = []
        for k, v in link_loads.items():
            # Directional Flows
            df.iloc[mapping.network_ab_idx, df.columns.get_loc((*k, "ab"))] = np.nan_to_num(v[mapping.graph_ab_idx])
            df.iloc[mapping.network_ba_idx, df.columns.get_loc((*k, "ba"))] = np.nan_to_num(v[mapping.graph_ba_idx])

            # Tot Flow
            added_dfs.append(pd.DataFrame({(*k, "tot"): df[(*k, "ab")] + df[(*k, "ba")]}))

        df = pd.concat([df] + added_dfs, axis=1).set_index("link_id")
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df.sort_index()

    def set_select_links(
        self, links: Dict[Hashable, List[Union[Tuple[int, int], List[Tuple[int, int]]]]], link_loading=True
    ):
        """
        Set the selected links. Checks if the links and directions are valid. Supports **OR** and
        **AND** sets of links.

        Dictionary values should be a list of either a single ``(link_id, direction)`` tuple or a
        list of ``(link_id, dirirection)``.

        The elements of the first list represent the **AND** sets, together they are OR'ed. If any
        of these sets is satisfied the link are loaded as appropriate.

        The **AND** sets are comprised of either a single ``(link_id, direction)`` tuple or a list
        of ``(link_id, direction)``. The single tuple represents an **AND** set with a single
        element.

        All links and directions in an **AND** set must appear in any order within a route for it
        to be considered satisfied.

        Supply ``links=None`` to disable select link analysis.

        :Arguments:
            **links** (:obj:`Union[None, Dict[Hashable, List[Union[Tuple[int, int], List[Tuple[int, int]]]]]]`):
            Name of link set and link IDs and directions to be used in select link analysis.

            **link_loading** (:obj:`bool`): Enable select link loading. If disabled only OD matrix
            results are available.

        """
        self._selected_links = {}
        self.sl_link_loading = link_loading

        if links is None:
            del self._config["select_links"]
            return

        max_id = self.graph.compact_graph.id.max() + 1

        for name, link_set in links.items():
            normalised_link_set = []
            for link_ids in link_set:
                if isinstance(link_ids, tuple) and len(link_ids) == 2 and link_ids[1] == 0:
                    warnings.warn(
                        f"Adding both directions of a link ({link_ids[0]}) to a single AND set is likely "
                        f"unintentional. Replacing with {(link_ids[0], -1)} OR {(link_ids[0], 1)}"
                    )
                    normalised_link_set.append((link_ids[0], -1))
                    normalised_link_set.append((link_ids[0], 1))
                else:
                    normalised_link_set.append(link_ids)

            or_set = set()
            for link_ids in normalised_link_set:
                and_set = set()
                for link, dir in link_ids:
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

    def get_select_link_loading_results(self) -> pd.DataFrame:
        """
        Get the select link loading results.

        :Returns:
            **dataset** (:obj:`Tuple[pd.DataFrame, pd.DataFrame]`): Select link loading results
            as DataFrames. Columns are the matrix name concatenated with the select link set
            and direction.
        """

        if self.demand.no_demand():
            raise ValueError("no demand was provided. To perform link loading add a demand matrix or data frame")
        elif not self.sl_link_loading:
            raise ValueError("select link loading was disabled via `set_select_links(..., link_loading=False)`")

        sl_link_loads = {}
        for sl_name, sl_res in self.__rc.get_sl_link_loading().items():
            for demand_name, res in sl_res.items():
                sl_link_loads[demand_name, sl_name] = res

        # Create a data store with a row for each uncompressed link
        m = _get_graph_to_network_mapping(self.graph.graph.link_id.values, self.graph.graph.direction.values)
        lids = np.unique(self.graph.graph.link_id.values)
        df = self.__link_loads_to_df(m, lids, sl_link_loads)

        return df

    def get_select_link_od_matrix_results(self) -> Dict[str, Dict[str, scipy.sparse.coo_matrix]]:
        """
        Get the select link OD matrix results as a sparse matrix.

        :Returns:
            **select link OD matrix results** (:obj:`Dict[str, Dict[str, scipy.sparse.coo_matrix]]`):
            Returns a dict of select link set names to a dict of demand column names to a sparse OD
            matrix
        """

        if self.demand.no_demand():
            raise ValueError("No demand was provided. To perform link loading add a demand matrix or data frame")

        return self.__rc.get_sl_od_matrices()

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
        res_path = path.join(project.project_base_path, "results_database.sqlite")
        with commit_and_close(res_path, missing_ok=True) as conn:
            df.to_sql(table_name, conn, index=True)

        with self.project.db_connection as conn:
            conn.execute(
                """Insert into results(table_name, procedure, procedure_id, procedure_report, timestamp,
                                                description) Values(?,?,?,?,?,?)""",
                data,
            )

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
        Saves the select link link flows for all classes into the results database. Additionally,
        it exports the OD matrices into OMX format.

        :Arguments:
            **table_name** (:obj:`str`): Name of the table being inserted to and the name of the
            OpenMatrix file used for OD matrices.

            **project** (:obj:`Project`, `Optional`): Project we want to save the results to.
            Defaults to the active project
        """
        if not project:
            project = self.project or get_active_project()

        u = self.get_select_link_loading_results()
        info = self.info()
        self.__save_dataframe(
            u,
            "Select link analysis",
            "Uncompressed select link analysis results",
            table_name + "_uncompressed",
            info,
            project=project,
        )

        path = (pathlib.Path(project.project_base_path) / "matrices" / table_name).with_suffix(".omx")
        for sl_name, v in self.get_select_link_od_matrix_results().items():
            for demand_name, mat in v.items():
                mat.to_disk(path, sl_name + "_" + demand_name)

        file = omx.open_file(path, "a")
        try:
            file.create_mapping(self.index_name, self.graph.centroids)
        finally:
            file.close()
