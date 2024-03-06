import numpy as np
import socket
from aequilibrae.context import get_active_project
from aequilibrae.paths.graph import Graph
from aequilibrae.paths.route_choice_set import RouteChoiceSet
from typing import Optional
import pyarrow as pa
import pathlib

import logging


class RouteChoice:
    all_algorithms = ["bfsle", "lp", "link-penalisation"]
    default_paramaters = {
        "beta": 1.0,
        "theta": 1.0,
        "penalty": 1.1,
        "seed": 0,
        "max_routes": 0,
        "max_depth": 0,
    }

    def __init__(self, graph: Graph, project=None):
        self.paramaters = self.default_paramaters.copy()

        proj = project or get_active_project(must_exist=False)
        self.project = proj

        self.logger = proj.logger if proj else logging.getLogger("aequilibrae")

        self.cores: int = 0
        self.graph = graph
        self.__rc = RouteChoiceSet(graph)

        self.schema = RouteChoiceSet.schema
        self.psl_schema = RouteChoiceSet.psl_schema

        self.compact_link_loads: Optional[np.array] = None
        self.link_loads: Optional[np.array] = None
        self.results: Optional[pa.Table] = None
        self.where: Optional[pathlib.Path] = None

    def set_algorithm(self, algorithm: str):
        """
        Chooses the assignment algorithm.
        Options are, 'bfsle' for breadth first search with link removal, or 'link-penalisation'

        'lp' is also accepted as an alternative to 'link-penalisation'

        :Arguments:
            **algorithm** (:obj:`str`): Algorithm to be used
        """
        algo_dict = {i: i for i in self.all_algorithms}
        algo_dict["lp"] = "link-penalisation"
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
        if not self.classes:
            raise RuntimeError("You need load transit classes before overwriting the number of cores")

        self.cores = cores

    def set_paramaters(self, par: dict):
        """
        Sets the parameters for the route choice  TODO, do we want link specific values?

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


        Parameter values can be scalars (same values for the entire network) or network field names
        (link-specific values) - Examples: {'alpha': 0.15, 'beta': 4.0} or  {'alpha': 'alpha', 'beta': 'beta'}


        :Arguments:
            **par** (:obj:`dict`): Dictionary with all parameters for the chosen VDF
        """

        if any(key not in self.default_paramaters for key in par.keys()):
            raise ValueError("Invalid parameter provided")

        self.paramaters = self.default_paramaters | par

    def set_save_path_files(self, save_it: bool) -> None:
        """Turn path saving on or off.

        :Arguments:
            **save_it** (:obj:`bool`): Boolean to indicate whether paths should be saved
        """
        self.save_path_files = save_it

    def set_save_routes(self, where: Optional[str] = None) -> None:
        """
        Set save path for route choice resutls. Provide ``None`` to disable.

        **warning** enabling route saving will disable in memory results. Viewing the results will read the results
        from disk first.

        :Arguments:
            **save_it** (:obj:`bool`): Boolean to indicate whether routes should be saved
        """
        self.where = pathlib.Path(where) if where is not None else None

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

    def results(self):
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
