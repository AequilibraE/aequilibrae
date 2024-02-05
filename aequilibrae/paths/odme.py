"""
ODME Infrastructure (User Interaction Class):
"""

# TODO - 3 todo's remaining in code, see below

from typing import Tuple
from uuid import uuid4
from datetime import datetime
from os.path import join
from pathlib import Path
import json
import importlib.util as iutil
import numpy as np
import pandas as pd

from aequilibrae.paths import TrafficAssignment, TrafficClass
from aequilibrae.paths.odme_ import ScalingFactors, ODMEResults

from aequilibrae.context import get_active_project
from aequilibrae.matrix import AequilibraeMatrix

# Checks if we can display OMX
spec = iutil.find_spec("openmatrix")
has_omx = spec is not None
if has_omx:
    import openmatrix as omx


class ODME(object):
    """Origin-Destination Matrix Estimation class.

    For a comprehensive example on use, see the Use examples page.

    .. code-block:: python

        >>> from aequilibrae import TrafficAssignment, TrafficClass, Graph, Project, ODME
        >>> project = Project.from_path("/tmp/test_project")
        >>> project.network.build_graphs()

        >>> graph = project.network.graphs['c'] # we grab the graph for cars
        >>> graph.set_blocked_centroid_flows(False)

        >>> matrix = project.matrices.get_matrix("demand_omx")
        >>> matrix.computational_view(['car']) # The demand matrix is what we want to estimate

        # See the TrafficAssignment class on details for how to create an assignment
        >>> assignment = TrafficAssignment()
        # Make sure we have the matrix we want to perturb included in the TrafficClass
        >>> assignclass = TrafficClass("car", graph, matrix)
        >>> assignment.set_classes([assignclass])
        >>> assignment.set_vdf("BPR")
        >>> assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
        >>> assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})
        >>> assignment.set_capacity_field("capacity")
        >>> assignment.set_time_field("free_flow_time")
        >>> assignment.max_iter = 5
        >>> assignment.set_algorithm("msa")

        # We now need to create our data (count volumes), suppose we have these in a csv file:
        >>> import pandas as pd
        >>> counts = pd.read_csv("/tmp/test_data.csv")

        # We can now run the ODME procedure, see Use examples page for a more
        # comprehensive overview of the options available when initialising.
        >>> odme = ODME(assignment, counts)
        >>> odme.execute() # See Use examples for optional arguments

        # There are multiple ways to deal with the output to ODME,
        # the simplest is to save the procedure as output demand matrices
        # and procedure statistics to the project database.
        >>> odme.save_to_project("odme_test", "odme_test.omx", project=project)

        # We can also get statistics in memory immediately as pandas dataframes
        >>> results = odme.results

        # Dataframe of the cumulative factors applied to the input demands
        >>> cumulative_factors = results.get_cumulative_factors()

        # Statistics on the procedure across each iteration
        >>> iterative_stats = results.get_iteration_statistics()

        # Statistics on the procedure tracking each link (count volumes)
        >>> link_stats = results.get_link_statistics()
    """

    # Input count volume columns (assigned volumes will be added during execution)
    COUNT_VOLUME_COLS = ["class", "link_id", "direction", "obs_volume"]
    GMEAN_LIMIT = 0.01  # FACTOR LIMITING VARIABLE - FOR TESTING PURPOSES - DEFUNCT!
    ALL_ALGORITHMS = ["gmean", "spiess", "reg_spiess"]
    DEFAULT_STOP_CRIT = {"max_outer": 50, "max_inner": 50, "convergence_crit": 10**-4, "inner_convergence": 10**-4}

    def __init__(
        self,
        assignment: TrafficAssignment,
        count_volumes: pd.DataFrame,
        stop_crit=None,
        algorithm: str = "spiess",
        alpha: float = None,
    ) -> None:
        """
        Parameters:
            assignment: the TrafficAssignment object - should be initialised with volume
                    delay functions and their parameters and an assignment algorithm,
                    as well as TrafficClass's containing initial demand matrices (which
                    will be duplicated so input data is not corrupted). Doesn't
                    need to have preset select links (these will be overwritten).
            count_volumes: a dataframe detailing the links, the class they are associated with,
                    the direction and their observed volume.
            stop_crit: the maximum number of iterations and the convergence criterion
                    (see ODME.DEFAULT_STOP_CRIT for formatting).
            algorithm: specification for which gradient-descent based algorithm to use
                    (see ODME.ALL_ALGORITHMS for options).
            alpha: used as a hyper-parameter for regularised spiess (see technical document for
                    details).

        NOTE - certain functionality is only implemented for single class ODME - see docstrings for
               such cases.
        """
        self.__check_inputs(count_volumes, stop_crit, alpha, algorithm)

        self.assignment = assignment
        self.classes = assignment.classes
        self.__duplicate_matrices()

        self.class_names = [user_class._id for user_class in self.classes]
        self.names_to_indices = {name: index for index, name in enumerate(self.class_names)}

        # The following are implicitly ordered by the list of Traffic Classes:
        self.aequilibrae_matrices = [user_class.matrix for user_class in self.classes]
        self.demands = [user_class.matrix.matrix_view for user_class in self.classes]
        # Reshaping matrices because when computational_view is done with a single class we get
        # n x n instead of n x n x 1 - NOTE: this may be redundant now
        for i, demand in enumerate(self.demands):
            if len(demand.shape) == 2:
                self.demands[i] = demand[:, :, np.newaxis]

        self.original_demands = [np.copy(matrix) for matrix in self.demands]

        # Observed Links & Associated Volumes
        self.count_volumes = count_volumes.copy(deep=True)

        # Select Link:
        self._sl_matrices = dict()
        self.__set_select_links()

        # Algorithm Specifications:
        self._norms = self.__get_norms(algorithm)
        self._algorithm = algorithm

        # Objective Function:
        self._obj_func = None
        self.__init_objective_func()
        self.last_convergence = None
        # Component of objective function from flows/regularisation:
        self.flow_obj, self.reg_obj = None, None
        # Initially inf to ensure inner iterations begin
        self.convergence_change = float("inf")

        # Stopping criterion
        if not stop_crit:
            stop_crit = self.DEFAULT_STOP_CRIT
        self.max_outer = stop_crit["max_outer"]
        self.max_inner = stop_crit["max_inner"]
        self.outer_convergence_crit = stop_crit["convergence_crit"]
        self.inner_convergence_crit = stop_crit["inner_convergence"]

        # Hyper-parameters for regularisation:
        if algorithm in ["reg_spiess"]:
            if alpha is None or alpha > 1 or alpha < 0:  # THIS CHECK SHOULD PROBABLY BE MORE ROBUST
                raise ValueError("Hyper-parameter alpha should be between 0 and 1")
            self.alpha = alpha
            self.beta = 1 - alpha

        # Results/Statistics:
        self.results = ODMEResults(self)

        # Procedure Information:
        self.procedure_date = ""
        self.procedure_id = ""

    # Utilities:
    def __check_inputs(self, counts: pd.DataFrame, stop_crit: dict, alpha: float, algorithm: str) -> None:
        """
        Ensures all user input is of correct format/value.
        NOTE - we do not check if the assignment is given properly,
        this is assumed to either be done correctly or for any errors
        to be picked up by the TrafficAssignment class.
        """
        # Check algorithm
        if not isinstance(algorithm, str):
            raise ValueError("Algorithm must be input as a string")
        elif algorithm not in self.ALL_ALGORITHMS:
            raise ValueError(
                f"'{algorithm}' is not a valid algorithm.\n"
                + "Currently implemented algorithms include:\n"
                + "\n".join(self.ALL_ALGORITHMS)
            )

        # Check stopping criteria if given
        stop_error = False
        if stop_crit is not None:
            keys = self.DEFAULT_STOP_CRIT.keys()
            if not isinstance(stop_crit, dict):
                stop_error = True
            else:
                for key in keys:
                    if key not in stop_crit:
                        stop_error = True
                    elif key in ["max_outer", "max_inner"]:
                        if not isinstance(stop_crit[key], int):
                            stop_error = True
                        elif stop_crit[key] < 1:
                            stop_error = True
                    else:
                        if not isinstance(stop_crit[key], (float, int)):
                            stop_error = True
                        elif stop_crit[key] < 0:
                            stop_error = True

        if stop_error:
            raise ValueError(
                "Stopping criterion must be given as a dictionary as follows,"
                + "(key -> type of value):"
                + "max_outer -> positive integer"
                + "max_inner -> positive integer"
                + "convergence_crit -> non-negative integer/float"
                + "inner_convergence -> non-negative integer/float"
            )

        # Check count volumes
        counts_error = False
        if not isinstance(counts, pd.DataFrame):
            counts_error = True
        elif len(counts) < 1:
            counts_error = True
        elif len(counts.columns) != len(self.COUNT_VOLUME_COLS):
            counts_error = True

        if not counts_error:
            for col in counts.columns:
                if col not in self.COUNT_VOLUME_COLS:
                    counts_error = True

        if not counts_error:
            observed = counts["obs_volume"]
            if not (pd.api.types.is_float_dtype(observed) or pd.api.types.is_integer_dtype(observed)):
                counts_error = True
            elif not np.all(observed >= 0):
                counts_error = True

        if not counts_error:
            if counts.duplicated(subset=["class", "link_id", "direction"]).any():
                counts_error = True

        if counts_error:
            raise ValueError(
                "Count volumes must be a non-empty pandas dataframe with columns:\n"
                + "\n".join(self.COUNT_VOLUME_COLS)
                + "\n and all observed volumes must be non-negative floats or integers, and"
                + "only a single count volume should be given for a"
                + "particular class, link_id and direction"
            )

        # Check alpha value if given
        if alpha is not None:
            if not isinstance(alpha, (float, int)):
                raise ValueError("Input alpha should be a float or integer (0 to 1)")
            elif alpha > 1 or alpha < 0:
                raise ValueError("Input alpha should be between 0 and 1")

    def __duplicate_matrices(self):
        """
        Duplicates the given matrices in memory only and replaces the TrafficClass objects.
        """
        # Loop through TrafficClasses - create new and replace, then set classes
        new_classes = []
        for usr_cls in self.classes:
            mat = usr_cls.matrix.copy(cores=usr_cls.matrix.view_names, memory_only=True)
            mat.computational_view()

            new_cls = TrafficClass(usr_cls._id, usr_cls.graph, mat)
            new_cls.set_pce(usr_cls.pce)
            if usr_cls.fixed_cost_field:
                new_cls.set_fixed_cost(usr_cls.fixed_cost_field, usr_cls.fc_multiplier)
            new_cls.set_vot(usr_cls.vot)
            new_classes.append(new_cls)

        self.assignment.set_classes(new_classes)
        self.classes = self.assignment.classes

    def estimate_alpha(self, alpha: float) -> float:
        """
        Estimates a starting hyper-paramater for regularised
        spiess given a number between 0-1.

        NOTE - currently only implemented for single class
        """
        demand_sum = np.sum(self.demands[0])
        flow_sum = np.sum(self.count_volumes["obs_volume"])
        return (alpha * demand_sum) / ((alpha * flow_sum) + ((1 - alpha) * demand_sum))

    def __get_norms(self, algo: str) -> Tuple[int, int]:
        """
        Sets the specifications for the objective function for the algorithm chosen.
        """
        if algo in ["gmean", "spiess"]:
            return (2, 0)
        elif algo in ["reg_spiess"]:
            return (2, 2)

    def __set_select_links(self) -> None:
        """
        Sets all select links for each class and for each associated set of count volumes.
        """
        c_v = self.count_volumes
        for user_class in self.classes:
            user_class.set_select_links(
                {
                    self.get_sl_key(row): [(row["link_id"], row["direction"])]
                    for _, row in c_v[c_v["class"] == user_class._id].iterrows()
                }
            )

    @staticmethod
    def get_sl_key(row: pd.Series) -> str:
        """
        Given a particular row from the observervations (count_volumes) returns
        a key corresponding to it for use in all select link extraction.

        NOTE - this is intended for internal use only - it has no relevance outside
        of this class and classes within the odme_ submodule.
        """
        return f"sl_{row['class']}_{row['link_id']}_{row['direction']}"

    def __set_convergence_values(self, new_convergence: float) -> None:
        """
        Given a new convergence value calculates the difference between the previous convergence
        and new convergence, and sets appropriate values.
        """
        if self.last_convergence:
            self.convergence_change = abs(self.last_convergence - new_convergence)
        self.last_convergence = new_convergence

    # TODO - I have previously forgotten to include the pce into the objective
    # function for multi-class purposes, this needs to be done by multiplying the
    # component of flow for each class contributing to the objective function by
    # the pce of that class.
    def __init_objective_func(self) -> None:
        """
        Initialises the objective function - depends on algorithm chosen by user.

        Current objective functions have 2 parts which are summed:
            1. The p-norm raised to the power p of the error vector for observed flows.
            2. The p-norm raised to the power p of the error matrix (treated as a n^2 vector)
               for the demand matrix.

        NOTE - currently (1.) must always be present, but (2.) (the regularisation term)
               need not be present.
        """
        p_1 = self._norms[0]
        p_2 = self._norms[1]

        def __reg_obj_func(self) -> None:
            """
            Objective function containing regularisation term.

            # NOTE - pce not yet included for multi-class
            """
            obs_vals = self.count_volumes["obs_volume"].to_numpy()
            assign_vals = self.count_volumes["assign_volume"].to_numpy()
            self.flow_obj = self.alpha * np.sum(np.abs(obs_vals - assign_vals) ** p_1) / p_1
            self.reg_obj = self.beta * np.sum(np.abs(self.original_demands[0] - self.demands[0]) ** p_2) / p_2
            self.__set_convergence_values(self.flow_obj + self.reg_obj)

        def __obj_func(self) -> None:
            """
            Objective function with no regularisation term.

            # NOTE - pce not yet included for multi-class
            """
            obs_vals = self.count_volumes["obs_volume"].to_numpy()
            assign_vals = self.count_volumes["assign_volume"].to_numpy()
            self.flow_obj = np.sum(np.abs(obs_vals - assign_vals) ** p_1) / p_1
            self.__set_convergence_values(self.flow_obj)

        if p_2:
            self._obj_func = __reg_obj_func
        else:
            self._obj_func = __obj_func

    # Output/Results:

    # TODO - the procedure report is not yet properly implemented since the
    # procedure_report column in the matrix sql file doesn't yet exist.
    # Need to find out how to fix this and then check the data is stored
    # correctly and can be retrieved to obtain the same results as directly
    # extracting the results from the corresponding ODMEResults object.
    def save_to_project(self, name: str, file_name: str, project=None) -> None:
        """Saves the final demand matrix output to the project file

        :Arguments:
            **name** (:obj:`str`): Name of the desired matrix record
            **file_name** (:obj:`str`): Name for the matrix file name. AEM and OMX supported
            **project** (:obj:`Project`, Optional): Project we want to save the results to.
            Defaults to the active project
        """
        project = project or get_active_project()
        mats = project.matrices
        file_path = join(mats.fldr, file_name)

        if Path(file_name).suffix.lower() == ".omx":
            self.__save_as_omx(file_path)
        elif ".aem" in file_name:
            self.__save_as_aem(file_path)
        else:  # unsupported file-type
            raise ValueError("Only supporting .omx and .aem")

        record = mats.new_record(name, file_name)
        record.procedure_id = self.procedure_id
        record.timestamp = self.procedure_date
        record.procedure = "Origin-Destination Matrix Estimation"
        record.report = json.dumps(
            {
                "iterations": self.results.get_iteration_statistics().to_dict(),
                "by_link": self.results.get_link_statistics().to_dict(),
            }
        )
        record.save()

    def __save_as_omx(self, file_path: str) -> None:
        """Saves the final demand matrix output as a .omx file to the project file

        :Arguments:
            **file_path** (:obj:`str`): File path for the matrix file name (must end with .omx)
        """
        omx_mat = omx.open_file(file_path, "w")
        for cls_name, matrix in zip(self.class_names, self.aequilibrae_matrices):
            for core in matrix.view_names:
                omx_mat[f"{cls_name}_{core}"] = matrix.matrix[core]

        index = self.aequilibrae_matrices[0].current_index
        omx_mat.create_mapping(index, self.aequilibrae_matrices[0].index)

        for cls_name, matrix in zip(self.class_names, self.aequilibrae_matrices):
            for core in matrix.view_names:
                description = f"ODME Procedure {self.procedure_id}"
                omx_mat[f"{cls_name}_{core}"].attrs.description = description
        omx_mat.close()

    def __save_as_aem(self, file_path: str) -> None:
        """
        Saves the final demand matrix output as a .aem file to the project file

        :Arguments:
            **file_path** (:obj:`str`): File path for the matrix file name (must end with .aem)
        """
        mat = AequilibraeMatrix()
        matrix_names = []
        for cls_name, matrix in zip(self.class_names, self.aequilibrae_matrices):
            for core in matrix.view_names:
                matrix_names.append(f"{cls_name}_{core}")

        args = {
            "zones": self.aequilibrae_matrices[0].zones,
            "matrix_names": matrix_names,
            "index_names": self.aequilibrae_matrices[0].index_names,
            "memory_only": False,
            "file_name": file_path,
        }
        mat.create_empty(**args)
        mat.indices[:, :] = self.aequilibrae_matrices[0].indices[:, :]
        for cls_name, matrix in zip(self.class_names, self.aequilibrae_matrices):
            for core in matrix.view_names:
                mat.matrix[f"{cls_name}_{core}"][:, :] = matrix.matrix[core][:, :]
        mat.description = f"ODME Procedure {self.procedure_id}"
        mat.save()
        mat.close()
        del mat

    def get_demands(self) -> list[np.ndarray]:
        """
        Returns all demand matrices (can be called before or after execution).
        """
        return self.demands

    # ODME Execution:
    def execute(self, verbose=False, print_rate=1) -> None:
        """
        Run ODME algorithm until either the maximum iterations has been reached,
        or the convergence criterion has been met.

        Parameters:
            verbose: if true will print output to screen during runtime so user
                can track ODME progress.
            print_rate: the rate at which to print output to user (if verbose is true).
                Does nothing if verbose is false.
        """
        # Initialise Procedure:
        self.results.init_timer()
        self.procedure_date = str(datetime.today())
        self.procedure_id = uuid4().hex

        # Create values for SL matrices & assigned flows
        self.__perform_assignment()

        # Outer iterations
        outer = 0
        while outer < self.max_outer and self.last_convergence > self.outer_convergence_crit:
            # Log stats before running algorithm:
            outer += 1
            self.results.log_iter(ODMEResults.OUTER)

            # Inner iterations:
            # Ensure at least 1 inner iteration is run per outer loop
            self.convergence_change = float("inf")
            inner = 0
            while inner < self.max_inner and self.convergence_change > self.inner_convergence_crit:
                inner += 1
                self.__execute_inner_iter()
                self.results.log_iter(ODMEResults.INNER)

            if verbose and outer % print_rate == 0:
                print(f"Outer iteration {outer} is complete.")

            # Reassign values at the end of each outer loop
            self.__perform_assignment()

        # Add final stats following final assignment:
        self.results.log_iter(ODMEResults.FINAL_LOG)

    # TODO - check whether the demand matrix replacement at the start of the following
    # function is sufficient - we may want to replace the matrix.matrices value
    # and call matrix.computational_view() (with appropriate arguments) instead.
    def __perform_assignment(self) -> None:
        """
        Uses current demand matrix to perform an assignment, then save
        the assigned flows and select link matrices. Also recalculates the
        objective function following an assignment.

        This function will only be called at the start of an outer
        iteration & during the final convergence test.
        """
        # Change the demand matrices within the TrafficClass's to the current
        # demand matrices that have been calculated from the previous outer iteration.
        for aeq_matrix, demand in zip(self.aequilibrae_matrices, self.demands):
            aeq_matrix.matrix_view = demand

        # Perform the assignment
        self.assignment.execute()

        # Store reference to select link demand matrices as proportion matrices
        for assignclass, demand in zip(self.classes, self.demands):
            sl_matrices = assignclass.results.select_link_od.matrix
            for link in sl_matrices:
                self._sl_matrices[link] = np.nan_to_num(sl_matrices[link] / demand)

        # Extract and store array of assigned volumes to the select links
        self.__extract_volumes()

        # Recalculate convergence values
        self._obj_func(self)

    def __extract_volumes(self) -> None:
        """
        Extracts and stores assigned volumes (corresponding for those for which we have
        observations - ie count volumes).

        NOTE - this does not take into account pce, ie this is the number of vehicles, not
        'flow'.
        """
        assign_df = self.assignment.results().reset_index(drop=False).fillna(0)
        # Dictionary to select correct column of results dataframe
        col = dict()
        for cls_name, matrix in zip(self.class_names, self.aequilibrae_matrices):
            name = matrix.view_names[0]
            col[cls_name] = {1: f"{name}_ab", -1: f"{name}_ba", 0: f"{name}_tot"}

        # For extracting a single assigned flow:
        def extract_volume(row) -> None:
            """
            Extracts volume corresponding to particular link (from row) and return it.
            For inner iterations need to calculate this via __calculate_volumes
            """
            return assign_df.loc[assign_df["link_id"] == row["link_id"], col[row["class"]][row["direction"]]].values[0]

        # Extract a flow for each count volume:
        self.count_volumes["assign_volume"] = self.count_volumes.apply(extract_volume, axis=1)

    def __execute_inner_iter(self) -> None:
        """
        Runs an inner iteration of the ODME algorithm.
        This assumes the SL matrices stay constant and modifies the current demand matrices.
        """
        # Element-wise multiplication of demand matrices by scaling factors
        factors = self.__get_scaling_factors()
        self.demands = [demand * factor for demand, factor in zip(self.demands, factors)]

        # Recalculate the link flows
        self.__calculate_volumes()

        # Recalculate convergence level:
        self._obj_func(self)

    def __get_scaling_factors(self) -> list[np.ndarray]:
        """
        Returns scaling matrices for each user class - depending on algorithm chosen.

        NOTE - we expect any algorithm to return a list of factor matrices in order of the
        stored user classes.
        """
        algorithm = ScalingFactors(self, self._algorithm)
        factors = algorithm.generate()
        self.results.record_factor_stats(factors)
        return factors

    def __calculate_volumes(self) -> None:
        """
        Calculates and stores link volumes using current sl_matrices & demand matrices.
        """

        # Calculate a single flow:
        def __calculate_volume(self, row: pd.Series) -> float:
            """
            Given a single row of the count volumes dataframe,
            calculates the appropriate corresponding assigned
            volume.
            """
            sl_matrix = self._sl_matrices[self.get_sl_key(row)]
            demand_matrix = self.demands[self.names_to_indices[row["class"]]]
            return np.sum(sl_matrix * demand_matrix)

        # Calculate flows for all rows:
        self.count_volumes["assign_volume"] = self.count_volumes.apply(
            lambda row: __calculate_volume(self, row), axis=1
        )
