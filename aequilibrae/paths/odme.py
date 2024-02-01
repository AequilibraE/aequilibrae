"""
Implementation of ODME Infrastructure:
"""

# NOTE - Until issue with select link flows not matching assigned flows ODME should not be used
# with biconjugate/conjugate frank-wolfe - refer to Issue #493

# NOTE - To Do:
#       All docstrings need to be updated appropriately
#       Any extra clean up needs to be done
#       Check the matrix replacement in __perform_assignment

from typing import Tuple
from uuid import uuid4
from datetime import datetime
from os.path import join
from pathlib import Path
import importlib.util as iutil
import numpy as np
import pandas as pd

from aequilibrae.paths import TrafficAssignment, TrafficClass
from aequilibrae.paths.odme_submodule import ScalingFactors, ODMEResults

from aequilibrae.context import get_active_project
from aequilibrae.matrix import AequilibraeMatrix

# Checks if we can display OMX
spec = iutil.find_spec("openmatrix")
has_omx = spec is not None
if has_omx:
    import openmatrix as omx

class ODME(object):
    """ ODME Infrastructure """
    # Input count volume columns (assigned volumes will be added during execution)
    COUNT_VOLUME_COLS = ["class", "link_id", "direction", "obs_volume"]
    GMEAN_LIMIT = 0.01 # FACTOR LIMITING VARIABLE - FOR TESTING PURPOSES - DEFUNCT!
    ALL_ALGORITHMS = ["gmean", "spiess", "reg_spiess"]
    DEFAULT_STOP_CRIT = {"max_outer": 50, "max_inner": 50,
        "convergence_crit": 10**-4, "inner_convergence": 10**-4}

    def __init__(self,
        assignment: TrafficAssignment,
        count_volumes: pd.DataFrame,
        stop_crit=None,
        alpha: float = None,
        algorithm: str = "spiess",
        verbose: bool = False
    ) -> None:
        """
        For now see description in pdf file in SMP internship team folder

        Parameters:
            assignment: the TrafficAssignment object - should be initialised with volume delay functions
                    and their parameters and an assignment algorithm, as well as a TrafficClass containing
                    an initial demand matrix. Doesn't need to have preset select links.
            count_volumes: a dataframe detailing the links, the class they are associated with, the direction
                    and their observed volume. NOTE - CURRENTLY ASSUMING SINGLE CLASS
            stop_crit: the maximum number of iterations and the convergence criterion.
            alg_spec: NOT YET AVAILABLE - will be implemented later to allow user flexibility on what sort 
                    of algorithm they choose.

        CURRENTLY ONLY IMPLEMENTED FOR SINGLE CLASS (MULTI-CLASS UNDER DEVELOPMENT)
        CHANGE STOPPING CRITERION TO BE A DICTIONARY!
        """
        self.__check_inputs(count_volumes, stop_crit, alpha, algorithm)

        self.assignment = assignment
        self.classes = assignment.classes
        self.__duplicate_matrices()

        self.class_names = [user_class.__id__ for user_class in self.classes]
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
        self.convergence_change = float('inf')

        # Stopping criterion
        if not stop_crit:
            stop_crit = self.DEFAULT_STOP_CRIT
        self.max_outer = stop_crit["max_outer"]
        self.max_inner = stop_crit["max_inner"]
        self.outer_convergence_crit = stop_crit["convergence_crit"]
        self.inner_convergence_crit = stop_crit["inner_convergence"]

        # Hyper-parameters for regularisation:
        if algorithm in ["reg_spiess"]:
            if alpha is None or alpha > 1 or alpha < 0: # THIS CHECK SHOULD PROBABLY BE MORE ROBUST
                raise ValueError("Hyper-parameter alpha should be between 0 and 1")
            self.alpha = alpha
            self.beta = 1 - alpha

        # Results/Statistics:
        self.results = ODMEResults(self)

        # Printing During Runtime:
        self._verbose = verbose
        
        # Procedure Information:
        self.procedure_date = ""
        self.procedure_id = ""

    # Utilities:
    def __check_inputs(self,
        counts: pd.DataFrame,
        stop_crit: dict,
        alpha: float,
        algorithm: str) -> None:
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
            raise ValueError(f"'{algorithm}' is not a valid algorithm.\n" +
                "Currently implemented algorithms include:\n" +
                '\n'.join(self.ALL_ALGORITHMS))

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
            raise ValueError("Stopping criterion must be given as a dictionary as follows," +
                "(key -> type of value):" +
                "max_outer -> positive integer" +
                "max_inner -> positive integer" +
                "convergence_crit -> non-negative integer/float" +
                "inner_convergence -> non-negative integer/float")

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
            if not (pd.api.types.is_float_dtype(observed) or
                pd.api.types.is_integer_dtype(observed)):
                counts_error = True
            elif not np.all(observed >= 0):
                counts_error = True

        if not counts_error:
            if counts.duplicated(subset=["class", "link_id", "direction"]).any():
                counts_error = True

        if counts_error:
            raise ValueError("Count volumes must be a non-empty pandas dataframe with columns:\n" +
                '\n'.join(self.COUNT_VOLUME_COLS) +
                "\n and all observed volumes must be non-negative floats or integers, and" +
                "only a single count volume should be given for a" +
                "particular class, link_id and direction")

        # Check alpha value if given
        if alpha is not None:
            if not isinstance(alpha, (float, int)):
                raise ValueError("Input alpha should be a float or integer (0 to 1)")
            elif alpha > 1 or  alpha < 0:
                raise ValueError("Input alpha should be between 0 and 1")

    def __duplicate_matrices(self):
        """
        Duplicates the given matrices in memory only and replaces the TrafficClass objects.
        """
        # Loop through TrafficClasses - create new and replace, then set classes
        new_classes = []
        for usr_cls in self.classes:
            mat = usr_cls.matrix.copy(cores = usr_cls.matrix.view_names, memory_only=True)
            mat.computational_view()

            new_cls = TrafficClass(usr_cls.__id__, usr_cls.graph, mat)
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
        
        ONLY IMPLEMENTED FOR SINGLE CLASS!
        """
        demand_sum = np.sum(self.demands[0])
        flow_sum = np.sum(self.count_volumes["obs_volume"])
        return (alpha * demand_sum) / ((alpha * flow_sum) + ((1 - alpha) * demand_sum))

    def __get_norms(self, algo: str) -> Tuple[int, int]:
        """
        Sets the specifications for the objective function for the algorithm chosen.

        SHOULD REALLY MAKE ALL THIS OBJECTIVE FUNCTION STUFF MAKE MORE SENSE
        """
        if algo in ["gmean", "spiess"]:
            return (2, 0)
        elif algo in ["reg_spiess"]:
            return (2, 2)

    def __set_select_links(self) -> None:
        """
        Sets all select links for each class and for each observation.
        """
        c_v = self.count_volumes
        for user_class in self.classes:
            user_class.set_select_links(
                {
                    self.get_sl_key(row):
                    [(row['link_id'], row['direction'])]
                    for _, row in c_v[c_v['class'] == user_class.__id__
                    ].iterrows()
                }
            )

    @staticmethod
    def get_sl_key(row: pd.Series) -> str:
        """
        Given a particular row from the observervations (count_volumes) returns
        a key corresponding to it for use in all select link extraction.
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

    def __init_objective_func(self) -> None:
        """
        Initialises the objective function - parameters must be specified by user.

        Current objective functions have 2 parts which are summed:
            1. The p-norm raised to the power p of the error vector for observed flows.
            2. The p-norm raised to the power p of the error matrix (treated as a n^2 vector) for the demand matrix.
        
        (1.) must always be present, but (2.) (the regularisation term) need not be present (ie, specified as 0 by user).
        Default currently set to l1 (manhattan) norm for (1.) with no regularisation term (p2 = 0).

        CURRENTLY ONLY IMPLEMENTED FOR SINGLE CLASS!
        NOT YET COMPLETED FOR SINGLE CLASS - STILL UNDER DEVELOPMENT!
        HOW DO I GENERALISE THIS TO MULTI-CLASS
        NEED TO CHECK HOW PCE AFFECTS THIS!
        """
        p_1 = self._norms[0]
        p_2 = self._norms[1]

        def __reg_obj_func(self) -> None:
            """
            Objective function containing regularisation term.

            NOTE - NOT YET READY FOR USE! REGULARISATION TERM SHOULD BE ALPHA/BETA WEIGHTED!
            NEED TO DECIDE WHETHER I WANT TO SOMEHOW NORMALISE THE ALPHA/BETA WEIGHTS

            NOTE - IT'S POSSIBLE TO ONLY USE 1 HYPER-PARAMETER INTERNALLY BY USING 
            GAMMA = ALPHA/(1-ALPHA) - BUT THIS MIGHT HAVE FLOATING POINT ERRORS.

            ONLY IMPLEMENTED FOR SINGLE CLASS!
            NEEDS TO INCLUDE PCE FOR MULTI-CLASS!
            """
            obs_vals = self.count_volumes["obs_volume"].to_numpy()
            assign_vals = self.count_volumes['assign_volume'].to_numpy()
            self.flow_obj = self.alpha * np.sum(np.abs(obs_vals - assign_vals)**p_1) / p_1
            self.reg_obj = self.beta * np.sum(np.abs(self.init_demand_matrices[0] - self.demand_matrices[0])**p_2) / p_2
            self.__set_convergence_values(self.flow_obj + self.reg_obj)

        def __obj_func(self) -> None:
            """
            Objective function with no regularisation term.

            NEEDS TO INCLUDE PCE!
            """
            obs_vals = self.count_volumes["obs_volume"].to_numpy()
            assign_vals = self.count_volumes['assign_volume'].to_numpy()
            self.flow_obj = np.sum(np.abs(obs_vals - assign_vals)**p_1) / p_1
            self.__set_convergence_values(self.flow_obj)

        if p_2:
            self._obj_func = __reg_obj_func
        else:
            self._obj_func = __obj_func

    # Output/Results:
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
        else: # unsupported file-type
            raise ValueError("Only supporting .omx and .aem")

        record = mats.new_record(name, file_name)
        record.procedure_id = self.procedure_id
        record.timestamp = self.procedure_date
        record.procedure = "Origin-Destination Matrix Estimation"
        # Note that below just involves doing str() to the particular results file.
        # CHECK WHETHER THIS IS ACCURATE - THIS SEEMS DIFFERENT TO PROCEDURE REPORT
        # record.procedure_report = Create json and save to this file
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

    def get_iteration_factors(self) -> pd.DataFrame:
        """
        Returns a dataframe on statistics of factors for each iteration.
        """
        return self.results.factor_stats

    def get_cumulative_factors(self) -> pd.DataFrame:
        """
        Return the cumulative factors (ratio of final to initial matrix) in a dataframe.
        """
        return self.results.get_cumulative_factors()

    def get_all_statistics(self) -> pd.DataFrame:
        """
        Returns dataframe of all assignment values across iterations,
        along with other statistical information (see self.FACTOR_COLS) 
        per iteration, per count volume.
        """
        return pd.concat(self.results.statistics, ignore_index=True)

    # ODME Execution:
    def execute(self) -> None:
        """ 
        Run ODME algorithm until either the maximum iterations has been reached, 
        or the convergence criterion has been met.
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
            self.convergence_change = float('inf')
            inner = 0
            while inner < self.max_inner and self.convergence_change > self.inner_convergence_crit:
                inner += 1
                self.__execute_inner_iter()
                self.results.log_iter(ODMEResults.INNER)
            
            if self._verbose:
                print(f"Outer iteration {outer} is complete.")

            # Reassign values at the end of each outer loop
            self.__perform_assignment()

        # Add final stats following final assignment:
        self.results.log_iter(ODMEResults.FINAL_LOG)

    def __perform_assignment(self) -> None:
        """ 
        Uses current demand matrix to perform an assignment, then save
        the assigned flows and select link matrices. Also recalculates the 
        objective function following an assignment.

        This function will only be called at the start of an outer
        iteration & during the final convergence test.

        NOTE - Need to check how matrix dimensions will work for multi-class.
        """
        # NEED TO CHECK THAT THIS IS DONE CORRECTLY AND WE DON'T NEED TO CHANGE
        # THE UNDERLYING aeq_matrix.matrices OBJECT INSTEAD! ASK PEDRO ABOUT THIS
        # Change matrix.matrix_view to the current demand matrix (as np.array)
        for aeq_matrix, demand in zip(self.aequilibrae_matrices, self.demands):
            aeq_matrix.matrix_view = demand

        # Perform the assignment
        self.assignment.execute()

        # Store reference to select link demand matrices as proportion matrices
        for assignclass, demand in zip(self.classes, self.demands):
            sl_matrices = assignclass.results.select_link_od.matrix
            for link in sl_matrices:
                self._sl_matrices[link] = np.nan_to_num(
                    sl_matrices[link] / demand
                    )

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
            return assign_df.loc[assign_df['link_id'] == row['link_id'],
                col[row['class']][row['direction']]].values[0]

        # Extract a flow for each count volume:
        self.count_volumes['assign_volume'] = self.count_volumes.apply(
            extract_volume,
            axis=1
        )

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
        Note: we expect any algorithm to return a list of factor matrices in order of the
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
            demand_matrix = self.demands[self.names_to_indices[row['class']]]
            return np.sum(sl_matrix * demand_matrix)

        # Calculate flows for all rows:
        self.count_volumes['assign_volume'] = self.count_volumes.apply(
            lambda row: __calculate_volume(self, row),
            axis=1)
