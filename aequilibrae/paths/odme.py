"""
Implementation of ODME algorithms:
"""

# NOTE - Until issue with select link flows not matching assigned flows ODME should not be used
# with biconjugate/conjugate frank-wolfe

from typing import Tuple
import time
import numpy as np
import scipy.stats as spstats
import pandas as pd

from aequilibrae import TrafficAssignment

class ODME(object):
    """ODME algorithm."""
    COUNT_VOLUME_COLS = ["class", "link_id", "direction", "obs_volume", "assign_volume"]
    DATA_COLS = ["Outer Loop #", "Inner Loop #", "Total Iteration #", "Total Run Time (s)" "Loop Time (s)", "Convergence", "Inner Convergence",
        "class", "link_id", "direction", "obs_volume", "assign_volume", "Assigned - Observed"]
    STATISTICS_COLS = ["Outer Loop #", "Inner Loop #", "Convergence", "Inner Convergence", "Time (s)"]
    FACTOR_COLS = ['Outer Loop #', 'Inner Loop #', 'Total Inner Iteration #', 'mean', 'median', 'std_deviation', 'variance', 'sum',
        'min', 'max']
    GMEAN_LIMIT = 0.01 # FACTOR LIMITING VARIABLE - FOR TESTING PURPOSES
    ALL_ALGORITHMS = ["gmean", "spiess"]

    def __init__(self,
        assignment: TrafficAssignment,
        count_volumes: pd.DataFrame, # [class, link_id, direction, volume]
        stop_crit=(50, 50, 10**-4,10**-4), # max_iterations (inner/outer), convergence criterion
        obj_func=(2, 0), # currently just the objective function specification
        algorithm="spiess" # currently defaults to geometric mean
    ):
        """
        For now see description in pdf file in SMP internship team folder
        Assume for now we only have a single car graph - can be generalised later

        Parameters:
            assignment: the TrafficAssignment object - should be initialised with volume delay functions
                    and their parameters and an assignment algorithm, as well as a TrafficClass containing
                    an initial demand matrix. Doesn't need to have preset select links.
            count_volumes: a dataframe detailing the links, the class they are associated with, the direction
                    and their observed volume. NOTE - CURRENTLY ASSUMING SINGLE CLASS
            stop_crit: the maximum number of iterations and the convergence criterion.
            alg_spec: NOT YET AVAILABLE - will be implemented later to allow user flexibility on what sort 
                    of algorithm they choose.
        """
        # Parameters for assignments
        self.assignment = assignment
        self.classes = assignment.classes
        self.class_names = [user_class.__id__ for user_class in self.classes]
        self.assignclass = self.classes[0] # - for now assume only one class TEMPORARY SINGLE CLASS

        # Demand matrices
        # The current demand matrices
        self.demand_matrices = {user_class.__id__: user_class.matrix.matrix_view for user_class in self.classes}
        self.demand_matrix = self.assignclass.matrix.matrix_view  # The current demand matrix TEMPORARY SINGLE CLASS
        # May be unecessary - if we do keep it need to make a copy ->
        # MAYBE PUT THIS IN AN IF STATEMENT AND ONLY COPY IF A REGULARISATION TERM IS SPECIFIED
        self.init_demand_matrices = [np.copy(matrix) for matrix in self.demand_matrices]
        self._demands_dims = {class_name: self.demand_matrices[class_name].shape for class_name in self.demand_matrices}
        self._demand_dims = self.demand_matrix.shape # Matrix is n x n

        # Observed Links & Associated Volumes
        self._count_volumes = count_volumes.copy(deep=True)
        self._num_counts = len(self._count_volumes)
        self._data = dict() # Contains a dataframe for each inner/outer iteration with all assigned & observed volumes.

        self._sl_matrices = None # Currently dictionary of proportion matrices
        
        # Set all select links:
        #self.assignclass.set_select_links(self.__get_select_links())
        self.__set_select_links()

        # Not yet relevant - Algorithm Specifications:
        self._norms = obj_func
        self._algorithm = algorithm

        # Initialise objective function
        self._obj_func = None
        self.__init_objective_func()
        self._last_convergence = None
        self._convergence_change = float('inf')

        # Stopping criterion
        # May need to specify this further to differentiate between inner & outer criterion
        self.max_outer = stop_crit[0]
        self.max_inner = stop_crit[1]
        self.outer_convergence_crit = stop_crit[2]
        self.inner_convergence_crit = stop_crit[3]

        self._total_iter, self._total_inner, self._outer, self._inner = 0, 0, 0, 0

        # May also want to save the last convergence value.
        # We may also want to store other variables dependent on the algorithm used,
        # e.g. the derivative of link flows w.r.t. step size.

        # Potentially set up some sort of logging information here:

        # Dataframe to log statistical information:
        self._statistics = pd.DataFrame(columns=self.STATISTICS_COLS)

        # Stats on scaling matrices
        self._factor_stats = pd.DataFrame(columns=self.FACTOR_COLS)

        # Time data for logging information
        self._total_time = 0
        self._time = None

    # Utilities:
    def __set_select_links(self) -> None:
        """
        Sets all select links for each class and for each observation.
        """
        cv = self._count_volumes
        for user_class in self.classes:
            user_class.set_select_links(
                {
                    self.__get_sl_key(row): [(row['link_id'], row['direction'])]
                    for _, row in
                    cv[cv['class'] == user_class.__id__].iterrows()
                }
            )

    def __get_sl_key(self, row: pd.Series) -> str:
        """
        Given a particular row from the observervations (count_volumes) returns
        a key corresponding to it for use in all select link extraction.
        """
        return f"sl_{row['class']}_{row['link_id']}_{row['direction']}"

    def __increment_outer(self) -> None:
        """
        Increments outer iteration number, increments total iterations and zeros inner iteration number.
        """
        self._outer += 1
        self._inner = 0
        self._total_iter += 1

    def __increment_inner(self) -> None:
        """
        Increments inner iteration number and total iteration and total inner iteration number.
        """
        self._inner += 1
        self._total_iter += 1
        self._total_inner += 1

    def __set_convergence_values(self, new_convergence: float) -> None:
        """
        Given a new convergence value calculates the difference between the previous convergence
        and new convergence, and sets appropriate values.
        """
        if self._last_convergence:
            self._convergence_change = abs(self._last_convergence - new_convergence)
        self._last_convergence = new_convergence

    def __init_objective_func(self) -> None:
        """
        Initialises the objective function - parameters must be specified by user.

        Current objective functions have 2 parts which are summed:
            1. The p-norm raised to the power p of the error vector for observed flows.
            2. The p-norm raised to the power p of the error matrix (treated as a n^2 vector) for the demand matrix.
        
        (1.) must always be present, but (2.) (the regularisation term) need not be present (ie, specified as 0 by user).
        Default currently set to l1 (manhattan) norm for (1.) with no regularisation term (p2 = 0).
        """
        p_1 = self._norms[0]
        p_2 = self._norms[1]

        def __reg_obj_func(self) -> None:
            """
            Objective function containing regularisation term.

            NOTE - NOT YET READY FOR USE! REGULARISATION TERM SHOULD BE ALPHA/BETA WEIGHTED!
            """
            obs_vals = self._count_volumes["obs_volume"].to_numpy()
            assign_vals = self._count_volumes['assign_volume'].to_numpy()
            obj1 = np.sum(np.abs(obs_vals - assign_vals)**p_1) / p_1
            regularisation = np.sum(np.abs(self.init_demand_matrix - self.demand_matrix)**p_2) / p_2
            self.__set_convergence_values(obj1 + regularisation)

        def __obj_func(self) -> None:
            """
            Objective function with no regularisation term.
            """
            obs_vals = self._count_volumes["obs_volume"].to_numpy()
            assign_vals = self._count_volumes['assign_volume'].to_numpy()
            self.__set_convergence_values(np.sum(np.abs(obs_vals - assign_vals)**p_1) / p_1)

        if p_2:
            self._obj_func = __reg_obj_func
        else:
            self._obj_func = __obj_func

    # Output/Results/Statistics:
    def get_results(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Returns final demand matrix and a dataframe of statistics regarding
        timing and convergence.

        CURRENTLY ONLY WORKS FOR SINGLE CLASS!!!
        NEED TO CHANGE ALL OF THESE TO BE MORE COHERENT
        """
        # return (self.demand_matrices, self._statistics)
        return (self.demand_matrix, self._statistics)

    def get_factor_stats(self) -> pd.DataFrame:
        """
        Returns a dataframe on statistics of factors for every iteration.
        """
        return self._factor_stats

    def get_assignment_data(self) -> pd.DataFrame:
        """
        Returns dataframe of all assignment values across iterations.
        """
        assignment_data = pd.concat(
            [self._data[self.__get_data_key(row['Outer Loop #'], row['Inner Loop #'])]
            for _, row in self._statistics.iterrows()
        ],
            ignore_index=True
        )
        return assignment_data

    def __get_data_key(self, outer: int, inner: int) -> str:
        """
        Returns a key for a particular set of assignment data corresponding
        to a particular outer/inner iteration.
        """
        # Currently fudging this to make the types the same always
        return f"data_{int(outer)}_{int(inner)}"

    def __log_stats(self) -> None:
        """
        Adds next row to statistics dataframe and data dictionary.
        """
        # Statistics DataFrame:
        old_time = self._time
        self._time = time.time()
        loop_time = self._time - old_time
        self._total_time += loop_time
        to_log = [self._outer, self._inner, self._last_convergence, self._convergence_change, loop_time]

        # Add row:
        self._statistics.loc[len(self._statistics)] = {
            col : to_log[i]
            for i, col in enumerate(self.STATISTICS_COLS)
        }

        # Create Data:
        data = self._count_volumes.copy(deep=True)
        data["Loop Time (s)"] = [loop_time for _ in range(self._num_counts)]
        data["Total Run Time (s)"] = [self._total_time for _ in range(self._num_counts)]
        data["Convergence"] = [self._last_convergence for _ in range(self._num_counts)]
        data["Inner Convergence"] = [self._convergence_change for _ in range(self._num_counts)]
        data["Total Iteration #"] = [self._total_iter for _ in range(self._num_counts)]
        data["Outer Loop #"] = [self._outer for _ in range(self._num_counts)]
        data["Inner Loop #"] = [self._inner for _ in range(self._num_counts)]
        data["Assigned - Observed"] = (self._count_volumes['assign_volume'].to_numpy() -
            self._count_volumes["obs_volume"].to_numpy())

        self._data[self.__get_data_key(self._outer, self._inner)] = data

    def ___record_factor_stats(self, factors: np.ndarray) -> None:
        """
        Logs information on the current scaling matrix.
        """
        factor_stats = [
            self._outer,
            self._inner,
            self._total_inner,
            np.mean(factors),
            np.median(factors),
            np.std(factors),
            np.var(factors),
            np.sum(factors),
            np.min(factors),
            np.max(factors),
        ]

        # Add row:
        self._factor_stats.loc[len(self._factor_stats)] = {
            col : factor_stats[i]
            for i, col in enumerate(self.FACTOR_COLS)
        }

    # Generic Algorithm Structure:
    def execute(self) -> None:
        """ 
        Run ODME algorithm until either the maximum iterations has been reached, 
        or the convergence criterion has been met.
        """
        # Initialise timing:
        self._time = time.time()

        # Create values for SL matrices & assigned flows
        self.__perform_assignment()

        # Begin outer iteration
        # OUTER STOPPING CRITERION - CURRENTLY TEMPORARY VALUE
        while self._outer < self.max_outer and self._last_convergence > self.outer_convergence_crit:
            # Set iteration values:
            self.__increment_outer()
            self.__log_stats()

            # Run inner iterations:
            # INNER STOPPING CRITERION - FIND A BETTER WAY TO DO INNER STOPPING CRITERION
            # MAYBE BASED ON DIFFERENCE IN CONVERGENCE
            self._convergence_change = float('inf')
            while self._inner < self.max_inner and self._convergence_change > self.inner_convergence_crit:
                self.__execute_inner_iter()
                self.__increment_inner()
                self.__log_stats()

            # Reassign values at the end of each outer loop
            self.__perform_assignment()
        
        # Add final stats following final assignment:
        self._outer += 1
        self._inner = 0
        self.__log_stats()

    def __execute_inner_iter(self) -> None:
        """
        Runs an inner iteration of the ODME algorithm. 
        This assumes the SL matrices stay constant and modifies
        the current demand matrix.
        """
        # Element-wise multiplication of demand matrix by scaling factor
        self.demand_matrix = self.demand_matrix * self.__get_scaling_factor()

        # Recalculate the link flows
        self.__calculate_flows()

        # Recalculate convergence level:
        self._obj_func(self)

    def __get_scaling_factor(self) -> np.ndarray:
        """
        Returns scaling matrix - depends on algorithm chosen.
        Currently implementing default as geometric mean.
        """
        if self._algorithm == "gmean":
            scaling_factor = self.__geometric_mean()
        elif self._algorithm == "spiess":
            scaling_factor = self.__spiess()
        else: # SHOULD NEVER HAPPEN - RAISE ERROR HERE LATER AND ERROR SHOULD HAVE BEEN RAISED EARLIER!!!
            scaling_factor = np.ones(self._demand_dims)

        self.___record_factor_stats(scaling_factor)
        return scaling_factor

    def __perform_assignment(self) -> None:
        """ 
        Uses current demand matrix to perform an assignment, then save
        the results in the relevant fields.
        This function will only be called at the start of an outer
        iteration & during the final convergence test.
        """
        # Change matrix.matrix_view to the current demand matrix (as np.array)
        self.assignclass.matrix.matrix_view = self.demand_matrix

        # Perform the assignment
        self.assignment.execute()
        # TEMPORARY FIX - I DON'T REALLY KNOW WHY WE HAVE AN EXTRA DIMENSION NOW BUT I'LL FLATTEN
        # IT SINCE IT ISN'T RELEVANT TO SINGLE CLASS OR SINGLE COUNT CASES
        self.assignclass.matrix.matrix_view = np.squeeze(self.assignclass.matrix.matrix_view, axis=2)

        # Store reference to select link demand matrices as proportion matrices
        # Can completely ignore old SL matrices from this point
        self._sl_matrices = self.assignclass.results.select_link_od.matrix
        for link in self._sl_matrices:
            self._sl_matrices[link] = np.nan_to_num(np.squeeze(self._sl_matrices[link], axis=2) / self.demand_matrix)
        # NOTE - squeeze since multiple matrices are stored for select link or class (ask Jamie/Jake),
        # but we only have one of each per set of select links so we can ignore this for now.
        # In future when multiple class ODME is implemented this needs to be changed.

        # Extract and store array of assigned volumes to select links
        assign_df = self.assignment.results().reset_index(drop=False).fillna(0)
        col = {1: "matrix_ab", -1: "matrix_ba", 0: "matrix_tot"}
        
        def extract_flow(row) -> None:
            """
            Extracts flow corresponding to particular link (from row) and return it.

            NOT YET GENERALISED FOR MULTI-CLASS!!!
            """
            # ^For inner iterations need to calculate this via sum sl_matrix * demand_matrix
            return assign_df.loc[assign_df["link_id"] == row["link_id"],
                col[row["direction"]]].values[0]

        self._count_volumes['assign_volume'] = self._count_volumes.apply(
            lambda row: extract_flow(row),
            axis=1
        )      

        # Recalculate convergence values
        self._obj_func(self)

    def __calculate_flows(self) -> None:
        """
        Calculates and stores link flows using current sl_matrices & demand matrix.
        """

        def __calculate_flow(self, row: pd.Series) -> float:
            """
            Given a single row of the count volumes dataframe, 
            calculates the appropriate corresponding assigned 
            value.
            """
            sl_matrix = self._sl_matrices[self.__get_sl_key(row)]
            return np.sum(sl_matrix * self.demand_matrix)

        self._count_volumes['assign_volume'] = self._count_volumes.apply(
            lambda row: __calculate_flow(self, row),
            axis=1)

    # Algorithm Specific Functions:
    def __geometric_mean(self) -> np.ndarray:
        """
        Calculates scaling factor based on geometric mean of ratio between 
        proportionally (via SL matrix) assigned flow & observed flows.

        CURRENTLY ONLY IMPLEMENTED FOR SINGLE CLASS
        """
        # Steps:
        # 1. For each link create a factor f_a given by \hat v_a / v_a
        # 2. Create a matrix of factors for each link where for a given OD pair i,
        #    the factor at i is f_a if the corresponding value in the SL matrix
        #    is non-zero, and 1 otherwise.
        # 3. Return the geometric mean of all the factor matrices component-wise
        # NOTE - This may be slower due to holding all these matrices in memory
        # simultaneously. It is possible to do this e.g element-wise or row-wise
        # to save on memory usage. Need to test this later on.
        # NOTE - by not approximating step size we may over-correct massively.

        # Steps 1 & 2:
        factors = np.empty((len(self._count_volumes), *(self._demand_dims)))
        for i, row in self._count_volumes.iterrows():
            # Create factor matrix:
            if row["obs_volume"] != 0 and row['assign_volume'] != 0:

                # Modulate factor by select link dependency:
                link_factor = (row['obs_volume'] / row['assign_volume']) - 1
                sl_matrix = self._sl_matrices[self.__get_sl_key(row)]
                factor_matrix = (sl_matrix * link_factor)

                # Apply factor limiting:
                # factor_matrix = np.clip(factor_matrix, -self.GMEAN_LIMIT, self.GMEAN_LIMIT)

                # Center factors at 1:
                factor_matrix = factor_matrix + 1

            # If assigned or observed value is 0 we cannot do anything right now
            else:
                factor_matrix = np.ones(self._demand_dims)
            
            factors[i, :, :] = factor_matrix

        # If the assigned volume was 0 (or both 0) no OD pair can have any effect
        factors = np.nan_to_num(factors, nan=1, posinf=1, neginf=1)

        # Step 3:
        return spstats.gmean(factors, axis=0)

    def __spiess(self) -> np.ndarray:
        """
        Calculates scaling factor based on gradient descent method via SL matrix,
        assigned flow & observed flows as described by Spiess (1990) - REFERENCE HERE

        CURRENTLY ONLY IMPLEMENTED FOR SINGLE CLASS
        """
        gradient_matrix = self.__get_derivative_matrix_spiess() # Derivative matrix for spiess algorithm
        step_size = self.__get_step_size_spiess(gradient_matrix) # Get optimum step size for current iteration
        return 1 - (step_size * gradient_matrix)
    
    def __get_derivative_matrix_spiess(self) -> np.ndarray:
        """
        Returns derivative matrix (see Spiess (1990) - REFERENCE HERE)

        CURRENTLY ONLY IMPLEMENTED FOR SINGLE CLASS
        """
        # There are probably some numpy/cython ways to do this in parallel and
        # without storing too many things in memory.
        factors = np.empty((len(self._count_volumes), *(self._demand_dims)))
        for i, row in self._count_volumes.iterrows():
            sl_matrix = self._sl_matrices[self.__get_sl_key(row)]
            factors[i, :, :] = sl_matrix * (row['assign_volume'] - row['obs_volume'])

        return np.sum(factors, axis=0)

    def __get_step_size_spiess(self, gradient: np.ndarray) -> float:
        """
        Returns estimate of optimal step size (see Spiess (1990) - REFERENCE HERE)

        Parameters:
            gradient: The currently calculating gradient matrix - required for calculating 
                derivative of link flows with respect to step size.

        CURRENTLY ONLY IMPLEMENTED FOR SINGLE CLASS
        """
        upper_lim, lower_lim = self.__get_step_size_limits_spiess(gradient)

        # Calculating link flow derivatives:
        flow_derivatives = np.empty(self._num_counts)
        for i, row in self._count_volumes.iterrows():
            sl_matrix = self._sl_matrices[self.__get_sl_key(row)]
            flow_derivatives[i] = -np.sum(self.demand_matrix * sl_matrix * gradient)
        
        # Calculate minimising step length:
        errors = self._count_volumes['obs_volume'].to_numpy() - self._count_volumes['assign_volume'].to_numpy()
        min_lambda = np.sum(flow_derivatives * errors) / np.sum(np.square(flow_derivatives))
         # Can only happen if all flow derivatives are 0 - ie we should not bother perturbing matrix
        if np.isnan(min_lambda):
            min_lambda = 0

        if min_lambda > upper_lim:
            return upper_lim
        elif min_lambda < lower_lim:
            return lower_lim
        else: # Minimum step size does not violate addition step size constraints
            return min_lambda

    def __get_step_size_limits_spiess(self, gradient: np.ndarray) -> Tuple[float, float]:
        """
        Returns bounds for step size in order of upper bound, then lower bound (see Spiess
        (1990) - REFERENCE HERE)

        Parameters:
            gradient: The currently calculating gradient matrix - required for calculating 
                derivative of link flows with respect to step size.

        CURRENTLY ONLY IMPLEMENTED FOR SINGLE CLASS
        """
        upper_mask = np.logical_and(self.demand_matrix > 0, gradient > 0)
        lower_mask = np.logical_and(self.demand_matrix > 0, gradient < 0)

        # Account for either mask being empty
        if np.any(upper_mask):
            upper_lim = 1 / np.min(gradient[upper_mask])
        else:
            upper_lim = float('inf')
        if np.any(lower_mask):
            lower_lim = 1 / np.max(gradient[lower_mask])
        else:
            lower_lim = float('-inf')

        return (upper_lim, lower_lim)
