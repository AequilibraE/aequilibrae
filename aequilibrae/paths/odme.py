"""
Implementation of ODME Infrastructure:
"""

# NOTE - Until issue with select link flows not matching assigned flows ODME should not be used
# with biconjugate/conjugate frank-wolfe

# NOTE - Lots of squeezing of matrices happens after assignment due to the functionality of select 
# link analysis and assignment with regards to traffic assignment.

# NOTE - Functions which are still Single Class Only include:
#           Initialiser - extraction of pce's & use of class to indices?
#               -> Needs to be seriously cleaned up.
#           Objective Function - Check how this works with pce
#           Extraction of Flows - Check how this works with pce
#           Calculation of Flows - Check how this works with pce
#
#               All the actual algorithms (but these should be done separately
#               and moved to a different class where they can interact with
#               Cython and be ran far more efficiently). I also need to re-derive
#               and very carefully go through the assumptions when finding the generalisation
#               of these algorithms to multi-class.

from typing import Tuple
import time
import numpy as np
import scipy.stats as spstats
import pandas as pd

from aequilibrae.paths import TrafficAssignment

class ODME(object):
    """ODME Infrastructure"""
    COUNT_VOLUME_COLS = ["class", "link_id", "direction", "obs_volume", "assign_volume"]
    DATA_COLS = ["Outer Loop #", "Inner Loop #", "Total Iteration #", "Total Run Time (s)" "Loop Time (s)", "Convergence", "Inner Convergence",
        "class", "link_id", "direction", "obs_volume", "assign_volume", "Assigned - Observed"]
    STATISTICS_COLS = ["Outer Loop #", "Inner Loop #", "Convergence", "Inner Convergence", "Time (s)"]
    FACTOR_COLS = ['class', 'Outer Loop #', 'Inner Loop #', 'Total Inner Iteration #', 'mean', 'median',
        'std_deviation', 'variance', 'sum', 'min', 'max']
    CUMULATIVE_FACTOR_COLS = ["class", "mean", "median", "standard deviation", "variance", "min", "max", "sum", "# of factors"]
    GMEAN_LIMIT = 0.01 # FACTOR LIMITING VARIABLE - FOR TESTING PURPOSES
    ALL_ALGORITHMS = ["gmean", "spiess"]

    def __init__(self,
        assignment: TrafficAssignment,
        count_volumes: pd.DataFrame, # [class, link_id, direction, volume]
        stop_crit=(50, 50, 10**-4,10**-4), # max_iterations (inner/outer), convergence criterion
        obj_func=(2, 0), # currently just the objective function specification
        algorithm="spiess" # currently defaults to spiess
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

        CURRENTLY ONLY IMPLEMENTED FOR SINGLE CLASS (MULTI-CLASS UNDER DEVELOPMENT)
        """
        # Parameters for assignments
        self.assignment = assignment
        self.classes = assignment.classes
        self.num_classes = len(self.classes)
        self.single_class = (self.num_classes == 1) # If we are doing single class ODME
        # Everything is implicitly ordered by this:
        self.class_names = [user_class.__id__ for user_class in self.classes]
        self.names_to_indices = {name: index for index, name in enumerate(self.class_names)}

        self.aequilibrae_matrices = [user_class.matrix for user_class in self.classes]
        # Current demand matrices:
        self.demand_matrices = [user_class.matrix.matrix_view for user_class in self.classes]
        # May be unecessary - if we do keep it need to make a copy ->
        # MAYBE PUT THIS IN AN IF STATEMENT AND ONLY COPY IF A REGULARISATION TERM IS SPECIFIED
        # Initial demand matrices:
        self.init_demand_matrices = [np.copy(matrix) for matrix in self.demand_matrices]
        self.demand_dims = [self.demand_matrices[i].shape for i in range(self.num_classes)]

        # Observed Links & Associated Volumes
        # MAYBE I SHOULD SPLIT THIS INTO ONE DATAFRAME PER CLASS
        self.count_volumes = count_volumes.copy(deep=True)
        self.num_counts = len(self.count_volumes)

        self._sl_matrices = dict() # Dictionary of proportion matrices

        # Set all select links:
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
        self._statistics = []

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
        cv = self.count_volumes
        for user_class in self.classes:
            user_class.set_select_links(
                {
                    self.get_sl_key(row):
                    [(row['link_id'], row['direction'])]
                    for _, row in cv[cv['class'] == user_class.__id__
                    ].iterrows()
                }
            )

    def get_sl_key(self, row: pd.Series) -> str:
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

            ONLY IMPLEMENTED FOR SINGLE CLASS!
            """
            obs_vals = self._count_volumes["obs_volume"].to_numpy()
            assign_vals = self._count_volumes['assign_volume'].to_numpy()
            obj1 = np.sum(np.abs(obs_vals - assign_vals)**p_1) / p_1
            regularisation = np.sum(np.abs(self.init_demand_matrices[0] - self.demand_matrices[0])**p_2) / p_2
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
    def get_demands(self) -> list[np.ndarray]:
        """
        Returns all demand matrices (can be called before or after execution).
        """
        return self.demand_matrices

    def get_iteration_factors(self) -> pd.DataFrame:
        """
        Returns a dataframe on statistics of factors for each iteration.
        """
        return self._factor_stats

    def get_cumulative_factors(self) -> pd.DataFrame:
        """
        Return the cumulative factors (ratio of final to initial matrix) in a dataframe.
        """
        # Get cumulative factors for each demand matrix
        cumulative_factors = []
        for i, demand_matrix in enumerate(self.demand_matrices):
            factors = np.nan_to_num(demand_matrix / self.init_demand_matrices[i], nan=1)
            cumulative_factors.append(
                pd.DataFrame({
                    "class": [self.class_names[i] for _ in range(demand_matrix.size)],
                    "Factors": factors.ravel()
                })
            )

        return pd.concat(cumulative_factors, ignore_index=True)

    def get_all_statistics(self) -> pd.DataFrame:
        """
        Returns dataframe of all assignment values across iterations,
        along with other statistical information (see self.FACTOR_COLS) 
        per iteration, per count volume.
        """
        return pd.concat(self._statistics, ignore_index=True)

    def __log_stats(self) -> None:
        """
        Computes statistics regarding previous iteration and stores them in the statistics list.
        """
        # Compute Statistics:
        old_time = self._time
        self._time = time.time()
        loop_time = self._time - old_time
        self._total_time += loop_time

        # Create Data:
        data = self.count_volumes.copy(deep=True)
        data["Loop Time (s)"] = [loop_time for _ in range(self.num_counts)]
        data["Total Run Time (s)"] = [self._total_time for _ in range(self.num_counts)]
        data["Convergence"] = [self._last_convergence for _ in range(self.num_counts)]
        data["Inner Convergence"] = [self._convergence_change for _ in range(self.num_counts)]
        data["Total Iteration #"] = [self._total_iter for _ in range(self.num_counts)]
        data["Outer Loop #"] = [self._outer for _ in range(self.num_counts)]
        data["Inner Loop #"] = [self._inner for _ in range(self.num_counts)]
        data["Assigned - Observed"] = (self.count_volumes['assign_volume'].to_numpy() -
            self.count_volumes["obs_volume"].to_numpy())

        # Add data to current list of dataframes
        self._statistics.append(data)

    def __record_factor_stats(self, factors: list[np.ndarray]) -> None:
        """
        Logs information on the current scaling matrix (ie
        factor statistics per iteration per class).
        """
        # Create statistics on all new factors:
        data = []
        for i, factor in enumerate(factors):
            data.append([
                self.class_names[i],
                self._outer,
                self._inner,
                self._total_inner,
                np.mean(factor),
                np.median(factor),
                np.std(factor),
                np.var(factor),
                np.sum(factor),
                np.min(factor),
                np.max(factor)
            ])
        new_stats = pd.DataFrame(data, columns=self.FACTOR_COLS)

        # Add the new data to the current list of factor statistics
        self._factor_stats = pd.concat([self._factor_stats, new_stats], ignore_index=True)

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
            self._convergence_change = float('inf') # Ensures at least 1 inner convergence is run per loop
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
        the current demand matrices.
        """
        # Element-wise multiplication of demand matrices by scaling factors
        factors = self.__get_scaling_factors()
        for i, factor in enumerate(factors):
            self.demand_matrices[i] = self.demand_matrices[i] * factor

        # Recalculate the link flows
        self.__calculate_volumes()

        # Recalculate convergence level:
        self._obj_func(self)

    def __get_scaling_factors(self) -> list[np.ndarray]:
        """
        Returns scaling matrices for each user class - depending on algorithm chosen.
        Note: we expect any algorithm to return a list of factor matrices in order of the
        stored user classes.

        NOTE - In future we should separate the algorithms from this class, and this function
        will be the only one which ever needs to interact with the algorithms, and simply needs
        to receive a list of scaling factors after initialising an algorithm and passing it the current
        state of this ODME object.
        """
        if self._algorithm == "gmean":
            scaling_factors = self.__geometric_mean()
        elif self._algorithm == "spiess":
            scaling_factors = self.__spiess()
        else: # SHOULD NEVER HAPPEN - RAISE ERROR HERE LATER AND ERROR SHOULD HAVE BEEN RAISED EARLIER!!!
            scaling_factors = [np.ones(dims) for dims in self.demand_dims]
            raise ValueError("Unsupported Algorithm!")

        self.__record_factor_stats(scaling_factors)
        return scaling_factors

    def __perform_assignment(self) -> None:
        """ 
        Uses current demand matrix to perform an assignment, then save
        the assigned flows and select link matrices. Also recalculates the 
        objective function following an assignment.

        This function will only be called at the start of an outer
        iteration & during the final convergence test.

        NOTE - Need to check how matrix dimensions will work for multi-class.
        """
        # Change matrix.matrix_view to the current demand matrix (as np.array)
        for i, assignclass in enumerate(self.classes):
            assignclass.matrix.matrix_view = self.demand_matrices[i]

        # Perform the assignment
        self.assignment.execute()
        
        # TEMPORARY FIX - I DON'T REALLY KNOW WHY WE HAVE AN EXTRA DIMENSION NOW BUT I'LL FLATTEN
        # IT SINCE IT ISN'T RELEVANT TO SINGLE CLASS OR SINGLE COUNT CASES
        for assignclass in self.classes:
            assignclass.matrix.matrix_view = np.squeeze(assignclass.matrix.matrix_view, axis=2)

        # Store reference to select link demand matrices as proportion matrices
        # MULTI-CLASS GENERALISATION REQUIRES TESTING IN FUTURE!!!
        for i, assignclass in enumerate(self.classes):
            sl_matrices = assignclass.results.select_link_od.matrix
            for link in sl_matrices:
                self._sl_matrices[link] = np.nan_to_num(
                    np.squeeze(sl_matrices[link], axis=2) / self.demand_matrices[i])
        # NOTE - squeeze since multiple matrices are stored for select link or class (ask Jamie/Jake),
        # but we only have one of each per set of select links so we can ignore this for now.
        # In future when multiple class ODME is implemented this needs to be checked/changed.

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
        # DECIDE WHETHER TO PUT THIS IN INITIALISER OR NOT!!!
        # Dictionary to select correct column of results dataframe
        col = dict()
        for i, cls_name in enumerate(self.class_names):
            # NOTE - due to the design change of the TrafficClass to only hold one
            # user class, this should not be necessary, however this is still a remnant
            # piece of code which uses the names from the aequilibrae matrix itself.
            name = self.aequilibrae_matrices[i].view_names[0]
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
            sl_matrix = self._sl_matrices[self.__get_sl_key(row)]
            demand_matrix = self.demand_matrices[self.names_to_indices[row['class']]]
            return np.sum(sl_matrix * demand_matrix)

        # Calculate flows for all rows:
        self.count_volumes['assign_volume'] = self.count_volumes.apply(
            lambda row: __calculate_volume(self, row),
            axis=1)


    # Algorithm Specific Functions:

    # gmean (Geometric Mean):
    def __geometric_mean(self) -> np.ndarray:
        """
        Calculates scaling factor based on geometric mean of ratio between 
        proportionally (via SL matrix) assigned flow & observed flows.

        MULTI-CLASS UNDER DEVELOPMENT! (REQUIRES TESTING)
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

        scaling_factors = []
        c_v = self.count_volumes
        # Steps 1 & 2:
        for i, name in enumerate(self.class_names):
            observed = c_v[c_v['class'] == name]

            # If there are no observations leave matrix unchanged
            if len(observed) == 0:
                scaling_factors.append(np.ones(self.demand_dims[i]))
                continue

            factors = np.empty((len(observed), *(self.demand_dims[i])))
            for j, row in self.count_volumes.iterrows():
                # Create factor matrix:
                if row["obs_volume"] != 0 and row['assign_volume'] != 0:

                    # Modulate factor by select link dependency:
                    link_factor = (row['obs_volume'] / row['assign_volume']) - 1
                    sl_matrix = self._sl_matrices[self.get_sl_key(row)]
                    factor_matrix = (sl_matrix * link_factor)

                    # Apply factor limiting:
                    # factor_matrix = np.clip(factor_matrix, -self.GMEAN_LIMIT, self.GMEAN_LIMIT)

                    # Center factors at 1:
                    factor_matrix = factor_matrix + 1

                # If assigned or observed value is 0 we cannot do anything right now
                else:
                    factor_matrix = np.ones(self.demand_dims[i])
                
                # Add factor matrix
                factors[j, :, :] = factor_matrix

            # If the assigned volume was 0 (or both 0) no OD pair can have any effect
            factors = np.nan_to_num(factors, nan=1, posinf=1, neginf=1)

        # Add the factors for this class to the array:
        scaling_factors.append(spstats.gmean(factors, axis=0))

        # Step 3:
        return scaling_factors

    # spiess (Gradient Descent - Objective Function (2,0))
    def __spiess(self) -> list[np.ndarray]:
        """
        Calculates scaling factor based on gradient descent method via SL matrix,
        assigned flow & observed flows as described by Spiess (1990) - REFERENCE HERE

        MULTI-CLASS UNDER DEVELOPMENT!
        """
        # THIS CAN BE SIGNIFICANTLY SIMPLIFIED - SEE NOTES

        # Derivative matrices for spiess algorithm:
        gradient_matrices = self.__get_derivative_matrices_spiess()

        # Get optimum step sizes for current iteration:
        step_sizes = self.__get_step_sizes_spiess(gradient_matrices)

        # Get scaling factors:
        scaling_factors = [1 - (step_sizes[i] * gradient_matrices[i]) for i in range(self.num_classes)]
        return scaling_factors

    def __get_derivative_matrices_spiess(self) -> list[np.ndarray]:
        """
        Returns derivative matrix (see Spiess (1990) - REFERENCE HERE)

        MULTI-CLASS UNDER DEVELOPMENT!
        """
        # THIS CAN BE SIGNIFICANTLY SIMPLIFIED - SEE NOTES

        # There are probably some numpy/cython ways to do this in parallel and
        # without storing too many things in memory.
        derivatives = []
        # Create a derivative matrix for each user class:
        for i, user_class in enumerate(self.class_names):
            observed = self.count_volumes[self.count_volumes['class'] == user_class]
            factors = np.empty((len(observed), *(self.demand_dims[i])))
            for i, row in observed.iterrows():
                sl_matrix = self._sl_matrices[self.get_sl_key(row)]
                factors[i, :, :] = sl_matrix * (row['assign_volume'] - row['obs_volume'])

            # Add derivative matrix to list of derivatives:
            derivatives.append(np.sum(factors, axis=0))

        return derivatives

    def __get_step_sizes_spiess(self, gradients: list[np.ndarray]) -> list[float]:
        """
        Returns estimate of optimal step size (see Spiess (1990) - REFERENCE HERE)

        Parameters:
            gradients: The previously calculated gradient matrices - required for calculating 
                derivative of link flows with respect to step size.

        MULTI-CLASS UNDER DEVELOPMENT!
        """
        # THIS CAN BE SIGNIFICANTLY SIMPLIFIED - SEE NOTES

        bounds = self.__get_step_size_limits_spiess(gradients)
        upper_lim, lower_lim = bounds[0]

        # SOME MINOR OPTIMISATIONS CAN BE DONE HERE IN TERMS OF WHAT PRECISELY TO CALCULATE:
        # Calculate step-sizes (lambdas) for each gradient matrix:
        lambdas = []
        for i, user_class in enumerate(self.class_names):
            class_counts = self.count_volumes[self.count_volumes['class'] == user_class]

            # Calculating link flow derivatives:
            flow_derivatives = np.empty(len(class_counts))
            for j, row in class_counts.iterrows():
                sl_matrix = self._sl_matrices[self.get_sl_key(row)]
                flow_derivatives[j] = -np.sum(self.demand_matrices[i] * sl_matrix * gradients[i])

            # Calculate minimising step length:
            errors = class_counts['obs_volume'].to_numpy() - class_counts['assign_volume'].to_numpy()
            min_lambda = np.sum(flow_derivatives * errors) / np.sum(np.square(flow_derivatives))

            # This can only happen if all flow derivatives are 0 - ie we should not bother perturbing matrix
            if np.isnan(min_lambda):
                min_lambda = 0

            # Check minimising lambda is within bounds:
            upper_lim, lower_lim = bounds[i]
            if min_lambda > upper_lim:
                lambdas.append(upper_lim)
            elif min_lambda < lower_lim:
                lambdas.append(lower_lim)
            else: # Minimum step size does not violate addition step size constraints
                lambdas.append(min_lambda)
        
        return lambdas

    def __get_step_size_limits_spiess(self,
            gradients: list[np.ndarray]) -> list[Tuple[float, float]]:
        """
        Returns bounds for step size in order of upper bound, then lower bound (see Spiess
        (1990) - REFERENCE HERE) for each gradient matrix.

        Parameters:
            gradient: The currently calculating gradient matrix - required for calculating 
                derivative of link flows with respect to step size.

        MULTI-CLASS UNDER DEVELOPMENT!
        """
        # THIS CAN BE SIGNIFICANTLY SIMPLIFIED - SEE NOTES
        bounds = []
        # Create each bound and check for edge cases with empty sets:
        for i, gradient in enumerate(gradients):
            # Upper bound:
            upper_mask = np.logical_and(self.demand_matrices[i] > 0, gradient > 0)
            if np.any(upper_mask):
                upper_lim = 1 / np.min(gradient[upper_mask])
            else:
                upper_lim = float('inf')

            # Lower bound:
            lower_mask = np.logical_and(self.demand_matrices[i] > 0, gradient < 0)
            if np.any(lower_mask):
                lower_lim = 1 / np.max(gradient[lower_mask])
            else:
                lower_lim = float('-inf')

            bounds.append((upper_lim, lower_lim)) # Tuple[float, float]

        return bounds
