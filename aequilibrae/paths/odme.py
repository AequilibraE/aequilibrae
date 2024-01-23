"""
Implementation of ODME Infrastructure:
"""

# NOTE - Until issue with select link flows not matching assigned flows ODME should not be used
# with biconjugate/conjugate frank-wolfe

# NOTE - Lots of squeezing of matrices happens after assignment due to the functionality of select
# link analysis and assignment with regards to traffic assignment.

# NOTE - We need to use matrix.view_names[0] to access the appropriate results rather than the class
# __id__, due to some artifacts of previous design choices/changes.

# NOTE - To Do:
#       Initialiser -> Needs to be seriously cleaned up.
#       Objective Function -> Needs to be updated to allowed for regularisation term
#                          -> May be useful to consider normalising alpha/beta
#                          -> Needs to be updated to include pce
#       Execution -> Need to work later on a better way to automate inner stopping criterion

# Ideally in future this class should act as an entirely top level class for user interaction.
# I.e, the user should be able to intialise, set parameters, call execute and get various results
# but this class does not need to hold any of the actual algorithms or statistics itself.
# It probably should also do any checking of user input that is required
# (ie no client classes should have to check)

import numpy as np
import pandas as pd
from typing import Tuple

from aequilibrae.paths import TrafficAssignment
from aequilibrae.paths.odme_submodule import ScalingFactors, ODMEResults

class ODME(object):
    """ ODME Infrastructure """
    COUNT_VOLUME_COLS = ["class", "link_id", "direction", "obs_volume", "assign_volume"]
    GMEAN_LIMIT = 0.01 # FACTOR LIMITING VARIABLE - FOR TESTING PURPOSES - DEFUNCT!
    ALL_ALGORITHMS = ["gmean", "spiess", "reg_spiess"]

    # DOCSTRING NEEDS UPDATING
    def __init__(self,
        assignment: TrafficAssignment,
        count_volumes: pd.DataFrame, # [class, link_id, direction, volume]
        stop_crit=(50, 50, 10**-4,10**-4), # max_iterations (inner/outer), convergence criterion
        alpha: float = None, # Used for regularisation - should be given in form (alpha, beta) as a Tuple
        algorithm: str = "spiess", # currently defaults to spiess
        verbose: bool = False # For printing as we go
    ):
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
        self._norms = self.__get_norms(algorithm)
        self._algorithm = algorithm

        # Initialise objective function
        self._obj_func = None
        self.__init_objective_func()
        self.last_convergence = None
        self.flow_obj = None # Component of objective function from flows
        self.reg_obj = None # Component of objective function from regularisation
        self.convergence_change = float('inf')

        # Stopping criterion
        # May need to specify this further to differentiate between inner & outer criterion
        self.max_outer = stop_crit[0]
        self.max_inner = stop_crit[1]
        self.outer_convergence_crit = stop_crit[2]
        self.inner_convergence_crit = stop_crit[3]

        # Hyper-parameters for regularisation:
        if algorithm in ["reg_spiess"]:
            if alpha is None or alpha > 1 or alpha < 0: # THIS CHECK SHOULD PROBABLY BE MORE ROBUST
                raise ValueError("Hyper-parameter alpha should be between 0 and 1")
            self.alpha = alpha
            self.beta = 1 - alpha

        # May also want to save the last convergence value.
        # We may also want to store other variables dependent on the algorithm used,
        # e.g. the derivative of link flows w.r.t. step size.

        # RESULTS & STATISTICS (NEW VERSION)
        self.results = ODMEResults(self)

        # For printing ongoing state
        self._verbose = verbose

    # Utilities:
    def estimate_alpha(self, alpha: float) -> float:
        """
        Estimates a starting hyper-paramater for regularised 
        spiess given a number between 0-1.
        
        ONLY IMPLEMENTED FOR SINGLE CLASS!
        """
        demand_sum = np.sum(self.demand_matrices[0])
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
    def get_demands(self) -> list[np.ndarray]:
        """
        Returns all demand matrices (can be called before or after execution).
        """
        return self.demand_matrices

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

    # Generic Algorithm Structure:
    # Should everything that isn't top level be moved to scaling factors?
    # I.e. execute just calls another class which acts independently?
    # And then this class just has to initialise, set variables, call execute and
    # load/save results?
    # ScalingFactors can then be turned into a class which runs the actual algorithm
    # (similar to LinearApproximation) and results can just hold all the statistics
    # and final results?
    # In addition, perhaps we should have a thread per class since if they are considered
    # independent they can all be done in parallel? This may not work with all different types of
    # algorithms.
    def execute(self) -> None:
        """ 
        Run ODME algorithm until either the maximum iterations has been reached, 
        or the convergence criterion has been met.
        """
        # Initialise timing:
        self.results.init_timer()

        # Create values for SL matrices & assigned flows
        self.__perform_assignment()

        # Begin outer iteration
        outer = 0
        while outer < self.max_outer and self.last_convergence > self.outer_convergence_crit:
            # Log stats before running algorithm:
            outer += 1
            self.results.log_iter(ODMEResults.OUTER)

            # Run inner iterations:
            # INNER STOPPING CRITERION - FIND A BETTER WAY TO DO INNER STOPPING CRITERION
            self.convergence_change = float('inf') # Ensures at least 1 inner convergence is run per loop
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

    # WE COULD POTENTIALLY MOVE EVERYTHING BELOW HERE TO THE SCALINGFACTORS CLASS AND RENAME IT
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
        """
        algorithm = ScalingFactors(self, self._algorithm)
        factors = algorithm.generate()
        #self.__record_factor_stats(factors)
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
            demand_matrix = self.demand_matrices[self.names_to_indices[row['class']]]
            return np.sum(sl_matrix * demand_matrix)

        # Calculate flows for all rows:
        self.count_volumes['assign_volume'] = self.count_volumes.apply(
            lambda row: __calculate_volume(self, row),
            axis=1)
