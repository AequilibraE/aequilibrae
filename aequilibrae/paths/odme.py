"""
Implementation of ODME algorithms:
"""

# NOTE - Until issue with select link flows not matching assigned flows ODME should not be used with biconjugate/conjugate frank-wolfe

from typing import Tuple
import time
import numpy as np
import scipy.stats as spstats
import pandas as pd

from aequilibrae import TrafficAssignment

class ODME(object):
    """ODME algorithm."""
    COUNT_VOLUME_COLS = ["class", "link_id", "direction", "obs_volume"]
    STATISTICS_COLS = ["Outer Loop #", "Inner Loop #", "Convergence", "Inner Convergence", "Time (s)"]

    def __init__(self,
        assignment: TrafficAssignment,
        count_volumes: pd.DataFrame, # [class, link_id, direction, volume]
        stop_crit=(5, 10**-2), # max_iterations, convergence criterion
        alg_spec=((2, 0),) # currently just the objective function specification
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
        # ENSURE ORDERING (PERHAPS BY SORTING INITIALLY) IS MAINTAINED EVERYWHERE
        # POTENTIALLY MIGHT BE A GOOD IDEA TO ADD ASSIGNED VOLUMES AS A COLUMN TO COUNT_VOLUMES DATAFRAME

        # Parameters for assignments
        self.assignment = assignment
        self.assignclass = assignment.classes[0] # - for now assume only one class

        # Demand matrices
        self.demand_matrix = self.assignclass.matrix.matrix_view  # The current demand matrix
        # May be unecessary - if we do keep it need to make a copy -> 
        # MAYBE PUT THIS IN AN IF STATEMENT AND ONLY COPY IF A REGULARISATION TERM IS SPECIFIED
        self.init_demand_matrix = np.copy(self.demand_matrix)
        self._demand_dims = self.demand_matrix.shape # Matrix is n x n

        # Observed Links & Associated Volumes
        self._count_volumes = count_volumes
        # MAY WANT TO INITIALISE THESE AS np.zeros:
        self._assign_vals = np.empty(len(count_volumes)) # v_a
        self._sl_matrices = None # Currently dictionary of proportion matrices

        # Set the select links:
        self.assignclass.set_select_links(self.__get_select_links())

        # Not yet relevant - Algorithm Specifications:
        self._alg_spec = alg_spec
        self._norms = alg_spec[0]

        # Initialise objective function
        self._obj_func = None
        self.__init_objective_func()
        self._last_convergence = None
        self._convergence_change = float('inf')

        # Stopping criterion
        # May need to specify this further to differentiate between inner & outer criterion
        self.max_iter = stop_crit[0]
        self.convergence_crit = stop_crit[1]

        self._outer, self._inner = 0, 0

        # May also want to save the last convergence value.
        # We may also want to store other variables dependent on the algorithm used,
        # e.g. the derivative of link flows w.r.t. step size.

        # Potentially set up some sort of logging information here:

        # Dataframe to log statistical information:
        self._statistics = pd.DataFrame(columns=self.STATISTICS_COLS)

        # Time data for logging information
        self._time = None


    def __get_select_links(self) -> dict:
        """
        Creates dictionary of select links to be stored within the assignment class.
        Select links will be singletons for each link with associated observation value.
        """
        return {
            f"sl_{link}_{dir}": [(link, dir)] 
            for link, dir in
            zip(self._count_volumes['link_id'], self._count_volumes['direction'])
        }

    def get_results(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Returns final demand matrix and a dataframe of statistics regarding
        timing and convergence
        """
        return (self.demand_matrix, self._statistics)

    def __log_stats(self) -> None:
        """
        Adds next row to statistics dataframe.
        """
        old_time = self._time
        self._time = time.time()
        to_log = [self._outer, self._inner, self._last_convergence, self._convergence_change, self._time - old_time]

        # Add row::
        self._statistics.loc[len(self._statistics)] = {
            col : to_log[i]
            for i, col in enumerate(self.STATISTICS_COLS)
        }

    def execute(self):
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
        while self._outer < self.max_iter and self._last_convergence > self.convergence_crit:
            # Set iteration values:
            self._outer += 1
            self._inner = 0
            self.__log_stats()

            # Run inner iterations:
            # INNER STOPPING CRITERION - FIND A BETTER WAY TO DO INNER STOPPING CRITERION
            # MAYBE BASED ON DIFFERENCE IN CONVERGENCE
            self._convergence_change = float('inf')
            while self._inner < self.max_iter and self._convergence_change > self.convergence_crit:
                self.__execute_inner_iter()
                self._inner += 1
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
        # Defaults to geometric mean currently - cannot yet specify choice.
        return self.__geometric_mean()

    def __geometric_mean(self) -> np.ndarray:
        """
        Calculates scaling factor based on geometric mean of ratio between 
        proportionally (via SL matrix) assigned flow & observed flows.

        Initial default scaling matrix:
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
            # Create factor:
            if row["obs_volume"] != 0 and self._assign_vals[i] != 0:
                link_factor = row['obs_volume'] / self._assign_vals[i]
                sl_matrix = self._sl_matrices[f"sl_{row['link_id']}_{row['direction']}"]
                factors[i, :, :] = np.where(sl_matrix == 0, 1, link_factor)
            # If assigned or observed value is 0 we cannot do anything right now
            else:
                factors[i, :, :] = np.ones(self._demand_dims)

        # If the assigned volume was 0 (or both 0) no OD pair can have any effect
        factors = np.nan_to_num(factors, nan=1, posinf=1, neginf=1)

        # Step 3:
        return spstats.gmean(factors, axis=0)

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
        # NOTE - NEED TO CHECK THAT THIS NOTATION WORKS ACROSS ALL DEMAND MATRICES!!!
        # FIND FASTER VECTORISED WAY TO DO THIS!
        for i, row in self._count_volumes.iterrows():
            self._assign_vals[i] = assign_df.loc[
                assign_df["link_id"] == row["link_id"],
                 col[row["direction"]]
            ].values[0]
        # ^For inner iterations need to calculate this via sum sl_matrix * demand_matrix

        # Recalculate convergence values
        self._obj_func(self)

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
            obj1 = np.sum(np.abs(obs_vals - self._assign_vals)**p_1) / p_1
            regularisation = np.sum(np.abs(self.init_demand_matrix - self.demand_matrix)**p_2) / p_2
            self.__set_convergence_values(obj1 + regularisation)

        def __obj_func(self) -> None:
            """
            Objective function with no regularisation term.
            """
            obs_vals = self._count_volumes["obs_volume"].to_numpy()
            self.__set_convergence_values(np.sum(np.abs(obs_vals - self._assign_vals)**p_1) / p_1)

        if p_2:
            self._obj_func = __reg_obj_func
        else:
            self._obj_func = __obj_func

    def __set_convergence_values(self, new_convergence: float) -> None:
        """
        Given a new convergence value calculates the difference between the previous convergence
        and new convergence, and sets appropriate values.
        """
        if self._last_convergence:
            self._convergence_change = abs(self._last_convergence - new_convergence)
        self._last_convergence = new_convergence 

    def __calculate_flows(self) -> None:
        """
        Calculates and stores link flows using current sl_matrices & demand matrix.
        """
        for i, row in self._count_volumes.iterrows():
            sl_matrix = self._sl_matrices[f"sl_{row['link_id']}_{row['direction']}"]
            self._assign_vals[i] = np.sum(sl_matrix * self.demand_matrix)
