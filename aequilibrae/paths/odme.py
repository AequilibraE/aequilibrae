"""
Implementation of ODME algorithms:
"""

import numpy as np
import scipy.stats as spstats
import pandas as pd

from aequilibrae import TrafficAssignment

class ODME(object):
    """ODME algorithm."""
    COUNT_VOLUME_COLS = ["class", "link_id", "direction", "volume"]

    def __init__(self, 
        assignment: TrafficAssignment,
        count_volumes: pd.DataFrame, # [class, link_id, direction, volume]
        stop_crit=(1, 10**-2), # max_iterations, convergence criterion
        alg_spec=((1, 0),) # currently just the objective function specification
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
        # CHANGE COUNT VOLUMES TO A PANDAS DATAFRAME


        # Parameters for assignments
        self.assignment = assignment
        self.assignclass = assignment.classes[0] # - for now assume only one class

        # Demand matrices
        self.demand_matrix = self.assignclass.matrix.matrix_view  # The current demand matrix
        # May be unecessary - if we do keep it need to make a copy ->
        self.init_demand_matrix = np.copy(self.demand_matrix)
        self._demand_dims = self.demand_matrix.shape # Matrix is n x n

        # Observed Links & Associated Volumes
        self._count_volumes = count_volumes
        # MAY WANT TO INITIALISE THESE AS np.zeros:
        self._assign_vals = np.empty(len(count_volumes)) # v_a
        self._sl_matrices = None # Currently dictionary of proportion matrices

        # Set the select links:
        self.assignclass.set_select_links(self._get_select_links())

        # Not yet relevant - Algorithm Specifications:
        self._alg_spec = alg_spec
        self._norms = alg_spec[0]

        # Initialise objective function
        self._obj_func = None
        self._init_objective_func()

        # Stopping criterion 
        # May need to specify this further to differentiate between inner & outer criterion
        self.max_iter = stop_crit[0]
        self.convergence_crit = stop_crit[1]

        # May also want to save the last convergence value.
        # We may also want to store other variables dependent on the algorithm used,
        # e.g. the derivative of link flows w.r.t. step size.

        # Potentially set up some sort of logging information here:

    def _get_select_links(self) -> dict:
        """
        Creates dictionary of select links to be stored within the assignment class.
        Select links will be singletons for each link with associated observation value.
        """
        return {
            f"sl_{link}_{dir}": [(link, dir)] 
            for link, dir in
            zip(self._count_volumes['link_id'], self._count_volumes['direction'])
        }

    def get_result(self):
        """
        Returns current demand matrix (may be called at any point regardless 
        of whether execution has been completed). Needs to be updated to store
        actual statistics information and return this.
        
        Ideally can be called at any point while execution is ongoing - but
        this more dynamic functionality is for later on - maybe this is 
        dumb since we can reach in and grab it anyway but could be nice
        if a gui is set up to be able to get this at any point in time 
        safely. Also kind of dumb since user should still have the 
        TrafficAssignment object anyway and can access it from there.
        """
        return self.demand_matrix

    def execute(self):
        """ 
        Run ODME algorithm until either the maximum iterations has been reached, 
        or the convergence criterion has been met.
        """
        # Create values for SL matrices & assigned flows
        self._perform_assignment()

        # Begin outer iteration
        outer = 0
        while outer < self.max_iter: # OUTER STOPPING CRITERION - CURRENTLY TEMPORARY VALUE
        # while outer < self.max_iter && self._obj_func() > self.convergence_crit

            # Run inner iterations:
            inner = 0
            while inner < self.max_iter: # INNER STOPPING CRITERION
            # while inner < self.max_iter && self._obj_func() > self.inner_convergence_crit
                self._execute_inner_iter()
                inner += 1

            # Reassign values at the end of each outer loop
            self._perform_assignment()
            outer += 1

    def _execute_inner_iter(self) -> None:
        """
        Runs an inner iteration of the ODME algorithm. 
        This assumes the SL matrices stay constant and modifies
        the current demand matrix.
        """
        # Element-wise multiplication of demand matrix by scaling factor
        self.demand_matrix = self.demand_matrix * self._get_scaling_factor()

        # Recalculate the link flows
        self._calculate_flows()

    def _get_scaling_factor(self) -> np.ndarray:
        """
        Returns scaling matrix - depends on algorithm chosen.
        Currently implementing default as geometric mean.
        """
        # Defaults to geometric mean currently - cannot yet specify choice.
        return self._geometric_mean()

    def _geometric_mean(self) -> np.ndarray:
        """
        Calculates scaling factor based on geometric mean of ratio between 
        proportionally (via SL matrix) assigned flow & observed flows.

        Initial default scaling matrix:
        """
        # Steps:
        # 1. Construct SL-demand matrices d^a = g * p^a element-wise (g = demand)
        # 2. For each observed flow v_a and assigned flow w_a do (v_a - w_a) / d^a (componentwise for d^a)
        # 3. Compute geometric mean of all matrices & return
        # NOTE - This may be slower due to holding all these matrices in memory
        # simultaneously. It is possible to do this e.g element-wise or row-wise
        # to save on memory usage. Need to test this later on.
        # NOTE - by not approximating step size we may over-correct massively.

        # Steps 1 & 2:
        factors = np.nan_to_num(
            np.array(
            [
            ((self._obs_vals[i] - self._assign_vals[i]) /
             (self._sl_matrices[f"sl_{link[0]}_{link[1]}"] * self.demand_matrix)
            )
            for i, link in enumerate(self._obs_links)
            ]
        )
        )

        # Step 3:
        return spstats.gmean(factors, axis=0)


    def _perform_assignment(self) -> None:
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

        # Store reference to select link demand matrices
        self._sl_matrices = self.assignclass.results.select_link_od.matrix
        for link in self._sl_matrices:
            self._sl_matrices[link] = np.nan_to_num(self._sl_matrices[link] / self.demand_matrix)
        # NEED TO DECIDE WHETHER OR NOT TO MUTATE THESE

        # Extract and store array of assigned volumes to select links
        assign_df = self.assignment.results().reset_index(drop=False).fillna(0)
        col = {1: "matrix_ab", -1: "matrix_ba", 0: "matrix_tot"}
        # NOTE - NEED TO CHECK THAT THIS NOTATION WORKS ACROSS ALL DEMAND MATRICES!!!
        for i, link in enumerate(self._obs_links):
            self._assign_vals[i] = assign_df.loc[assign_df["link_id"] == link[0], col[link[1]]].values[0]
        # ^For inner iterations need to calculate this via sum sl_matrix * demand_matrix

    def _init_objective_func(self) -> None:
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

        def _reg_obj_func(self) -> float:
            """
            Objective function containing regularisation term.

            NOTE - NOT YET READY FOR USE! REGULARISATION TERM SHOULD BE ALPHA/BETA WEIGHTED!
            """
            obj1 = np.sum(np.abs(self._obs_vals - self._assign_vals)**p_1) / p_1
            regularisation = np.sum(np.abs(self.init_demand_matrix - self.demand_matrix)**p_2) / p_2
            return obj1 + regularisation

        def _obj_func(self) -> float:
            """
            Objective function with no regularisation term.
            """
            return np.sum(np.abs(self._obs_vals - self._assign_vals)**p_1) / p_1
        
        if p_2:
            self._obj_func = _reg_obj_func
        else:
            self._obj_func = _obj_func

    def _calculate_flows(self) -> None:
        """
        Calculates and stores link flows using current sl_matrices & demand matrix.
        """
        for i, link in enumerate(self._obs_links):
            sl_matrix = self._sl_matrices[f"sl_{link[0]}_{link[1]}"]
            self._assign_vals[i] = np.sum(sl_matrix * self.demand_matrix)