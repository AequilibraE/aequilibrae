from typing import Tuple, Callable
import numpy as np
import pandas as pd

from aequilibrae import TrafficAssignment, Graph
from aequilibrae.matrix import AequilibraeMatrix # Used only for writing/reading from disk - may not be relevant

class ODME(object):
    """ODME algorithm."""

    def __init__(self, 
        assignment: TrafficAssignment,
        count_volumes: list[Tuple[Tuple, int]],
        stop_crit=(1, 10**-2), # max_iterations, convergence criterion
        alg_spec=((1, 0)) # currently just the objective function specification
    ):
        """
        For now see description in pdf file in SMP internship team folder
        Assume for now we only have a single car graph - can be generalised later

        Parameters:
            assignment: the TrafficAssignment object - should be initialised with volume delay functions
                    and their parameters and an assignment algorithm, as well as a TrafficClass containing
                    an initial demand matrix. Doesn't need to have preset select links.
            count_volumes: the observed links and their associated observed volumes.
            stop_crit: the maximum number of iterations and the convergence criterion.
            alg_spec: NOT YET AVAILABLE - will be implemented later to allow user flexibility on what sort 
                    of algorithm they choose.
        """
        # Parameters for assignments
        self.assignment = assignment
        self.assignclass = assignment.classes[0] # - for now assume only one class

        # Initial demand matrix
        self.init_demand_matrix = self.assignclass.matrix.matrix_view # May be unecessary - if we do keep it need to make a copy
        self.demand_matrix = self.init_demand_matrix # The current demand matrix
        self._demand_dims = self.demand_matrix.shape # Matrix is n x n
        # SHOULD COPY THIS IF I WANT A COPY OF ORIGINAL IN MEMORY

        # Observed Links & Associated Volumes
        ids, values = zip(*count_volumes)
        self._obs_links = np.array(ids)  # \hat A - indexing set for all other properties of observed links
        self._obs_vals = np.array(values) # \hat v_a
        # MAY WANT TO INITIALISE THESE AS np.zeros
        self._assign_vals = np.empty(len(count_volumes)) # v_a
        self._sl_matrices = None # Currently dictionary of proportion matrices
        # NEED TO DECIDE WHETHER TO STORE THESE AS PROPORTIONS OR NOT!!!

        # Set the select links:
        self.assignclass.set_select_links(self._get_select_links())

        # Not yet relevant - Algorithm Specifications:
        self._alg_spec = alg_spec
        self._norms = alg_spec[0]

        # Initialise objective function
        self._obj_func = None
        self._set_objective_func()

        # Stopping criterion
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
        select_links = dict()
        for link in self._obs_links:
            select_links[f"sl_{link[0]}_{link[1]}"] = [link]

        return select_links

    def execute(self):
        """ 
        Run ODME algorithm until either the maximum iterations has been reached, 
        or the convergence criterion has been met.

        Returns the modfied demand matrix.
        """
        # Create values for SL matrices & assigned flows
        self._perform_assignment()

        # Begin outer iteration
        i = 0
        while i < self.max_iter: # OUTER STOPPING CRITERION - CURRENTLY TEMPORARY VALUE

            # Run inner iterations:
            j = 0
            while j < self.max_iter: # INNER STOPPING CRITERION
                self._execute_inner_iter()
                j += 1

            # Reassign values at the end of each outer loop
            self._perform_assignment()
            i += 1

        return self.demand_matrix
    
    def _execute_inner_iter(self) -> None:
        """
        Runs an inner iteration of the ODME algorithm. 
        This assumes the SL matrices stay constant and modifies
        the current demand matrix.
        """
        # Get scaling matrix
        scaling_matrix = self._get_scaling_factor()

        # Element-wise multiplication of demand matrix by scaling factor
        self.demand_matrix = self.demand_matrix * scaling_matrix

    def _get_scaling_factor(self) -> np.ndarray:
        """
        Returns scaling matrix - depends on algorithm chosen.
        Initially implement basic method.
        """
        # NOT YET IMPLEMENTED
        return np.ones(self._demand_dims)

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
        for i, link in enumerate(self._obs_links):
            self._assign_vals[i] = assign_df.loc[assign_df["link_id"] == link[0], col[link[1]]].values[0]
        # ^For inner iterations need to calculate this via sum sl_matrix * demand_matrix

    def _set_objective_func(self) -> None:
        """
        Determines the objective function - parameters must be specified by user.

        Current objective functions have 2 parts which are summed:
            1. The p-norm raised to the power p of the error vector for observed flows.
            2. The p-norm raised to the power p of the error matrix (treated as a n^2 vector) for the demand matrix.
        
        (1.) must always be present, but (2.) (the regularisation term) need not be present (ie, specified as 0 by user).
        Default currently set to l1 (manhattan) norm for (1.) and no regularisation term (input 0).
        """
        p1 = self._norms[0]
        p2 = self._norms[1]

        def reg_obj_func(self) -> float:
            """
            Objective function containing regularisation term.
            """
            obj1 = np.sum(np.abs(self._obs_vals - self._assign_vals)**p1) / p1 
            regularisation = np.sum(np.abs(self.init_demand_matrix - self.demand_matrix)**p2) / p2
            return obj1 + regularisation

        def obj_func(self) -> float:
            """
            Objective function with no regularisation term.
            """
            return np.sum(np.abs(self._obs_vals - self._assign_vals)**p1) / p1
        
        if p2:
            self._obj_func = reg_obj_func
        else:
            self._obj_func = obj_func
        

    def manhattan(self) -> float:
        """
        l1 (Manhattan) distance function - only based on flows.
        """
        return np.sum(np.abs(self._obs_vals - self._assign_vals))

    def _calculate_flows(self) -> None:
        """
        Calculates and stores link flows using current sl_matrices & demand matrix.
        """
        for i, link in enumerate(self._obs_links):
            sl_matrix = self._sl_matrices[f"sl_{link[0]}_{link[1]}"]
            self._assign_vals[i] = sum(sl_matrix * self.demand_matrix)