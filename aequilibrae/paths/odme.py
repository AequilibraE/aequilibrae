from typing import Tuple
import numpy as np
import pandas as pd

from aequilibrae import TrafficAssignment, Graph
from aequilibrae.matrix import AequilibraeMatrix # Used only for writing/reading from disk - may not be relevant

class ODME(object):
    """ODME algorithm."""

    def __init__(self, 
        assignment: TrafficAssignment,
        count_volumes: list[Tuple[Tuple, int]],
        stop_crit=(1, 10**-2),
        alg_spec=None
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
        self.init_demand_matrix = self.assignclass.matrix.matrix_view # May be unecessary
        self.demand_matrix = self.init_demand_matrix # The current demand matrix
        self._demand_dims = self.demand_matrix.shape # Matrix is n x n
        # SHOULD COPY THIS IF I WANT A COPY OF ORIGINAL IN MEMORY

        # Observed Links & Associated Volumes
        ids, values = zip(*count_volumes)
        self._obs_links = np.array(ids)  # \hat A - indexing set for all other properties of observed links
        self._obs_vals = np.array(values) # \hat v_a
        # MAY WANT TO INITIALISE THESE AS np.zeros
        self._assign_vals = np.empty(len(count_volumes)) # v_a
        self._sl_matrices = np.empty((len(count_volumes, *self._demand_dims))) # d^a/p^a - d^a rn
        # NEED TO DECIDE WHETHER TO STORE THESE AS PROPORTIONS OR NOT!!!

        # Set the select links:
        self.assignclass.set_select_links(self._get_select_links())

        # Stopping criterion
        self.max_iter = stop_crit[0]
        self.convergence_crit = stop_crit[1]

        # Not yet relevant:
        self._alg_spec = alg_spec

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

            # Reassign values at the end of each outer loop
            self._perform_assignment()

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
        sl_matrices = self.assignclass.results.select_link_od.matrix
        for i, link in enumerate(self._obs_links):
            self._sl_matrices[i, :, :] = sl_matrices[f"sl_{link[0]}_{link[1]}"]

        # Extract and store array of assigned volumes to select links
        assign_df = self.assignment.results().reset_index(drop=False).fillna(0)
        col = {1: "matrix_ab", -1: "matrix_ba", 0: "matrix_tot"}
        for i, link in enumerate(self._obs_links):
            link = self._obs_links[i]
            self._assign_vals[i] = assign_df.loc[assign_df["link_id"] == link[0], col[link[1]]].values[0]
        # ^For inner iterations need to calculate this via sum sl_matrix * demand_matrix

    def objective_func(self) -> float:
        """
        Calculates the objective function - which must be specified by the user.
        """
        # NOT CURRENTLY IMPLEMENTED
        return 0.1