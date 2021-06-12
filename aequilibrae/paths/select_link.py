import os
from collections import defaultdict
from typing import Dict
import numpy as np
import pandas as pd

from aequilibrae.paths.assignment_paths import AssignmentPaths, TrafficClassIdentifier
from aequilibrae import logger


class SelectLink(object):
    """ Class for select link analysis. Depends on traffic assignment results with path files saved to disk.
    ::
        from aequilibrae.paths import SelectLink
        matrices = {class_id: matrix}
        name = 'assignment_results_table_name"
        sl = SelectLink(name, matrices)
        link_ids_for_sl = [111]
        sl.run_select_link_analysis(link_id_for_sl)
    """

    def __init__(self, table_name: str, demand_matrices: Dict) -> None:
        """
        Instantiates the class
         Args:
            table_name (str): Name of the traffic assignment result table used to generate the required path files
            demand_matrices (dict): Dict with assignment class id and corresponding demand matrix
        """
        self.table_name = table_name
        self.select_link_id = None
        self.assignment_results = None

        self.paths = AssignmentPaths(self.table_name)
        self.num_iters = self.paths.assignment_results.get_number_of_iterations()

        # TODO: assert class ids and matrix keys are identical in the following
        self.classes = self.paths.get_traffic_class_names_and_id()
        self.demand_matrices = demand_matrices

        self.select_link_matrices = {}
        self._initialise_matrices()

        # get weight of each iteration to weight corresponding demand.
        # FIXME (Jan 21/4/21): this is MSA and FW only atm, needs to be implemented for CFW and BFW
        self.demand_weights = None
        self._calculate_demand_weights()

    def _initialise_matrices(self) -> None:
        for c in self.classes:
            self.select_link_matrices[c.id] = np.zeros_like(self.demand_matrices[c.id].matrix_view)

    def _calculate_demand_weights(self):
        """Each iteration of traffic assignment contributes a certain fraction to the total solutions. This method
        figures out how much so we can calculate select link matrices by weighting paths per iteration."""

        assignment_method = self.paths.assignment_results.get_assignment_method()

        if assignment_method == "msa":
            self.demand_weights = np.repeat(1.0 / self.num_iters, self.num_iters)
        elif assignment_method in ["fw", "cfw", "bfw", "frank-wolfe"]:
            self._figure_out_demand_weights_for_linear_approximation()
        else:
            raise ValueError(
                f"Asignment method {assignment_method} cannot be used for select link analysis at the moment."
            )

        sum_of_contribs = np.sum(self.demand_weights)
        assert np.allclose(sum_of_contribs, 1.0), f"Contribution of iterations is not one, but {sum_of_contribs}"

    def _figure_out_demand_weights_for_linear_approximation(self):
        """ Linear approximation contribution for each iteration. """

        # solution^n+1 = alpha^n * sol^n + (1-alpha^n) * direction^n
        # direction = beta_0 * aon + beta_1 * previous_direction + beta_2 * pre_previous_direction
        alphas = self.paths.assignment_results.get_alphas()
        print(alphas)
        # betas_0 = assignment_report["convergence"]["beta0"]
        # betas_1 = assignment_report["convergence"]["beta1"]
        # betas_2 = assignment_report["convergence"]["beta2"]

        self.demand_weights = np.repeat(1.0, self.num_iters)

        # FIXME (Jan 21/4/21): implement CFW and BFW, this is only valid for FW
        # can we just multiply previous like below?
        # if assignment_report["setup"]["Algorithm"] in ["fw", "frank-wolfe"]:
        for i in range(0, self.num_iters):
            alpha = alphas[i]
            # beta_0 = betas_0[i]
            # beta_1 = betas_1[i]
            # beta_2 = betas_2[i]
            # demand_weights[i] += alpha * direction + (1.0 - alpha) * current
            # for FW: direction = AON, so multiply current weight
            # for CFW: direction = beta_0 * AON + beta_1 * previous_direction
            # for BFW: direction = beta_0 * AON + beta_1 * previous_direction + beta_2 * pre_previous_direction
            self.demand_weights[i] *= alpha
            self.demand_weights[:i] *= 1.0 - alpha

            # if (assignment_report["setup"]["Algorithm"] in ["cfw", "bfw"]) and (i > 1):
            #     self.demand_weights[i] *= beta_0
            #     self.demand_weights[:(i - 1)] *= beta_1
            #
            # if (assignment_report["setup"]["Algorithm"] == "bfw") and (i > 2):
            #     self.demand_weights[:(i-2)] *= beta_2

    # FIXME: pass in a list of links
    def run_select_link_analysis(self, link_id: int) -> None:
        assert len(self.classes) > 0, "Need at least one traffic class to run select link analysis, use set_classes"

        # FIXME
        self.select_link_id = link_id
        # look up compressed id for each class. Should be the same.
        for c in self.classes:
            graph = self.compressed_graph_correspondences[c.__id__]
            self.select_link_id_compressed[c.__id__] = graph.loc[graph["link_id"] == self.select_link_id][
                "__compressed_id__"
            ].values[0]
        # end FIXME

        # now process each iteration.
        num_centroids = self.classes[0].matrix.zones  # FIXME no TrafficClasses anymore

        for iteration in range(self.num_iters):
            logger.info(f"Procesing iteration {iteration} for select link analysis")
            weight = self.demand_weights[iteration]
            for c in self.classes:
                comp_link_id = self.select_link_id_compressed[c.id]
                demand_mat = c.matrix.matrix_view

                for origin in range(num_centroids):
                    path_o, path_o_index = self._read_path_file(iteration, c, origin)

                    # these are the indeces of the path file where the SL appears, so need to turn these into
                    # destinations by looking up the values in the path file
                    idx_to_look_up = path_o.loc[path_o.data == comp_link_id].index.values

                    # drop disconnected zones (and intrazonal). Depends on index being ordered.
                    path_o_index_no_zeros = path_o_index.drop_duplicates(keep="first")
                    destinations_this_o_and_iter = np.array(
                        [
                            path_o_index_no_zeros.loc[path_o_index_no_zeros["data"] >= x].index.min()
                            for x in idx_to_look_up
                        ]
                    )
                    destinations_this_o_and_iter = destinations_this_o_and_iter.astype(int)

                    self.matrices[c.__id__][origin, destinations_this_o_and_iter] += (
                        weight * demand_mat[origin, destinations_this_o_and_iter]
                    )
