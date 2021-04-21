import os
from collections import defaultdict
from typing import List
import numpy as np
import pandas as pd
from aequilibrae.paths.traffic_class import TrafficClass
from aequilibrae import logger
from aequilibrae.project.database_connection import ENVIRON_VAR
from aequilibrae.project.database_connection import database_connection


class SelectLink(object):
    """ Class for select link analysis. Depends on traffic assignment results with path files saved to disk.

    ::

        from aequilibrae.paths import SelectLink, TrafficClass

        tc = TrafficClass(graph, demand_matrix)
        sl = SelectLink(table_name_with_assignment_results)
        sl.set_classes([tc])
        link_id_for_sl = 111
        sl.run_select_link_analysis(link_id_for_sl)
    """

    def __init__(self, table_name: str) -> None:
        """
        Instantiates the class

         Args:
            table_name (str): Name of the traffic assignment result table used to generate the required path files
        """
        self.table_name = table_name
        self.select_link_id = None
        # self.select_link_id_compressed = None
        self.assignment_results = None
        self.demand_weights = None
        self.num_iters = None
        self.classes = []
        # class specific stuff
        self.matrices = {}
        self.compressed_graph_correspondences = {}
        self.select_link_id_compressed = {}

        # for critical link analyis
        self.critical_link_dict = None

    def set_classes(self, classes: List[TrafficClass]) -> None:
        """
        Sets Traffic classes to be used for Select Link Analysis

        Args:
            classes (:obj:`List[TrafficClass]`:) List of Traffic classes for assignment
        """
        ids = set([x.__id__ for x in classes])
        if len(ids) < len(classes):
            raise Exception("Classes need to be unique. Your list of classes has repeated items/IDs")
        self.classes = classes  # type: List[TrafficClass]

    def _initialise_matrices(self) -> None:
        for c in self.classes:
            self.matrices[c.__id__] = np.zeros_like(c.matrix.matrix_view)

    def _read_compressed_graph_correspondence(self) -> None:
        pth = os.environ.get(ENVIRON_VAR)
        path_base_dir = os.path.join(pth, "path_files", self.assignment_results.procedure_id.values[0])

        for c in self.classes:
            self.compressed_graph_correspondences[c.__id__] = pd.read_feather(
                os.path.join(path_base_dir, f"correspondence_c{c.mode}_{c.__id__}.feather")
            )

    def _read_assignment_results(self) -> None:
        conn = database_connection()
        results_df = pd.read_sql("SELECT * FROM 'results'", conn)
        conn.close()
        self.assignment_results = results_df.loc[results_df.table_name == self.table_name]
        assert (
            len(self.assignment_results) == 1
        ), f"Found {len(self.assignment_results)} assignment result with this table name, need exactly one"

    def _calculate_demand_weights(self):
        """Each iteration of traffic assignment contributes a certain fraction to the total solutions. This method
        figures out how much so we can calculate select link matrices by weighting paths per iteration."""

        # parse assignment report. This should really happen in a separate class, but that's a TODO for the future
        ass_rep = self.assignment_results["procedure_report"].values[0]
        ass_rep = ass_rep.replace("inf", "np.inf")
        ass_rep = ass_rep.replace("nan", "np.nan")
        assignment_report = eval(ass_rep)
        assignment_report["convergence"] = eval(assignment_report["convergence"])
        assignment_report["setup"] = eval(assignment_report["setup"])

        self.num_iters = np.max(assignment_report["convergence"]["iteration"])

        assignment_method = assignment_report["setup"]["Algorithm"]

        if assignment_method == "msa":
            self.demand_weights = np.repeat(1.0 / self.num_iters, self.num_iters)
        elif assignment_method in ["fw", "cfw", "bfw", "frank-wolfe"]:
            self._figure_out_demand_weights_for_linear_approximation(assignment_report)
        else:
            raise ValueError(
                f"Asignment method {assignment_method} cannot be used for select link analysis at the moment."
            )

        sum_of_contribs = np.sum(self.demand_weights)
        assert np.allclose(sum_of_contribs, 1.0), f"Contribution of iterations is not one, but {sum_of_contribs}"

    def _figure_out_demand_weights_for_linear_approximation(self, assignment_report):
        """ Linear approximation contribution for each iteration. """

        # solution^n+1 = alpha^n * sol^n + (1-alpha^n) * direction^n
        # direction = beta_0 * aon + beta_1 * previous_direction + beta_2 * pre_previous_direction
        alphas = assignment_report["convergence"]["alpha"]
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

    def _read_path_file(self, iteration, traffic_class, origin):

        iteration += 1  # aequilibrae's iterations are 1-based, but I used numpy's 0-based indexing in select_link

        pth = os.environ.get(ENVIRON_VAR)
        path_base_dir = os.path.join(pth, "path_files", self.assignment_results.procedure_id.values[0])
        base_dir = os.path.join(path_base_dir, f"iter{iteration}", f"path_c{traffic_class.mode}_{traffic_class.__id__}")

        path_o_f = os.path.join(base_dir, f"o{origin}.feather")
        path_o_index_f = os.path.join(base_dir, f"o{origin}_indexdata.feather")

        path_o = pd.read_feather(path_o_f)
        path_o_index = pd.read_feather(path_o_index_f)

        return path_o, path_o_index

    def get_path_for_destination(self, path_o, path_o_index, destination):
        """ Return all link ids, i.e. the full path, for a given destination """
        if destination == 0:
            lower_incl = 0
        else:
            lower_incl = path_o_index.loc[path_o_index.index == destination - 1].values[0][0]

        upper_non_incl = path_o_index.loc[path_o_index.index == destination].values[0][0]
        links_on_path = path_o.loc[(path_o.index >= lower_incl) & (path_o.index < upper_non_incl)].values.flatten()
        return links_on_path

    def run_select_link_analysis(self, link_id: int, do_critical_link_analysis=False) -> None:
        assert len(self.classes) > 0, "Need at least one traffic class to run select link analysis, use set_classes"

        self.select_link_id = link_id

        # initialise select link matrices to zero
        self._initialise_matrices()

        # process iterations
        self._read_assignment_results()

        # read in compressed correspondence, depends on read_assignment results for path lookup
        self._read_compressed_graph_correspondence()

        # look up compressed id for each class. Should be the same.
        for c in self.classes:
            graph = self.compressed_graph_correspondences[c.__id__]
            self.select_link_id_compressed[c.__id__] = graph.loc[graph["link_id"] == self.select_link_id][
                "__compressed_id__"
            ].values[0]

        # now get weight of each iteration to weight corresponding demand.
        # FIXME (Jan 21/4/21): this is MSA and FW only atm, needs to be implemented for CFW and BFW
        self._calculate_demand_weights()

        if do_critical_link_analysis:
            self.critical_link_dict = defaultdict(float)

        # now process each iteration.
        num_centroids = self.classes[0].matrix.zones

        for iteration in range(self.num_iters):
            logger.info(f"Procesing iteration {iteration} for select link analysis")
            weight = self.demand_weights[iteration]
            for c in self.classes:
                comp_link_id = self.select_link_id_compressed[c.__id__]
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

                    if do_critical_link_analysis:
                        for destination in destinations_this_o_and_iter:
                            links_on_path = self.get_path_for_destination(path_o, path_o_index, destination)
                            # each of these links will carry the weight of this iteration, need a map with default
                            # zero insertion and insert combining values -> defaultdict, then add. This will all have
                            # to be in C++ I think, it looks dog slow
                            for l_ in links_on_path:
                                self.critical_link_dict[l_] += weight * demand_mat[origin, destination]
