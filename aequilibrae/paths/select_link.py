import os
from typing import List
import numpy as np
import pandas as pd
from aequilibrae.paths.graph import Graph
from aequilibrae.paths import TrafficClass
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths.results import AssignmentResults
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
        self.select_link_id_compressed = None
        self.assignment_results = None
        self.classes = []
        # class specific stuff
        self.matrices = {}
        self.compressed_graph_correspondences = {}

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

    def initialise_matrices(self) -> None:
        for c in self.classes:
            self.matrices[c.__id__] = np.zeros_like(c.matrix)

    def read_compressed_graph_correspondence(self) -> None:
        pth = os.environ.get(ENVIRON_VAR)
        path_base_dir = os.path.join(pth, "path_files", self.assignment_results.procedure_id.values[0])

        for c in self.classes:
            self.compressed_graph_correspondences[c.__id__] = pd.read_feather(
                os.path.join(path_base_dir, f"correspondence_c{c.mode}_{c.__id__}.feather")
            )

    def read_assignment_results(self) -> None:
        conn = database_connection()
        results_df = pd.read_sql(f"SELECT * FROM 'results'", conn)
        conn.close()
        self.assignment_results = results_df.loc[results_df.table_name == self.table_name]
        assert (
            len(self.assignment_results) == 1
        ), "{len(self.assignment_results)} assignment result with this table name, need exactly one"

    def calculate_demand_weights(self):
        """Each iteration of traffic assignment contributes a certain fraction to the total solutions. This method
        figures out how much so we can calculate select link matrices by weighting paths per iteration."""

        # parse assignment report. This should really happen in a separate class, but that's a TODO for the future
        ass_rep = self.assignment_results["procedure_report"].values[0]
        ass_rep = ass_rep.replace("inf", "np.inf")
        ass_rep = ass_rep.replace("nan", "np.nan")
        assignment_report = eval(ass_rep)
        assignment_report["convergence"] = eval(assignment_report["convergence"])
        assignment_report["setup"] = eval(assignment_report["setup"])

        num_iters = np.max(assignment_report["convergence"]["iteration"])

        return np.repeat(1.0 / num_iters, num_iters)

    def run_select_link_analysis(self, link_id: int) -> None:
        assert len(self.classes) > 0, "Need at least one traffic class to run select link analysis, use set_classes"

        self.select_link_id = link_id

        # initialise select link matrices to zero
        self.initialise_matrices()

        # process iterations
        self.read_assignment_results()

        # read in compressed correspondence, depends on read_assignment results for path lookup
        self.read_compressed_graph_correspondence()

        # now get weight of each iteration to weight corresponding demand.
        # FIXME (Jan 18/4/21): this is MSA only atm, needs to be implemented
        self.calculate_demand_weights()

        # now process each iteration.
        # num_iters =

        for c in self.classes:
            graph = self.compressed_graph_correspondences[c.__id__]
            graph.loc[graph["link_id"] == self.link_id]
