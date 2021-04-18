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
    """ Class for select link analysis.

    ::

        from aequilibrae.paths import SelectLink, TrafficClass

        tc = TrafficClass(graph, demand_matrix)
        sl = SelectLink([tc], procedure_id)
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

    def initialise_matrices(self):
        for c in self.classes:
            self.matrices[c.__id__] = np.zeros_like(c.matrix)

    def read_compressed_graph_correspondence(self):
        pth = os.environ.get(ENVIRON_VAR)
        path_base_dir = os.path.join(pth, "path_files", self.assignment_results.procedure_id.values[0])

        for c in self.classes:
            self.compressed_graph_correspondences[c.__id__] = pd.read_feather(
                os.path.join(path_base_dir, f"correspondence_c{c.mode}_{c.__id__}.feather")
            )

    def read_assignment_results(self):
        conn = database_connection()
        results_df = pd.read_sql(f"SELECT * FROM 'results'", conn)
        conn.close()
        self.assignment_results = results_df.loc[results_df.table_name == self.table_name]
        assert (
            len(self.assignment_results) == 1
        ), "{len(self.assignment_results)} assignment result with this table name, need exactly one"

    def run_select_link_analysis(self, link_id):
        self.select_link_id = link_id

        # initialise select link matrices
        self.initialise_matrices()

        # process iterations
        self.read_assignment_results()

        # read in compressed correspondence
        self.read_compressed_graph_correspondence()
