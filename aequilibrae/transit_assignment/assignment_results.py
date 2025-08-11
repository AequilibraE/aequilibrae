import numpy as np
import pandas as pd

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths.assignment_results import AssignmentResultsBase
from aequilibrae.transit_assignment.transit_graph import TransitGraph


class TransitAssignmentResults(AssignmentResultsBase):
    """
    Assignment result holder for a single :obj:`Transit`
    """

    def __init__(self):
        super().__init__()

        self.link_loads = np.array([])

    def prepare(self, graph: TransitGraph, matrix: AequilibraeMatrix) -> None:
        """
        Prepares the object with dimensions corresponding to the assignment matrix and graph objects

        :Arguments:
            **graph** (:obj:`TransitGraph`): Needs to have been set with number of centroids

            **matrix** (:obj:`AequilibraeMatrix`): Matrix properly set for computation with
            ``matrix.computational_view(:obj:`list`)``
        """
        self.reset()
        self.nodes = graph.num_nodes
        self.zones = graph.num_zones
        self.centroids = graph.centroids
        self.links = graph.num_links
        self.lids = graph.graph.link_id.values

    def reset(self) -> None:
        """
        Resets object to prepared and pre-computation state
        """

        # Since all memory for the assignment is managed by the HyperpathGenerating
        # object we don't need to do much here
        self.link_loads.fill(0)

    def get_load_results(self) -> pd.DataFrame:
        """
        Translates the assignment results from the graph format into the network format

        :Returns:
            **dataset** (:obj:`pd.DataFrame`): DataFrame data with the transit class assignment results
        """
        if not self.link_loads.shape[0]:
            raise ValueError("Transit assignment has not been executed yet")

        return pd.DataFrame({"volume": self.link_loads}, index=self.lids)
