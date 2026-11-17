import numpy as np
import pandas as pd

from aequilibrae._assignment import AssignmentResultsBase


class TransitAssignmentResults(AssignmentResultsBase):
    """Transit loads are owned by the hyperpath computation."""

    def __init__(self):
        super().__init__()
        self.link_loads = np.array([])
        self.skims = None

    def prepare(self, graph, matrix):
        self.reset()
        self.nodes = graph.num_nodes
        self.zones = graph.num_zones
        self.centroids = graph.centroids
        self.links = graph.num_links
        self.lids = graph.graph.link_id.to_numpy(copy=False)

    def reset(self):
        # Loads may borrow a read-only pandas buffer; leave retained results untouched.
        self.link_loads = np.zeros_like(self.link_loads)
        self.skims = None

    def get_load_results(self):
        if not self.link_loads.shape[0]:
            raise ValueError("Transit assignment has not been executed yet")
        return pd.DataFrame({"volume": self.link_loads}, index=self.lids)
