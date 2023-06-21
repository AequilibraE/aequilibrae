""" Create the graph used by public transport assignment algorithms.
"""

import numpy as np
import pandas as pd


class SF_graph_builder:
    """Graph builder for the transit assignment Spiess & Florian algorithm."""

    def __init__(self, conn):
        self.conn = conn  # sqlite connection

    def create_stop_vertices(self):
        df_stop_vertices = pd.read_sql(sql="SELECT stop_id, ST_AsText(geometry) coord FROM stops", con=self.conn)
        df_stop_vertices["line_id"] = None
        df_stop_vertices["taz_id"] = None
        df_stop_vertices["line_seg_idx"] = np.int32(-1)
        df_stop_vertices["type"] = "stop"
        df_stop_vertices["vert_idx"] = np.arange(len(df_stop_vertices))

        return df_stop_vertices

    def create_boarding_vertices(self):
        pass

    def create_alighting_vertices(self):
        pass

    def create_od_vertices(self):
        pass

    def create_vertices(self):
        """Graph vertices creation as a dataframe.

        Vertices have the following attributes:
            - type (either 'stop', 'boarding', 'alighting', 'od', 'walking' or 'fictitious'): str
            - coord (WKT): str
            - stop_id (only applies to 'stop', 'boarding' and 'alighting' vertices): str
            - line_id (only applies to 'boarding' and 'alighting' vertices): str
            - line_seg_idx (only applies to 'boarding' and 'alighting' vertices): int
            - taz_id (only applies to 'od' nodes): str

        Note that line_seg_idx is using a 1-based indexing.

        """

        df_stop_vertices = self.create_stop_vertices()
        # df_boarding_vertices = self.create_boarding_vertices()
        # df_alighting_vertices = self.create_alighting_vertices()
        # df_od_vertices = self.create_od_vertices()
        # Create ids

    def create_on_board_edges(self):
        pass

    def create_boarding_edges(self):
        pass

    def create_alighting_edges(self):
        pass

    def create_dwell_edges(self):
        pass

    def create_connector_edges(self):
        pass

    def create_inner_stop_transfer_edges(self):
        pass

    def create_outer_stop_transfer_edges(self):
        pass

    def create_transfer_edges(self):
        df_inner_stop_transfer_edges = self.create_inner_stop_transfer_edges()
        df_outer_stop_transfer_edges = self.create_outer_stop_transfer_edges()
        pass

    def create_walking_edges(self):
        pass

    def __create_edges(self):
        """Graph edges creation as a dataframe.

        Edges have the following attributes:
            - type (either 'on-board', 'boarding', 'alighting', 'dwell', 'transfer', 'connector' or 'walking'): str
            - line_id (only applies to 'on-board', 'boarding', 'alighting' and 'dwell' edges): str
            - stop_id: str
            - line_seg_idx (only applies to 'on-board', 'boarding' and 'alighting' edges): int
            - tail_vert_idx: int
            - head_vert_idx: int
            - trav_time (edge travel time): float
            - freq (frequency): float
            - o_line_id: str
            - d_line_id: str
            - transfer_id: str

        """
