""" Create the graph used by public transport assignment algorithms.
"""

import numpy as np
import pandas as pd


class SF_graph_builder:
    """Graph builder for the transit assignment Spiess & Florian algorithm."""

    def __init__(self, conn):
        self.conn = conn  # sqlite connection

    def create_stop_vertices(self):
        df_stop_vertices = pd.read_sql(sql="SELECT stop_id, ST_AsText(geometry) wkt FROM stops", con=self.conn)
        df_stop_vertices["line_id"] = None
        df_stop_vertices["taz_id"] = None
        df_stop_vertices["line_seg_idx"] = np.int32(-1)
        df_stop_vertices["type"] = "stop"
        df_stop_vertices["vert_idx"] = np.arange(len(df_stop_vertices))

        return df_stop_vertices
