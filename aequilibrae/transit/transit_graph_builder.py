""" Create the graph used by public transport assignment algorithms.
"""

import numpy as np
import pandas as pd


class SF_graph_builder:
    """Graph builder for the transit assignment Spiess & Florian algorithm.

    ASSUMPIONS:
    - trips dir is always 0: opposite directions is not supported

    TODO: transform some of the filtering pandas operations to SQL queries (filter down in the SQL part).
    """

    def __init__(self, conn, start=61200, end=64800, margin=0):
        """
        start and end must be expressed in seconds starting from 00h00m00s,
        e.g. 6am is 21600.
        """
        self.conn = conn  # sqlite connection
        self.start = start - margin  # starting time of the selected time period
        self.end = end + margin  # ending time of the selected time period

        self.vertex_cols = ["vert_id", "type", "stop_id", "line_id", "line_seg_idx", "taz_id", "coord"]

        self.line_segments = None
        self.stop_vertices = None
        self.boarding_vertices = None
        self.alighting_vertices = None
        self.od_vertices = None

    def create_line_segments(self):
        # trip ids corresponding to the given time range
        sql = f"""SELECT DISTINCT trip_id FROM trips_schedule 
        WHERE arrival>={self.start} AND departure<={self.end}"""
        self.trip_ids = pd.read_sql(
            sql=sql,
            con=self.conn,
        ).trip_id.values

        # pattern ids corresponding to the given time range
        sql = f"""SELECT DISTINCT pattern_id FROM trips INNER JOIN 
        (SELECT DISTINCT trip_id FROM trips_schedule 
        WHERE departure>={self.start} AND arrival<={self.end}) selected_trips
        ON trips.trip_id = selected_trips.trip_id"""
        pattern_ids = pd.read_sql(
            sql=sql,
            con=self.conn,
        ).pattern_id.values

        # route links corresponding to the given time range
        sql = "SELECT pattern_id, seq, from_stop, to_stop FROM route_links"
        route_links = pd.read_sql(
            sql=sql,
            con=self.conn,
        )
        route_links = route_links.loc[route_links.pattern_id.isin(pattern_ids)]

        # create a line segment table
        sql = "SELECT pattern_id, longname FROM routes" ""
        routes = pd.read_sql(
            sql=sql,
            con=self.conn,
        )
        routes["line_id"] = routes["longname"] + "_" + routes["pattern_id"].astype(str)
        self.line_segments = pd.merge(route_links, routes, on="pattern_id", how="left")
        self.compute_mean_travel_time()

    def compute_mean_travel_time(self):
        # Compute the travel time for each trip segment
        tt = pd.read_sql(sql="SELECT trip_id, seq, arrival, departure FROM trips_schedule", con=self.conn)
        tt.sort_values(by=["trip_id", "seq"], ascending=True, inplace=True)
        tt["last_departure"] = tt["departure"].shift(1)
        tt["last_departure"] = tt["last_departure"].fillna(0.0)
        tt["travel_time_s"] = tt["arrival"] - tt["last_departure"]
        tt.loc[tt.seq == 0, "travel_time_s"] = 0.0
        tt.drop(["arrival", "departure", "last_departure"], axis=1, inplace=True)

        # tt.seq refers to the stop sequence index
        # Because we computed the travel time between two stops, we are now dealing
        # with a segment sequence index.
        tt = tt[tt.seq > 0]
        tt.seq -= 1

        # Merge pattern ids with trip_id
        trips = pd.read_sql(sql="SELECT trip_id, pattern_id FROM trips", con=self.conn)
        tt = pd.merge(tt, trips, on="trip_id", how="left")
        tt.drop("trip_id", axis=1, inplace=True)

        # Compute the mean travel time from the different trips corresponding to
        # a given pattern/segment couple
        tt = tt.groupby(["pattern_id", "seq"]).mean().reset_index(drop=False)
        self.line_segments = pd.merge(self.line_segments, tt, on=["pattern_id", "seq"], how="left")

    def create_stop_vertices(self):
        # select all stops
        self.stop_vertices = pd.read_sql(sql="SELECT stop_id, ST_AsText(geometry) coord FROM stops", con=self.conn)
        stops_ids = pd.concat((self.line_segments.from_stop, self.line_segments.to_stop), axis=0).unique()

        # filter stops that are used on the given time range
        self.stop_vertices = self.stop_vertices.loc[self.stop_vertices.stop_id.isin(stops_ids)]

        # add metadata
        self.stop_vertices["line_id"] = None
        self.stop_vertices["taz_id"] = None
        self.stop_vertices["line_seg_idx"] = np.nan
        self.stop_vertices["line_seg_idx"] = self.stop_vertices["line_seg_idx"].astype("Int32")
        self.stop_vertices["type"] = "stop"

    def create_boarding_vertices(self):
        self.boarding_vertices = self.line_segments[["line_id", "seq", "from_stop"]].copy(deep=True)
        self.boarding_vertices.rename(columns={"seq": "line_seg_idx", "from_stop": "stop_id"}, inplace=True)
        self.boarding_vertices.line_seg_idx = self.boarding_vertices.line_seg_idx.astype("Int32")
        self.boarding_vertices = pd.merge(
            self.boarding_vertices, self.stop_vertices[["stop_id", "coord"]], on="stop_id", how="left"
        )
        self.boarding_vertices["type"] = "boarding"
        self.boarding_vertices["taz_id"] = None

    def create_alighting_vertices(self):
        self.alighting_vertices = self.line_segments[["line_id", "seq", "to_stop"]].copy(deep=True)
        self.alighting_vertices.rename(columns={"seq": "line_seg_idx", "to_stop": "stop_id"}, inplace=True)
        self.alighting_vertices.line_seg_idx = self.alighting_vertices.line_seg_idx.astype("Int32")
        self.alighting_vertices = pd.merge(
            self.alighting_vertices, self.stop_vertices[["stop_id", "coord"]], on="stop_id", how="left"
        )
        self.alighting_vertices["type"] = "alighting"
        self.alighting_vertices["taz_id"] = None

    def create_od_vertices(self):
        pass

    def create_vertices(self):
        """Graph vertices creation as a dataframe.

        Verticealighting_verticess have the following attributes:
            - vert_idx: int
            - type (either 'stop', 'boarding', 'alighting', 'od', 'walking' or 'fictitious'): str
            - stop_id (only applies to 'stop', 'boarding' and 'alighting' vertices): int
            - line_id (only applies to 'boarding' and 'alighting' vertices): str
            - line_seg_idx (only applies to 'boarding' and 'alighting' vertices): int
            - taz_id (only applies to 'od' nodes): str
            - coord (WKT): str
        """

        self.create_line_segments()
        self.create_stop_vertices()
        self.create_boarding_vertices()
        self.create_alighting_vertices()
        # self.create_od_vertices()

        # stack the dataframes on top of each other
        self.vertices = pd.concat(
            [
                self.stop_vertices,
                self.boarding_vertices,
                self.alighting_vertices,
                # self.od_vertices
            ],
            axis=0,
        )

        # reset index and copy it to column
        self.vertices.reset_index(drop=True, inplace=True)
        self.vertices.index.name = "index"
        self.vertices["vert_id"] = self.vertices.index

        self.vertices = self.vertices[self.vertex_cols]

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

    def create_on_board_edges(self):
        pass

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
