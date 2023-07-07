""" Create the graph used by public transport assignment algorithms.
"""

import numpy as np
import pandas as pd
import shapely
import shapely.ops
import pyproj
from scipy.spatial import cKDTree


class SF_graph_builder:
    """Graph builder for the transit assignment Spiess & Florian algorithm.

    ASSUMPIONS:
    - trips dir is always 0: opposite directions are not supported
    - all times are expressed in seconds [s], all frequencies in [1/s]
    - headways are uniform for trips of the same pattern

    TODO:
    - transform some of the filtering pandas operations to SQL queries (filter down in the SQL part).
    - instanciate properly using a project path, an aequilibrae project or anything else that follow the
      package guideline (without explicit public_transport_conn and project_conn)
    """

    def __init__(
        self,
        public_transport_conn,
        project_conn,
        start=61200,
        end=64800,
        margin=0,
        global_crs="EPSG:4326",
        projected_crs="EPSG:2154",
        num_threads=-1,
    ):
        """
        start and end must be expressed in seconds starting from 00h00m00s,
        e.g. 6am is 21600.
        """
        self.pt_conn = public_transport_conn  # sqlite connection
        self.pt_conn.enable_load_extension(True)
        self.pt_conn.load_extension("mod_spatialite")

        self.proj_conn = project_conn  # sqlite connection
        self.proj_conn.enable_load_extension(True)
        self.proj_conn.load_extension("mod_spatialite")

        self.start = start - margin  # starting time of the selected time period
        self.end = end + margin  # ending time of the selected time period
        self.num_threads = num_threads

        self.vertex_cols = ["vert_id", "type", "stop_id", "line_id", "line_seg_idx", "taz_id", "coord"]
        self.edges_cols = [
            "type",
            "line_id",
            "stop_id",
            "line_seg_idx",
            "tail_vert_id",
            "head_vert_id",
            "trav_time",
            "freq",
            "o_line_id",
            "d_line_id",
            "transfer_id",
        ]

        self.line_segments = None
        self.stop_vertices = None
        self.boarding_vertices = None
        self.alighting_vertices = None
        self.od_vertices = None
        self.on_board_edges = None
        self.dell_edges = None
        self.alighting_edges = None
        self.boarding_edges = None

        self.global_crs = global_crs
        self.projected_crs = projected_crs

        # edge weight parameters
        self.uniform_dwell_time = 30
        self.alighting_penalty = 480
        self.a_tiny_time_duration = 1.0e-08
        self.wait_time_factor = 2.0
        self.walking_speed = 1.0
        self.access_time_factor = 1.0
        self.egress_time_factor = 1.0

    def create_line_segments(self):
        # trip ids corresponding to the given time range
        sql = f"""SELECT DISTINCT trip_id FROM trips_schedule 
        WHERE departure>={self.start} AND arrival<={self.end}"""
        self.trip_ids = pd.read_sql(
            sql,
            self.pt_conn,
        ).trip_id.values

        # pattern ids corresponding to the given time range
        sql = f"""SELECT DISTINCT pattern_id FROM trips INNER JOIN 
        (SELECT DISTINCT trip_id FROM trips_schedule 
        WHERE departure>={self.start} AND arrival<={self.end}) selected_trips
        ON trips.trip_id = selected_trips.trip_id"""
        pattern_ids = pd.read_sql(
            sql,
            self.pt_conn,
        ).pattern_id.values

        # route links corresponding to the given time range
        sql = "SELECT pattern_id, seq, from_stop, to_stop FROM route_links"
        route_links = pd.read_sql(
            sql=sql,
            con=self.pt_conn,
        )
        route_links = route_links.loc[route_links.pattern_id.isin(pattern_ids)]

        # create a line segment table
        sql = "SELECT pattern_id, longname FROM routes" ""
        routes = pd.read_sql(
            sql=sql,
            con=self.pt_conn,
        )
        routes["line_id"] = routes["longname"] + "_" + routes["pattern_id"].astype(str)
        self.line_segments = pd.merge(route_links, routes, on="pattern_id", how="left")

        self.add_mean_travel_time_to_segments()
        self.add_mean_headway_to_segments()

        # compute the frequency
        self.line_segments["freq"] = np.inf
        self.line_segments.loc[self.line_segments.headway > 0.0, "freq"] = (
            1.0 / self.line_segments.loc[self.line_segments.headway > 0.0, "headway"]
        )

    def compute_segment_travel_time(self, time_filter=True):
        if time_filter:
            sql = f"""SELECT trips_schedule.trip_id, trips_schedule.seq, trips_schedule.arrival, 
                trips_schedule.departure, trips.pattern_id FROM trips_schedule LEFT JOIN trips
                ON trips_schedule.trip_id = trips.trip_id
                WHERE trips_schedule.departure>={self.start} AND trips_schedule.arrival<={self.end}"""
        else:
            sql = f"""SELECT trips_schedule.trip_id, trips_schedule.seq, trips_schedule.arrival, 
                trips_schedule.departure, trips.pattern_id FROM trips_schedule LEFT JOIN trips
                ON trips_schedule.trip_id = trips.trip_id"""
        tt = pd.read_sql(sql, self.pt_conn)

        # compute the travel time on the segments
        tt.sort_values(by=["pattern_id", "trip_id", "seq"], ascending=True, inplace=True)
        tt["last_departure"] = tt["departure"].shift(+1)
        tt["last_trip_id"] = tt["trip_id"].shift(+1)
        tt["last_pattern_id"] = tt["pattern_id"].shift(+1)
        tt["trav_time"] = tt["arrival"] - tt["last_departure"]
        tt.loc[tt.seq == 0, "trav_time"] = np.nan
        tt.loc[(tt.last_pattern_id != tt.pattern_id) | (tt.last_trip_id != tt.trip_id), "trav_time"] = np.nan

        # tt.seq refers to the stop sequence index.
        # Because we computed the travel time between two stops, we are now dealing
        # with a segment sequence index.
        tt = tt.loc[tt.seq > 0]
        tt.seq -= 1

        # take the min of the travel times computed among the trips of a pattern segment
        tt = tt[["pattern_id", "seq", "trav_time"]].groupby(["pattern_id", "seq"]).mean().reset_index(drop=False)

        return tt

    def add_mean_travel_time_to_segments(self):
        tt = self.compute_segment_travel_time(time_filter=True)
        tt_full = self.compute_segment_travel_time(time_filter=False)
        tt_full.rename(columns={"trav_time": "trav_time_full"}, inplace=True)

        # Compute the mean travel time from the different trips corresponding to
        self.line_segments = pd.merge(self.line_segments, tt, on=["pattern_id", "seq"], how="left")
        self.line_segments = pd.merge(self.line_segments, tt_full, on=["pattern_id", "seq"], how="left")
        self.line_segments.trav_time = self.line_segments.trav_time.fillna(self.line_segments.trav_time_full)
        self.line_segments.drop("trav_time_full", axis=1, inplace=True)
        self.line_segments.trav_time = self.line_segments.trav_time.fillna(self.end - self.start)

    def add_mean_headway_to_segments(self):
        # start from the trip_schedule table
        sql = f"""SELECT trip_id, seq, arrival FROM trips_schedule
            WHERE departure>={self.start} AND arrival<={self.end}"""
        mh = pd.read_sql(sql, self.pt_conn)

        # merge the trips schedules with pattern ids
        trips = pd.read_sql(sql="SELECT trip_id, pattern_id FROM trips", con=self.pt_conn)
        mh = pd.merge(mh, trips, on="trip_id", how="left")
        mh.sort_values(by=["pattern_id", "seq", "trip_id", "arrival"], inplace=True)
        mh["headway"] = mh["arrival"].diff()

        # count the number of trips per stop
        trip_count = mh.groupby(["pattern_id", "seq"]).size().to_frame("trip_count")
        mh = pd.merge(mh, trip_count, on=["pattern_id", "seq"], how="left")

        # compute the trip index for a given couple pattern & stop
        trip_id_last = -1
        seq_last = -1
        pattern_id_last = -1
        trip_idx_values = np.zeros(len(mh), dtype=int)
        trip_idx = 0
        i = 0
        for row in mh.itertuples():
            trip_id = row.trip_id
            seq = row.seq
            pattern_id = row.seq
            assert (trip_id != trip_id_last) | (seq == seq_last + 1)

            if seq != seq_last:
                trip_idx = 0
            if pattern_id != pattern_id_last:
                trip_idx = 0

            trip_id_last = trip_id
            seq_last = seq
            pattern_id_last = pattern_id

            trip_idx_values[i] = trip_idx
            i += 1
            trip_idx += 1
        mh["trip_idx"] = trip_idx_values

        # deal with single trip case for a given stop
        largest_headway = self.end - self.start
        mh.loc[mh["trip_count"] == 1, "headway"] = largest_headway

        # deal with first trip for a stop & pattern
        mh.loc[(mh["trip_count"] > 1) & (mh["trip_idx"] == 0), "headway"] = np.nan
        mh["headway"] = mh["headway"].fillna(method="bfill")

        # take the min of the headways computed among the stops of a given trip
        mh = mh[["pattern_id", "trip_id", "headway"]].groupby("pattern_id").min().reset_index(drop=False)

        # compute the mean headway computed among the trips of a given pattern
        mh = mh[["pattern_id", "headway"]].groupby("pattern_id").mean().reset_index(drop=False)

        self.line_segments = pd.merge(self.line_segments, mh, on=["pattern_id"], how="left")

    def create_stop_vertices(self):
        # select all stops
        sql = "SELECT stop_id, ST_AsText(geometry) coord FROM stops"
        self.stop_vertices = pd.read_sql(sql, self.pt_conn)

        # filter stops that are used on the given time range
        stops_ids = pd.concat((self.line_segments.from_stop, self.line_segments.to_stop), axis=0).unique()
        self.stop_vertices = self.stop_vertices.loc[self.stop_vertices.stop_id.isin(stops_ids)]

        # uniform attributes
        self.stop_vertices["line_id"] = None
        self.stop_vertices["taz_id"] = None
        self.stop_vertices["line_seg_idx"] = np.nan
        self.stop_vertices["type"] = "stop"

    def create_boarding_vertices(self):
        self.boarding_vertices = self.line_segments[["line_id", "seq", "from_stop"]].copy(deep=True)
        self.boarding_vertices.rename(columns={"seq": "line_seg_idx", "from_stop": "stop_id"}, inplace=True)
        self.boarding_vertices = pd.merge(
            self.boarding_vertices, self.stop_vertices[["stop_id", "coord"]], on="stop_id", how="left"
        )

        # uniform attributes
        self.boarding_vertices["type"] = "boarding"
        self.boarding_vertices["taz_id"] = None

    def create_alighting_vertices(self):
        self.alighting_vertices = self.line_segments[["line_id", "seq", "to_stop"]].copy(deep=True)
        self.alighting_vertices.rename(columns={"seq": "line_seg_idx", "to_stop": "stop_id"}, inplace=True)
        self.alighting_vertices = pd.merge(
            self.alighting_vertices, self.stop_vertices[["stop_id", "coord"]], on="stop_id", how="left"
        )

        # uniform attributes
        self.alighting_vertices["type"] = "alighting"
        self.alighting_vertices["taz_id"] = None

    def create_od_vertices(self):
        sql = """SELECT node_id AS taz_id, ST_AsText(geometry) AS coord FROM nodes WHERE is_centroid = 1"""
        self.od_vertices = pd.read_sql(sql, self.proj_conn)

        # uniform attributes
        self.od_vertices["type"] = "od"
        self.od_vertices["stop_id"] = None
        self.od_vertices["line_id"] = None
        self.od_vertices["line_seg_idx"] = np.nan

    def create_vertices(self):
        """Graph vertices creation as a dataframe.

        Verticealighting_verticess have the following attributes:
            - vert_id: int
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
        self.create_od_vertices()

        # stack the dataframes on top of each other
        self.vertices = pd.concat(
            [
                self.od_vertices,
                self.stop_vertices,
                self.boarding_vertices,
                self.alighting_vertices,
            ],
            axis=0,
        )
        self.vertices.line_seg_idx = self.vertices.line_seg_idx.astype("Int32")

        # reset index and copy it to column
        self.vertices.reset_index(drop=True, inplace=True)
        self.vertices.index.name = "index"
        self.vertices["vert_id"] = self.vertices.index

        self.vertices = self.vertices[self.vertex_cols]

    def create_on_board_edges(self):
        self.on_board_edges = self.line_segments[["line_id", "seq", "trav_time"]].copy(deep=True)
        self.on_board_edges.rename(columns={"seq": "line_seg_idx"}, inplace=True)

        # get tail vertex index
        self.on_board_edges = pd.merge(
            self.on_board_edges,
            self.vertices[self.vertices.type == "boarding"][["line_id", "line_seg_idx", "vert_id"]],
            on=["line_id", "line_seg_idx"],
            how="left",
        )
        self.on_board_edges.rename(columns={"vert_id": "tail_vert_id"}, inplace=True)

        # get head vertex index
        self.on_board_edges = pd.merge(
            self.on_board_edges,
            self.vertices[self.vertices.type == "alighting"][["line_id", "line_seg_idx", "vert_id"]],
            on=["line_id", "line_seg_idx"],
            how="left",
        )
        self.on_board_edges.rename(columns={"vert_id": "head_vert_id"}, inplace=True)

        # uniform attributes
        self.on_board_edges["type"] = "on-board"
        self.on_board_edges["freq"] = np.inf
        self.on_board_edges["stop_id"] = None
        self.on_board_edges["o_line_id"] = None
        self.on_board_edges["d_line_id"] = None
        self.on_board_edges["transfer_id"] = None

    def create_boarding_edges(self):
        self.boarding_edges = self.line_segments[["line_id", "seq", "from_stop", "freq"]].copy(deep=True)
        self.boarding_edges.rename(columns={"seq": "line_seg_idx", "from_stop": "stop_id"}, inplace=True)

        # get tail vertex index (stop vertex)
        self.boarding_edges = pd.merge(
            self.boarding_edges,
            self.vertices[self.vertices.type == "stop"][["stop_id", "vert_id"]],
            on="stop_id",
            how="left",
        )
        self.boarding_edges.rename(columns={"vert_id": "tail_vert_id"}, inplace=True)

        # get head vertex index (boarding vertex)
        self.boarding_edges = pd.merge(
            self.boarding_edges,
            self.vertices[self.vertices.type == "boarding"][["line_id", "line_seg_idx", "vert_id"]],
            on=["line_id", "line_seg_idx"],
            how="left",
        )
        self.boarding_edges.rename(columns={"vert_id": "head_vert_id"}, inplace=True)

        # frequency update : line_freq / wait_time_factor
        wait_time_factor_inv = 1.0 / self.wait_time_factor
        self.boarding_edges["freq"] *= wait_time_factor_inv

        # uniform attributes
        self.boarding_edges["type"] = "boarding"
        self.boarding_edges["trav_time"] = 0.5 * self.uniform_dwell_time + self.a_tiny_time_duration
        self.boarding_edges["o_line_id"] = None
        self.boarding_edges["d_line_id"] = None
        self.boarding_edges["transfer_id"] = None

    def create_alighting_edges(self):
        self.alighting_edges = self.line_segments[["line_id", "seq", "to_stop"]].copy(deep=True)
        self.alighting_edges.rename(columns={"seq": "line_seg_idx", "to_stop": "stop_id"}, inplace=True)

        # get tail vertex index (alighting vertex)
        self.alighting_edges = pd.merge(
            self.alighting_edges,
            self.vertices[self.vertices.type == "alighting"][["line_id", "line_seg_idx", "vert_id"]],
            on=["line_id", "line_seg_idx"],
            how="left",
        )
        self.alighting_edges.rename(columns={"vert_id": "tail_vert_id"}, inplace=True)

        # get head vertex index (stop vertex)
        self.alighting_edges = pd.merge(
            self.alighting_edges,
            self.vertices[self.vertices.type == "stop"][["stop_id", "vert_id"]],
            on="stop_id",
            how="left",
        )
        self.alighting_edges.rename(columns={"vert_id": "head_vert_id"}, inplace=True)

        # uniform attributes
        self.alighting_edges["type"] = "alighting"
        self.alighting_edges["o_line_id"] = None
        self.alighting_edges["d_line_id"] = None
        self.alighting_edges["transfer_id"] = None
        self.alighting_edges["freq"] = np.inf
        self.alighting_edges["trav_time"] = (
            0.5 * self.uniform_dwell_time + self.alighting_penalty + self.a_tiny_time_duration
        )

    def create_dwell_edges(self):
        # we start by removing the first segment of each line
        self.dwell_edges = self.line_segments.loc[self.line_segments.seq != 0][["line_id", "from_stop", "seq"]]
        self.dwell_edges.rename(columns={"seq": "line_seg_idx"}, inplace=True)

        # we take the first stop of the segment
        self.dwell_edges["stop_id"] = self.dwell_edges.from_stop

        # head vertex index (boarding vertex)
        # boarding vertices of line segments [1:segment_count+1]
        self.dwell_edges = pd.merge(
            self.dwell_edges,
            self.vertices[self.vertices.type == "boarding"][["line_id", "stop_id", "vert_id", "line_seg_idx"]],
            on=["line_id", "stop_id", "line_seg_idx"],
            how="left",
        )
        self.dwell_edges.rename(columns={"vert_id": "head_vert_id"}, inplace=True)

        # tail vertex index (alighting vertex)
        # aligthing vertices of line segments [0:segment_count]
        self.dwell_edges.line_seg_idx -= 1
        self.dwell_edges = pd.merge(
            self.dwell_edges,
            self.vertices[self.vertices.type == "alighting"][["line_id", "stop_id", "vert_id", "line_seg_idx"]],
            on=["line_id", "stop_id", "line_seg_idx"],
            how="left",
        )
        self.dwell_edges.rename(columns={"vert_id": "tail_vert_id"}, inplace=True)

        # clean-up
        self.dwell_edges.drop("from_stop", axis=1, inplace=True)

        # uniform values
        self.dwell_edges["line_seg_idx"] = np.nan
        self.dwell_edges["type"] = "dwell"
        self.dwell_edges["o_line_id"] = None
        self.dwell_edges["d_line_id"] = None
        self.dwell_edges["transfer_id"] = None
        self.dwell_edges["freq"] = np.inf
        self.dwell_edges["trav_time"] = self.uniform_dwell_time

    def create_connector_edges(self):
        """Create the connector edges between each stop and the closest od."""

        # longlat to projected CRS transfromer
        transformer = pyproj.Transformer.from_crs(
            pyproj.CRS(self.global_crs), pyproj.CRS(self.projected_crs), always_xy=True
        ).transform

        # Select/copy the od vertices and project their coordinates
        od_vertices = self.vertices[self.vertices.type == "od"][["vert_id", "taz_id", "coord"]].copy(deep=True)
        od_coords = od_vertices["coord"].apply(
            lambda coord: shapely.ops.transform(transformer, shapely.from_wkt(coord))
        )
        od_coords = np.array(list(od_coords.apply(lambda coord: (coord.x, coord.y))))

        # Select/copy the stop vertices and project their coordinates
        stop_vertices = self.vertices[self.vertices.type == "stop"][["vert_id", "stop_id", "coord"]].copy(deep=True)
        stop_coords = stop_vertices["coord"].apply(
            lambda coord: shapely.ops.transform(transformer, shapely.from_wkt(coord))
        )
        stop_coords = np.array(list(stop_coords.apply(lambda coord: (coord.x, coord.y))))

        # query the kdTree for the closest (k=1) od for each stop in parallel (workers=-1)
        kdTree = cKDTree(od_coords)
        distance, index = kdTree.query(stop_coords, k=1, distance_upper_bound=np.inf, workers=self.num_threads)
        nearest_od = od_vertices.iloc[index][["vert_id", "taz_id"]].reset_index(drop=True)
        trav_time = pd.Series(distance * self.walking_speed, name="trav_time")

        # access connectors
        access_connector_edges = pd.concat(
            [
                stop_vertices[["stop_id", "vert_id"]]
                .reset_index(drop=True)
                .rename(columns={"vert_id": "head_vert_id"}),
                nearest_od.rename(columns={"vert_id": "tail_vert_id"}),
                trav_time,
            ],
            axis=1,
        )
        # uniform values
        access_connector_edges["type"] = "connector"
        access_connector_edges["line_seg_idx"] = np.nan
        access_connector_edges["freq"] = np.inf
        access_connector_edges["o_line_id"] = None
        access_connector_edges["d_line_id"] = None
        access_connector_edges["transfer_id"] = None

        # egress connectors
        egress_connector_edges = access_connector_edges.copy(deep=True)
        egress_connector_edges.rename(
            columns={"head_vert_id": "tail_vert_id", "tail_vert_id": "head_vert_id"}, inplace=True
        )

        # uniform values
        egress_connector_edges["type"] = "connector"
        egress_connector_edges["line_seg_idx"] = np.nan
        egress_connector_edges["freq"] = np.inf
        egress_connector_edges["o_line_id"] = None
        egress_connector_edges["d_line_id"] = None
        egress_connector_edges["transfer_id"] = None

        # travel time update
        access_connector_edges.trav_time *= self.access_time_factor
        egress_connector_edges.trav_time *= self.egress_time_factor

        self.connector_edges = pd.concat([access_connector_edges, egress_connector_edges], axis=0)

    def create_inner_stop_transfer_edges(self):
        alighting = self.vertices[self.vertices.type == "alighting"][["stop_id", "line_id"]].rename(
            columns={"line_id": "o_line_id"}
        )
        boarding = self.vertices[self.vertices.type == "boarding"][["stop_id", "line_id"]].rename(
            columns={"line_id": "d_line_id"}
        )
        inner_stop_transfer_edges = pd.merge(alighting, boarding, on="stop_id", how="inner")

        # remove entries that have the same line as origin and destination
        inner_stop_transfer_edges = inner_stop_transfer_edges.loc[
            inner_stop_transfer_edges["o_line_id"] != inner_stop_transfer_edges["d_line_id"]
        ]

        # get tail vertex index (alighting vertex)
        inner_stop_transfer_edges = pd.merge(
            inner_stop_transfer_edges,
            self.vertices[self.vertices.type == "alighting"][["stop_id", "line_id", "vert_id"]],
            left_on=["o_line_id", "stop_id"],
            right_on=["line_id", "stop_id"],
            how="inner",
        )
        inner_stop_transfer_edges.rename(columns={"vert_id": "tail_vert_id"}, inplace=True)
        inner_stop_transfer_edges.drop(["line_id"], axis=1, inplace=True)

        # get head vertex index (boarding vertex)
        inner_stop_transfer_edges = pd.merge(
            inner_stop_transfer_edges,
            self.vertices[self.vertices.type == "boarding"][["stop_id", "line_id", "vert_id"]],
            left_on=["d_line_id", "stop_id"],
            right_on=["line_id", "stop_id"],
            how="inner",
        )
        inner_stop_transfer_edges.rename(columns={"vert_id": "head_vert_id"}, inplace=True)
        inner_stop_transfer_edges.drop(["line_id"], axis=1, inplace=True)

        # update the transfer edge frequency
        inner_stop_transfer_edges = pd.merge(
            inner_stop_transfer_edges,
            self.line_segments[["from_stop", "line_id", "freq"]],
            left_on=["stop_id", "d_line_id"],
            right_on=["from_stop", "line_id"],
            how="left",
        )
        inner_stop_transfer_edges.drop(["from_stop", "line_id"], axis=1, inplace=True)

        # uniform attributes
        inner_stop_transfer_edges["line_id"] = None
        inner_stop_transfer_edges["line_seg_idx"] = np.nan
        inner_stop_transfer_edges["type"] = "transfer"
        inner_stop_transfer_edges["transfer_id"] = None

        # frequency update : line_freq / wait_time_factor
        wait_time_factor_inv = 1.0 / self.wait_time_factor
        inner_stop_transfer_edges["freq"] *= wait_time_factor_inv

        # travel time update : dwell_time + alighting_penalty
        inner_stop_transfer_edges["trav_time"] = self.uniform_dwell_time + self.alighting_penalty

        self.inner_stop_transfer_edges = inner_stop_transfer_edges

    def create_outer_stop_transfer_edges(self):
        pass

    def create_transfer_edges(self):
        self.create_inner_stop_transfer_edges()

        self.transfer_edges = self.inner_stop_transfer_edges

    def create_walking_edges(self):
        pass

    def create_edges(self):
        """Graph edges creation as a dataframe.

        Edges have the following attributes:
            - type (either 'on-board', 'boarding', 'alighting', 'dwell', 'transfer', 'connector' or 'walking'): str
            - line_id (only applies to 'on-board', 'boarding', 'alighting' and 'dwell' edges): str
            - stop_id: str
            - line_seg_idx (only applies to 'on-board', 'boarding' and 'alighting' edges): int
            - tail_vert_id: int
            - head_vert_id: int
            - trav_time (edge travel time): float
            - freq (frequency): float
            - o_line_id: str
            - d_line_id: str
            - transfer_id: str

        """

        self.create_on_board_edges()
        self.create_dwell_edges()
        self.create_boarding_edges()
        self.create_alighting_edges()
        self.create_connector_edges()
        self.create_transfer_edges()

        # stack the dataframes on top of each other
        self.edges = pd.concat(
            [
                self.on_board_edges,
                self.boarding_edges,
                self.alighting_edges,
                self.dwell_edges,
                self.connector_edges,
                self.transfer_edges,
            ],
            axis=0,
        )
        self.edges.line_seg_idx = self.edges.line_seg_idx.astype("Int32")

        # reset index and copy it to column
        self.edges.reset_index(drop=True, inplace=True)
        self.edges.index.name = "index"
        self.edges["edge_id"] = self.edges.index

        self.edges = self.edges[self.edges_cols]
