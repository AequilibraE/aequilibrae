""" Create the graph used by public transport assignment algorithms.
"""

import numpy as np
import pandas as pd
import pyproj
import shapely
import shapely.ops
from scipy.spatial import cKDTree, minkowski_distance
from shapely.geometry import Point


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2.0 * np.arcsin(np.sqrt(a))
    distance_m = 6367000.0 * c
    return distance_m


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
        seed=124,
        coord_noise=True,
        noise_coef=1.0e-5,
        with_inner_stop_transfers=True,
        with_outer_stop_transfers=True,
        with_walking_edges=True,
        distance_upper_bound=np.inf,
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

        # vertices
        self.stop_vertices = None
        self.boarding_vertices = None
        self.alighting_vertices = None
        self.od_vertices = None

        # edges
        self.on_board_edges = None
        self.dwell_edges = None
        self.alighting_edges = None
        self.boarding_edges = None
        self.connector_edges = None
        self.inner_stop_transfer_edges = pd.DataFrame()
        self.outer_stop_transfer_edges = pd.DataFrame()
        self.walking_edges = pd.DataFrame()

        self.global_crs = pyproj.CRS(global_crs)
        self.projected_crs = pyproj.CRS(projected_crs)

        # longlat to projected CRS transfromer
        self.transformer = pyproj.Transformer.from_crs(
            self.global_crs, self.projected_crs, always_xy=True
        ).transform

        # Add some spatial noise so that stop, boarding and aligthing vertices
        # are not colocated
        self.rng = np.random.default_rng(seed=seed)
        self.coord_noise = coord_noise
        self.noise_coef = noise_coef

        # graph parameters
        self.uniform_dwell_time = 30
        self.alighting_penalty = 480
        self.a_tiny_time_duration = 1.0e-06
        self.wait_time_factor = 1.0
        self.walk_time_factor = 1.0
        self.walking_speed = 1.0
        self.access_time_factor = 1.0
        self.egress_time_factor = 1.0
        self.with_inner_stop_transfers = with_inner_stop_transfers
        self.with_outer_stop_transfers = with_outer_stop_transfers
        self.with_walking_edges = with_walking_edges
        self.distance_upper_bound = distance_upper_bound

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
        sql = (
            "SELECT pattern_id, seq, CAST(from_stop AS TEXT) from_stop, CAST(to_stop AS TEXT) to_stop FROM route_links"
        )
        route_links = pd.read_sql(
            sql=sql,
            con=self.pt_conn,
        )
        route_links = route_links.loc[route_links.pattern_id.isin(pattern_ids)]

        # create a line segment table
        sql = "SELECT pattern_id, shortname FROM routes" ""
        routes = pd.read_sql(
            sql=sql,
            con=self.pt_conn,
        )
        routes["line_id"] = routes["shortname"] + "_" + routes["pattern_id"].astype(str)
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
            sql = """SELECT trips_schedule.trip_id, trips_schedule.seq, trips_schedule.arrival,
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
        sql = "SELECT CAST(stop_id AS TEXT) stop_id, ST_AsText(geometry) coord FROM stops"
        stop_vertices = pd.read_sql(sql, self.pt_conn)

        # filter stops that are used on the given time range
        stops_ids = pd.concat((self.line_segments.from_stop, self.line_segments.to_stop), axis=0).unique()
        stop_vertices = stop_vertices.loc[stop_vertices.stop_id.isin(stops_ids)]

        # uniform attributes
        stop_vertices["line_id"] = None
        stop_vertices["taz_id"] = None
        stop_vertices["line_seg_idx"] = np.nan
        stop_vertices["type"] = "stop"

        self.stop_vertices = stop_vertices

    def create_boarding_vertices(self):
        boarding_vertices = self.line_segments[["line_id", "seq", "from_stop"]].copy(deep=True)
        boarding_vertices.rename(columns={"seq": "line_seg_idx", "from_stop": "stop_id"}, inplace=True)
        boarding_vertices = pd.merge(
            boarding_vertices, self.stop_vertices[["stop_id", "coord"]], on="stop_id", how="left"
        )

        # uniform attributes
        boarding_vertices["type"] = "boarding"
        boarding_vertices["taz_id"] = None

        # add noise
        if self.coord_noise:
            boarding_vertices["x"] = boarding_vertices.coord.map(lambda c: shapely.wkt.loads(c).x)
            boarding_vertices["y"] = boarding_vertices.coord.map(lambda c: shapely.wkt.loads(c).y)
            n_boarding = len(boarding_vertices)
            boarding_vertices["x"] += self.noise_coef * (np.random.rand(n_boarding) - 0.5)
            boarding_vertices["y"] += self.noise_coef * (np.random.rand(n_boarding) - 0.5)
            boarding_vertices["coord"] = boarding_vertices.apply(lambda row: Point(row.x, row.y).wkt, axis=1)
            boarding_vertices.drop(["x", "y"], axis=1, inplace=True)

        self.boarding_vertices = boarding_vertices

    def create_alighting_vertices(self):
        alighting_vertices = self.line_segments[["line_id", "seq", "to_stop"]].copy(deep=True)
        alighting_vertices.rename(columns={"seq": "line_seg_idx", "to_stop": "stop_id"}, inplace=True)
        alighting_vertices = pd.merge(
            alighting_vertices, self.stop_vertices[["stop_id", "coord"]], on="stop_id", how="left"
        )

        # uniform attributes
        alighting_vertices["type"] = "alighting"
        alighting_vertices["taz_id"] = None

        # add noise
        if self.coord_noise:
            alighting_vertices["x"] = alighting_vertices.coord.map(lambda c: shapely.wkt.loads(c).x)
            alighting_vertices["y"] = alighting_vertices.coord.map(lambda c: shapely.wkt.loads(c).y)
            n_alighting = len(alighting_vertices)
            alighting_vertices["x"] += self.noise_coef * (np.random.rand(n_alighting) - 0.5)
            alighting_vertices["y"] += self.noise_coef * (np.random.rand(n_alighting) - 0.5)
            alighting_vertices["coord"] = alighting_vertices.apply(lambda row: Point(row.x, row.y).wkt, axis=1)
            alighting_vertices.drop(["x", "y"], axis=1, inplace=True)

        self.alighting_vertices = alighting_vertices

    def create_od_vertices(self):
        sql = """SELECT CAST(node_id AS TEXT) AS taz_id, ST_AsText(geometry) AS coord FROM nodes 
            WHERE is_centroid = 1"""
        od_vertices = pd.read_sql(sql, self.proj_conn)

        # uniform attributes
        od_vertices["type"] = "od"
        od_vertices["stop_id"] = None
        od_vertices["line_id"] = None
        od_vertices["line_seg_idx"] = np.nan

        self.od_vertices = od_vertices

    def create_vertices(self):
        """Graph vertices creation as a dataframe.

        Vertices have the following attributes:
            - vert_id: int
            - type (either 'stop', 'boarding', 'alighting', 'od'): str
            - stop_id (only applies to 'stop', 'boarding' and 'alighting' vertices): str
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

        # reset index and copy it to column
        self.vertices.reset_index(drop=True, inplace=True)
        self.vertices.index.name = "index"
        self.vertices["vert_id"] = self.vertices.index
        self.vertices = self.vertices[self.vertex_cols]

        # data types
        self.vertices.vert_id = self.vertices.vert_id.astype(int)
        self.vertices["type"] = self.vertices["type"].astype("category")
        self.vertices.stop_id = self.vertices.stop_id.astype(str)
        self.vertices.line_id = self.vertices.line_id.astype(str)
        self.vertices.line_seg_idx = self.vertices.line_seg_idx.astype("Int32")
        self.vertices.taz_id = self.vertices.taz_id.astype(str)
        self.vertices.coord = self.vertices.coord.astype(str)

    def create_on_board_edges(self):
        on_board_edges = self.line_segments[["line_id", "seq", "trav_time"]].copy(deep=True)
        on_board_edges.rename(columns={"seq": "line_seg_idx"}, inplace=True)

        # get tail vertex index
        on_board_edges = pd.merge(
            on_board_edges,
            self.vertices[self.vertices.type == "boarding"][["line_id", "line_seg_idx", "vert_id"]],
            on=["line_id", "line_seg_idx"],
            how="left",
        )
        on_board_edges.rename(columns={"vert_id": "tail_vert_id"}, inplace=True)

        # get head vertex index
        on_board_edges = pd.merge(
            on_board_edges,
            self.vertices[self.vertices.type == "alighting"][["line_id", "line_seg_idx", "vert_id"]],
            on=["line_id", "line_seg_idx"],
            how="left",
        )
        on_board_edges.rename(columns={"vert_id": "head_vert_id"}, inplace=True)

        # uniform attributes
        on_board_edges["type"] = "on-board"
        on_board_edges["freq"] = np.inf
        on_board_edges["stop_id"] = None
        on_board_edges["o_line_id"] = None
        on_board_edges["d_line_id"] = None
        on_board_edges["transfer_id"] = None

        self.on_board_edges = on_board_edges

    def create_boarding_edges(self):
        boarding_edges = self.line_segments[["line_id", "seq", "from_stop", "freq"]].copy(deep=True)
        boarding_edges.rename(columns={"seq": "line_seg_idx", "from_stop": "stop_id"}, inplace=True)

        # get tail vertex index (stop vertex)
        boarding_edges = pd.merge(
            boarding_edges,
            self.vertices[self.vertices.type == "stop"][["stop_id", "vert_id"]],
            on="stop_id",
            how="left",
        )
        boarding_edges.rename(columns={"vert_id": "tail_vert_id"}, inplace=True)

        # get head vertex index (boarding vertex)
        boarding_edges = pd.merge(
            boarding_edges,
            self.vertices[self.vertices.type == "boarding"][["line_id", "line_seg_idx", "vert_id"]],
            on=["line_id", "line_seg_idx"],
            how="left",
        )
        boarding_edges.rename(columns={"vert_id": "head_vert_id"}, inplace=True)

        # frequency update : line_freq / wait_time_factor
        wait_time_factor_inv = 1.0 / self.wait_time_factor
        boarding_edges["freq"] *= wait_time_factor_inv

        # uniform attributes
        boarding_edges["type"] = "boarding"
        boarding_edges["trav_time"] = 0.5 * self.uniform_dwell_time + self.a_tiny_time_duration
        boarding_edges["o_line_id"] = None
        boarding_edges["d_line_id"] = None
        boarding_edges["transfer_id"] = None

        self.boarding_edges = boarding_edges

    def create_alighting_edges(self):
        alighting_edges = self.line_segments[["line_id", "seq", "to_stop"]].copy(deep=True)
        alighting_edges.rename(columns={"seq": "line_seg_idx", "to_stop": "stop_id"}, inplace=True)

        # get tail vertex index (alighting vertex)
        alighting_edges = pd.merge(
            alighting_edges,
            self.vertices[self.vertices.type == "alighting"][["line_id", "line_seg_idx", "vert_id"]],
            on=["line_id", "line_seg_idx"],
            how="left",
        )
        alighting_edges.rename(columns={"vert_id": "tail_vert_id"}, inplace=True)

        # get head vertex index (stop vertex)
        alighting_edges = pd.merge(
            alighting_edges,
            self.vertices[self.vertices.type == "stop"][["stop_id", "vert_id"]],
            on="stop_id",
            how="left",
        )
        alighting_edges.rename(columns={"vert_id": "head_vert_id"}, inplace=True)

        # uniform attributes
        alighting_edges["type"] = "alighting"
        alighting_edges["o_line_id"] = None
        alighting_edges["d_line_id"] = None
        alighting_edges["transfer_id"] = None
        alighting_edges["freq"] = np.inf
        alighting_edges["trav_time"] = (
            0.5 * self.uniform_dwell_time + self.alighting_penalty + self.a_tiny_time_duration
        )

        self.alighting_edges = alighting_edges

    def create_dwell_edges(self):
        # we start by removing the first segment of each line
        dwell_edges = self.line_segments.loc[self.line_segments.seq != 0][["line_id", "from_stop", "seq"]]
        dwell_edges.rename(columns={"seq": "line_seg_idx"}, inplace=True)

        # we take the first stop of the segment
        dwell_edges["stop_id"] = dwell_edges.from_stop

        # head vertex index (boarding vertex)
        # boarding vertices of line segments [1:segment_count+1]
        dwell_edges = pd.merge(
            dwell_edges,
            self.vertices[self.vertices.type == "boarding"][["line_id", "stop_id", "vert_id", "line_seg_idx"]],
            on=["line_id", "stop_id", "line_seg_idx"],
            how="left",
        )
        dwell_edges.rename(columns={"vert_id": "head_vert_id"}, inplace=True)

        # tail vertex index (alighting vertex)
        # aligthing vertices of line segments [0:segment_count]
        dwell_edges.line_seg_idx -= 1
        dwell_edges = pd.merge(
            dwell_edges,
            self.vertices[self.vertices.type == "alighting"][["line_id", "stop_id", "vert_id", "line_seg_idx"]],
            on=["line_id", "stop_id", "line_seg_idx"],
            how="left",
        )
        dwell_edges.rename(columns={"vert_id": "tail_vert_id"}, inplace=True)

        # clean-up
        dwell_edges.drop("from_stop", axis=1, inplace=True)

        # uniform values
        dwell_edges["line_seg_idx"] = np.nan
        dwell_edges["type"] = "dwell"
        dwell_edges["o_line_id"] = None
        dwell_edges["d_line_id"] = None
        dwell_edges["transfer_id"] = None
        dwell_edges["freq"] = np.inf
        dwell_edges["trav_time"] = self.uniform_dwell_time

        self.dwell_edges = dwell_edges

    def create_connector_edges(self, method="overlapping_regions", allow_missing_connections=True):
        """
        Create the connector edges between each stops and ODs.

        Nearest neighbour: Creates edges between every stop and its nearest OD.

        Overlapping regions: Creates edges between all stops that lying within the circle
            centered each OD whose radius is the distance to the other nearest OD.
        """
        assert method in ["overlapping_regions", "nearest_neighbour"]

        # Select/copy the od vertices and project their coordinates
        od_vertices = self.vertices[self.vertices.type == "od"][["vert_id", "taz_id", "coord"]].copy(deep=True)
        od_vertices.reset_index(drop=True, inplace=True)
        od_coords = od_vertices["coord"].apply(
            lambda coord: shapely.ops.transform(self.transformer, shapely.from_wkt(coord))
        )
        od_coords = np.array(list(od_coords.apply(lambda coord: (coord.x, coord.y))))

        # Select/copy the stop vertices and project their coordinates
        stop_vertices = self.vertices[self.vertices.type == "stop"][["vert_id", "stop_id", "coord"]].copy(deep=True)
        stop_vertices.reset_index(drop=True, inplace=True)
        stop_coords = stop_vertices["coord"].apply(
            lambda coord: shapely.ops.transform(self.transformer, shapely.from_wkt(coord))
        )
        stop_coords = np.array(list(stop_coords.apply(lambda coord: (coord.x, coord.y))))

        kdTree = cKDTree(od_coords)

        if method == "nearest_neighbour":
            # query the kdTree for the closest (k=1) od for each stop in parallel (workers=-1)
            distance, index = kdTree.query(
                stop_coords, k=1, distance_upper_bound=self.distance_upper_bound, workers=self.num_threads
            )
            nearest_od = od_vertices.iloc[index][["vert_id", "taz_id"]].reset_index(drop=True)
            trav_time = pd.Series(distance / self.walking_speed, name="trav_time")

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

        elif method == "overlapping_regions":
            # Construct a kdtree so we can lookup the 2nd closest OD to each OD (the first being itself)
            distance, _ = kdTree.query(
                od_coords, k=[2], distance_upper_bound=self.distance_upper_bound, workers=self.num_threads
            )
            distance = distance.reshape(-1)

            # Construct a kdtree so we can query all the stops within the radius around each OD
            stop_kdTree = cKDTree(stop_coords)
            results = stop_kdTree.query_ball_point(od_coords, distance, workers=self.num_threads)

            # Build up a list of dataframes to concat, each dataframe corresponds to all connectors for a given OD
            connectors = []
            for i, verts in enumerate(results):
                distance = minkowski_distance(od_coords[i], stop_coords[verts])
                df = stop_vertices["vert_id"].iloc[verts].to_frame()
                df["head_vert_id"] = od_vertices.iloc[i]["vert_id"]
                df["trav_time"] = distance / self.walking_speed
                connectors.append(df)

            # access connectors
            access_connector_edges = (
                pd.concat(connectors).rename(columns={"vert_id": "tail_vert_id"}).reset_index(drop=True)
            )

            if not allow_missing_connections:
                # Now we need to build up the edges for the stops without connectors
                missing = stop_vertices["vert_id"].isin(access_connector_edges["tail_vert_id"])
                missing = missing[~missing].index

                distance, index = kdTree.query(
                    stop_coords[missing], k=1, distance_upper_bound=np.inf, workers=self.num_threads
                )
                nearest_od = od_vertices["vert_id"].iloc[index].reset_index(drop=True)
                trav_time = pd.Series(distance / self.walking_speed, name="trav_time")
                missing_edges = pd.concat(
                    [
                        stop_vertices["vert_id"].iloc[missing].reset_index(drop=True).rename("tail_vert_id"),
                        nearest_od.rename("head_vert_id"),
                        trav_time,
                    ],
                    axis=1,
                )

                access_connector_edges = pd.concat([access_connector_edges, missing_edges], axis=0)

        # uniform values
        access_connector_edges["type"] = "access_connector"
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
        egress_connector_edges["type"] = "egress_connector"
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
        """Create transfer edges between distinct lines of each stop."""

        alighting = self.vertices[self.vertices.type == "alighting"][["stop_id", "line_id", "vert_id"]].rename(
            columns={"line_id": "o_line_id", "vert_id": "tail_vert_id"}
        )
        boarding = self.vertices[self.vertices.type == "boarding"][["stop_id", "line_id", "vert_id"]].rename(
            columns={"line_id": "d_line_id", "vert_id": "head_vert_id"}
        )
        inner_stop_transfer_edges = pd.merge(alighting, boarding, on="stop_id", how="inner")

        # remove entries that have the same line as origin and destination
        inner_stop_transfer_edges = inner_stop_transfer_edges.loc[
            inner_stop_transfer_edges["o_line_id"] != inner_stop_transfer_edges["d_line_id"]
        ]

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
        inner_stop_transfer_edges["type"] = "inner_transfer"
        inner_stop_transfer_edges["transfer_id"] = None

        # frequency update : line_freq / wait_time_factor
        wait_time_factor_inv = 1.0 / self.wait_time_factor
        inner_stop_transfer_edges["freq"] *= wait_time_factor_inv

        # travel time update : dwell_time + alighting_penalty
        inner_stop_transfer_edges["trav_time"] = self.uniform_dwell_time + self.alighting_penalty

        self.inner_stop_transfer_edges = inner_stop_transfer_edges

    def create_outer_stop_transfer_edges(self):
        """Create transfer edges between distinct lines/stops of each station."""

        sql = """
        SELECT CAST(stop_id as TEXT) stop_id, CAST(parent_station as TEXT) parent_station FROM stops
        WHERE parent_station IS NOT NULL AND parent_station <> ''
        """
        stops = pd.read_sql(sql, self.pt_conn)
        stations = stops.groupby("parent_station").size().to_frame("stop_count").reset_index(drop=False)

        # we only keep the stations which contain at least 2 stops
        stations = stations[stations["stop_count"] > 1]
        station_list = stations["parent_station"].values
        stops = stops[stops.parent_station.isin(station_list)]

        # load the aligthing vertices (tail of transfer edges)
        alighting = self.vertices[self.vertices.type == "alighting"][["stop_id", "line_id", "vert_id", "coord"]].rename(
            columns={"line_id": "o_line_id", "coord": "o_coord", "vert_id": "tail_vert_id"}
        )
        # add the station id
        alighting = pd.merge(alighting, stops, on="stop_id", how="inner")
        alighting.rename(columns={"stop_id": "o_stop_id"}, inplace=True)

        # load the boarding vertices (head of transfer edges)
        boarding = self.vertices[self.vertices.type == "boarding"][["stop_id", "line_id", "vert_id", "coord"]].rename(
            columns={"line_id": "d_line_id", "coord": "d_coord", "vert_id": "head_vert_id"}
        )
        # add the station id
        boarding = pd.merge(boarding, stops, on="stop_id", how="inner")
        boarding.rename(columns={"stop_id": "d_stop_id"}, inplace=True)

        outer_stop_transfer_edges = pd.merge(alighting, boarding, on="parent_station", how="inner")
        outer_stop_transfer_edges.drop("parent_station", axis=1, inplace=True)

        outer_stop_transfer_edges.dropna(how="any", inplace=True)

        # remove entries that share the same stop
        outer_stop_transfer_edges = outer_stop_transfer_edges.loc[
            outer_stop_transfer_edges["o_stop_id"] != outer_stop_transfer_edges["d_stop_id"]
        ]

        # remove entries that have the same line as origin and destination
        outer_stop_transfer_edges = outer_stop_transfer_edges.loc[
            outer_stop_transfer_edges["o_line_id"] != outer_stop_transfer_edges["d_line_id"]
        ]

        # update the transfer edge frequency
        outer_stop_transfer_edges = pd.merge(
            outer_stop_transfer_edges,
            self.line_segments[["from_stop", "line_id", "freq"]],
            left_on=["d_stop_id", "d_line_id"],
            right_on=["from_stop", "line_id"],
            how="left",
        )
        outer_stop_transfer_edges.drop(["o_stop_id", "d_stop_id", "from_stop", "line_id"], axis=1, inplace=True)

        # uniform attributes
        outer_stop_transfer_edges["line_id"] = None
        outer_stop_transfer_edges["line_seg_idx"] = np.nan
        outer_stop_transfer_edges["type"] = "outer_transfer"
        outer_stop_transfer_edges["transfer_id"] = None

        # frequency update : line_freq / wait_time_factor
        wait_time_factor_inv = 1.0 / self.wait_time_factor
        outer_stop_transfer_edges["freq"] *= wait_time_factor_inv

        # compute the walking time
        outer_stop_transfer_edges["o_coord_str"] = outer_stop_transfer_edges["o_coord"].str.extract(r"\((.*?)\)")
        outer_stop_transfer_edges[["o_lon", "o_lat"]] = outer_stop_transfer_edges["o_coord_str"].str.split(
            " ", expand=True
        )
        outer_stop_transfer_edges["d_coord_str"] = outer_stop_transfer_edges["d_coord"].str.extract(r"\((.*?)\)")
        outer_stop_transfer_edges[["d_lon", "d_lat"]] = outer_stop_transfer_edges["d_coord_str"].str.split(
            " ", expand=True
        )
        outer_stop_transfer_edges[["o_lon", "o_lat", "d_lon", "d_lat"]] = outer_stop_transfer_edges[
            ["o_lon", "o_lat", "d_lon", "d_lat"]
        ].astype(float)
        outer_stop_transfer_edges["distance"] = haversine(
            outer_stop_transfer_edges.o_lon.to_numpy(),
            outer_stop_transfer_edges.o_lat.to_numpy(),
            outer_stop_transfer_edges.d_lon.to_numpy(),
            outer_stop_transfer_edges.d_lat.to_numpy(),
        )
        outer_stop_transfer_edges["trav_time"] = outer_stop_transfer_edges["distance"] / self.walking_speed
        outer_stop_transfer_edges["trav_time"] *= self.walk_time_factor
        outer_stop_transfer_edges["trav_time"] += self.alighting_penalty

        # cleanup
        outer_stop_transfer_edges.drop(
            ["o_coord_str", "d_coord_str", "o_lon", "o_lat", "d_lon", "d_lat", "o_coord", "d_coord", "distance"],
            axis=1,
            inplace=True,
        )

        self.outer_stop_transfer_edges = outer_stop_transfer_edges

    def create_walking_edges(self):
        """Create walking edges between distinct stops of each station."""

        sql = """
        SELECT CAST(stop_id AS TEXT) stop_id, CAST(parent_station AS TEXT) parent_station FROM stops
        WHERE parent_station IS NOT NULL AND parent_station <> '' 
        """
        stops = pd.read_sql(sql, self.pt_conn)
        stops.drop_duplicates(inplace=True)
        stations = stops.groupby("parent_station").size().to_frame("stop_count").reset_index(drop=False)

        # we only keep the stations which contain at least 2 stops
        stations = stations[stations["stop_count"] > 1]
        station_list = stations["parent_station"].values
        stops = stops[stops.parent_station.isin(station_list)]

        # tail vertex
        o_walking = self.vertices[self.vertices.type == "stop"][["stop_id", "vert_id", "coord"]].rename(
            columns={"coord": "o_coord", "vert_id": "tail_vert_id"}
        )
        o_walking = pd.merge(o_walking, stops, on="stop_id", how="inner")
        o_walking.rename(columns={"stop_id": "o_stop_id"}, inplace=True)

        # head vertex
        d_walking = self.vertices[self.vertices.type == "stop"][["stop_id", "vert_id", "coord"]].rename(
            columns={"coord": "d_coord", "vert_id": "head_vert_id"}
        )
        d_walking = pd.merge(d_walking, stops, on="stop_id", how="inner")
        d_walking.rename(columns={"stop_id": "d_stop_id"}, inplace=True)

        walking_edges = pd.merge(o_walking, d_walking, on="parent_station", how="inner")

        # remove entries that share the same stop
        walking_edges = walking_edges.loc[walking_edges["o_stop_id"] != walking_edges["d_stop_id"]]
        walking_edges.drop("parent_station", axis=1, inplace=True)

        # uniform attributes
        walking_edges["line_id"] = None
        walking_edges["line_seg_idx"] = np.nan
        walking_edges["type"] = "walking"
        walking_edges["transfer_id"] = None
        walking_edges["freq"] = np.inf

        # compute the walking time
        walking_edges["o_coord_str"] = walking_edges["o_coord"].str.extract(r"\((.*?)\)")
        walking_edges[["o_lon", "o_lat"]] = walking_edges["o_coord_str"].str.split(" ", expand=True)
        walking_edges["d_coord_str"] = walking_edges["d_coord"].str.extract(r"\((.*?)\)")
        walking_edges[["d_lon", "d_lat"]] = walking_edges["d_coord_str"].str.split(" ", expand=True)
        walking_edges[["o_lon", "o_lat", "d_lon", "d_lat"]] = walking_edges[
            ["o_lon", "o_lat", "d_lon", "d_lat"]
        ].astype(float)
        walking_edges["distance"] = haversine(
            walking_edges.o_lon.to_numpy(),
            walking_edges.o_lat.to_numpy(),
            walking_edges.d_lon.to_numpy(),
            walking_edges.d_lat.to_numpy(),
        )
        walking_edges["trav_time"] = walking_edges["distance"] / self.walking_speed
        walking_edges["trav_time"] *= self.walk_time_factor

        # cleanup
        walking_edges.drop(
            ["o_coord_str", "d_coord_str", "o_lon", "o_lat", "d_lon", "d_lat", "o_coord", "d_coord", "distance"],
            axis=1,
            inplace=True,
        )

        self.walking_edges = walking_edges

    def create_edges(self):
        """Graph edges creation as a dataframe.

        Edges have the following attributes:
            - type (either 'on-board', 'boarding', 'alighting', 'dwell', 'inner_transfer', 'outer_transfer',
              'access_connector', "egress_connector" or 'walking'): str
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

        # create the graph edges
        self.create_on_board_edges()
        self.create_dwell_edges()
        self.create_boarding_edges()
        self.create_alighting_edges()
        self.create_connector_edges()
        if self.with_inner_stop_transfers:
            self.create_inner_stop_transfer_edges()
        self.create_inner_stop_transfer_edges()
        if self.with_outer_stop_transfers:
            self.create_outer_stop_transfer_edges()
        if self.with_walking_edges:
            self.create_walking_edges()

        # stack the dataframes on top of each other
        self.edges = pd.concat(
            [
                self.on_board_edges,
                self.boarding_edges,
                self.alighting_edges,
                self.dwell_edges,
                self.connector_edges,
                self.inner_stop_transfer_edges,
                self.outer_stop_transfer_edges,
                self.walking_edges,
            ],
            axis=0,
        )

        # reset index and copy it to column
        self.edges.reset_index(drop=True, inplace=True)
        self.edges.index.name = "index"
        self.edges["edge_id"] = self.edges.index
        self.edges = self.edges[self.edges_cols]

        # data types
        self.edges["type"] = self.edges["type"].astype("category")
        self.edges.line_id = self.edges.line_id.astype(str)
        self.edges.stop_id = self.edges.stop_id.astype(str)
        self.edges.line_seg_idx = self.edges.line_seg_idx.astype("Int32")
        self.edges.tail_vert_id = self.edges.tail_vert_id.astype(int)
        self.edges.head_vert_id = self.edges.head_vert_id.astype(int)
        self.edges.trav_time = self.edges.trav_time.astype(float)
        self.edges.freq = self.edges.freq.astype(float)
        self.edges.o_line_id = self.edges.o_line_id.astype(str)
        self.edges.d_line_id = self.edges.d_line_id.astype(str)
        self.edges.transfer_id = self.edges.transfer_id.astype(str)

    def create_additional_db_fields(self):
        # This graph requires some additional tables and fields inorder to store all our information.
        # Currently it mimics what we are storing in the df

        # `node_types` table: contains the different types of nodes we have. This is almost indentical to
        # `link_types`
        self.proj_conn.execute(
            """
            CREATE TABLE node_types (node_type     VARCHAR UNIQUE NOT NULL PRIMARY KEY,
                                     node_type_id  VARCHAR UNIQUE NOT NULL,
                                     description   VARCHAR,
                                     lanes         NUMERIC,
                                     lane_capacity NUMERIC,
                                     speed         NUMERIC)
            """
        )

        # Lets add our current node types
        self.proj_conn.executemany(
            """
            INSERT INTO node_types (node_type, node_type_id, description) VALUES (?, ?, ?)
            """,
            [
                ("OD", "od", "Origin/Destination"),
                ("Stop", "stop", "Stop"),
                ("Alighting", "alighting", "Alighting node"),
                ("Boarding", "boarding", "Boarding node")
            ]
        )

        # We also need to add some additional columns to the `nodes` table
        cols = [
            ("node_type", "TEXT", "REFERENCES link_types"),
            ("stop_id", "TEXT", ""),
            ("line_id", "TEXT", ""),
            ("line_seg_idx", "INTEGER", ""),
            ("taz_id", "TEXT", "")
        ]

        for col, node_type, constraints in cols:
            # Can't use executemany here because we need to template the fields not values.
            self.proj_conn.execute(f"ALTER TABLE nodes ADD COLUMN {col} {node_type} {constraints};")

        # Now onto the links, we need to specifiy all our link types
        self.proj_conn.executemany(
            """
            INSERT INTO link_types (link_type, link_type_id, description) VALUES (?, ?, ?)
            """,
            [
                ("on-board", "o", "From boarding to alighting"),
                ("boarding", "b", "From stop to boarding"),
                ("alighting", "a", "From alighting to stop"),
                ("dwell", "d", "From alighting to boarding"),
                ("access_connector", "A", ""),
                ("egress_connector", "e", ""),
                ("inner_transfer", "t", "Transfer edge within station, from alighting to boarding"),
                ("outer_transfer", "T", "Transfer edge outside of a station, from alighting to boarding"),
                ("walking", "w", "Walking, from stop or walking to stop or walking"),
            ]
        )

        # As well as add our extra columns
        cols = [
            ("line_id", "TEXT", ""),
            ("stop_id", "TEXT", ""),
            ("line_seg_idx", "INTEGER", ""),
            ("trav_time", "NUMERIC", ""),
            ("freq", "NUMERIC", ""),
            ("o_line_id", "TEXT", ""),
            ("d_line_id", "TEXT", ""),
            ("transfer_id", "TEXT", "")
        ]
        for col, type, constraints in cols:
            self.proj_conn.execute(f"ALTER TABLE links ADD COLUMN {col} {type} {constraints};")

        self.proj_conn.commit()

    def save_vertices(self):
        # FIXME: Currently can only save uniquely located nodes. Pedro has dealt with the before by repeatedly shifting
        # the node until it is uniquely located. I think this is a restriction of the rtree used but haven't confirmed.
        # To do that here we'd have to move away from executemany and write the loop ourselves or deal with them seperately.

        # FIXME: We also avoid adding the nodes of type od as they are already in the db from when the zones where added in the
        # notebook. I don't think this is a good solution but I'm not sure what do to without adding the zones and such
        # ourselves.

        # The below query is formated to guarantee the columns line up. The order of the tuples should be the same
        # as the order of the columns.
        #
        #     An object to iterate over namedtuples for each row in the DataFrame with the first field possibly being
        #     the index and following fields being the column values.
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.itertuples.html

        # It should look like the below query. This is generated from
        #
        # from aequilibrae.project.network import Link, Node
        # from aequilibrae.project.network.safe_class import SafeClass
        #
        # node = Node(verticies.iloc[0], project)
        # data, sql = SafeClass._save_new_with_geometry(node)
        #
        # Insert into nodes ("node_id","node_type","stop_id","line_id","line_seg_idx","taz_id",geometry)
        # values(?,?,?,?,?,?,GeomFromWKB(?, 4326))

        # FIXME: We also nned to rename some columns, convert the geometry to wkb as well as replace the NANs with something.
        # I think these changes should be propegated throughout but we'll leave them here for now.
        vertices = self.vertices.rename(columns={"coord": "geometry", "vert_id": "node_id", "type": "node_type"})
        vertices.geometry = shapely.to_wkb(shapely.from_wkt(vertices.geometry.values))
        vertices.line_seg_idx = vertices.line_seg_idx.fillna(-1)

        self.proj_conn.executemany(
            """
            Insert into nodes ("{}","{}","{}","{}","{}","{}",{})
            values(?,?,?,?,?,?,GeomFromWKB(?, {}));
            """.format(*vertices.columns, self.global_crs.to_epsg()),
            vertices[(vertices.node_type != "od") & (~vertices.geometry.duplicated())].itertuples(index=False, name=None)
        )

        self.proj_conn.commit()

    def save_edges(self):
        # FIXME: Just like the verticies, our edges need a little massaging as well. I think the renames, and conversions
        # should be propegated as well.
        edges = self.edges.rename(columns={"head_vert_id": "a_node", "tail_vert_id": "b_node", "type": "link_type"})
        edges["link_id"] = edges.index
        edges["modes"] = "t"

        # We also need to generate the geometry for each edge, this may take a bit
        lines = []
        for row in edges.itertuples():
            line = (shapely.from_wkt(self.vertices["coord"][row.a_node]), shapely.from_wkt(self.vertices["coord"][row.b_node]))
            lines.append(shapely.LineString(line))

        edges["geometry"] = lines

        edges.geometry = shapely.to_wkb(lines)
        edges.line_seg_idx = edges.line_seg_idx.fillna(-1)

        # Just as with the nodes the query should look like thisThis is generated from
        #
        # from aequilibrae.project.network import Link, Node
        # from aequilibrae.project.network.safe_class import SafeClass
        #
        # link = Link(edges.iloc[0], project)
        # data, sql = SafeClass._save_new_with_geometry(link)
        #
        # Insert into links ("link_type","line_id","stop_id","line_seg_idx","b_node","a_node","trav_time","freq","o_line_id","d_line_id","transfer_id","link_id","modes",geometry)
        # values(?,?,?,?,?,?,?,?,?,?,?,?,?,GeomFromWKB(?, 4326))
        self.proj_conn.executemany(
            """
            Insert into links ("{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}",{})
            values(?,?,?,?,?,?,?,?,?,?,?,?,?,GeomFromWKB(?, {}))
            """.format(*edges.columns, self.global_crs.to_epsg()),
            edges.itertuples(index=False, name=None)
        )

        self.proj_conn.commit()
