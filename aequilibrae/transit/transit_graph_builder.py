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
    - trips dir is always 0: opposite directions is not supported

    TODO:
    - transform some of the filtering pandas operations to SQL queries (filter down in the SQL part).
    - instanciate properly using a project path, an aequilibrae project or anything else that follow the
      package guideline (without explicit public_transport_conn and project_conn)
    """

    def __init__(self, public_transport_conn, project_conn, start=61200, end=64800, margin=0, num_threads=-1):
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

        self.line_segments = None
        self.stop_vertices = None
        self.boarding_vertices = None
        self.alighting_vertices = None
        self.od_vertices = None
        self.on_board_edges = None

        self.local_crs = "EPSG:4326"
        self.walking_speed = 1.0

    def create_line_segments(self):
        # trip ids corresponding to the given time range
        sql = f"""SELECT DISTINCT trip_id FROM trips_schedule 
        WHERE arrival>={self.start} AND departure<={self.end}"""
        self.trip_ids = pd.read_sql(
            sql=sql,
            con=self.pt_conn,
        ).trip_id.values

        # pattern ids corresponding to the given time range
        sql = f"""SELECT DISTINCT pattern_id FROM trips INNER JOIN 
        (SELECT DISTINCT trip_id FROM trips_schedule 
        WHERE departure>={self.start} AND arrival<={self.end}) selected_trips
        ON trips.trip_id = selected_trips.trip_id"""
        pattern_ids = pd.read_sql(
            sql=sql,
            con=self.pt_conn,
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
        self.compute_mean_travel_time()

    def compute_mean_travel_time(self):
        # Compute the travel time for each trip segment
        tt = pd.read_sql(sql="SELECT trip_id, seq, arrival, departure FROM trips_schedule", con=self.pt_conn)
        tt.sort_values(by=["trip_id", "seq"], ascending=True, inplace=True)
        tt["last_departure"] = tt["departure"].shift(1)
        tt["last_departure"] = tt["last_departure"].fillna(0.0)
        tt["travel_time_s"] = tt["arrival"] - tt["last_departure"]
        tt.loc[tt.seq == 0, "travel_time_s"] = 0.0
        tt.drop(["arrival", "departure", "last_departure"], axis=1, inplace=True)

        # tt.seq refers to the stop sequence index
        # Because we computed the travel time between two stops, we are now dealing
        # with a segment sequence index.
        tt = tt.loc[tt.seq > 0]
        tt.seq -= 1

        # Merge pattern ids with trip_id
        trips = pd.read_sql(sql="SELECT trip_id, pattern_id FROM trips", con=self.pt_conn)
        tt = pd.merge(tt, trips, on="trip_id", how="left")
        tt.drop("trip_id", axis=1, inplace=True)

        # Compute the mean travel time from the different trips corresponding to
        # a given pattern/segment couple
        tt = tt.groupby(["pattern_id", "seq"]).mean().reset_index(drop=False)
        self.line_segments = pd.merge(self.line_segments, tt, on=["pattern_id", "seq"], how="left")

    def create_stop_vertices(self):
        # select all stops
        self.stop_vertices = pd.read_sql(
            sql="SELECT stop_id, ST_AsText(geometry) coord, parent_station FROM stops", con=self.pt_conn
        )
        stops_ids = pd.concat((self.line_segments.from_stop, self.line_segments.to_stop), axis=0).unique()

        # filter stops that are used on the given time range
        self.stop_vertices = self.stop_vertices.loc[self.stop_vertices.stop_id.isin(stops_ids)]

        # add metadata
        self.stop_vertices["line_id"] = None
        self.stop_vertices["taz_id"] = None
        self.stop_vertices["line_seg_idx"] = np.nan
        self.stop_vertices["line_seg_idx"] = self.stop_vertices["line_seg_idx"].astype("Int32")
        self.stop_vertices["parent_station"] = self.stop_vertices["parent_station"].astype("Int32")
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
        sql = """SELECT node_id AS taz_id, ST_AsText(geometry) AS geometry FROM nodes WHERE is_centroid = 1"""
        self.od_vertices = pd.read_sql(sql=sql, con=self.proj_conn)
        self.od_vertices["type"] = "od"
        self.od_vertices["stop_id"] = None
        self.od_vertices["line_id"] = None
        self.od_vertices["line_seg_idx"] = np.nan
        self.od_vertices.rename(columns={"geometry": "coord"}, inplace=True)

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

    def create_on_board_edges(self):
        self.on_board_edges = self.line_segments[["line_id", "seq", "travel_time_s"]].copy(deep=True)
        self.on_board_edges.rename(columns={"seq": "line_seg_idx", "travel_time_s": "trav_time"}, inplace=True)

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
        pass

    def create_alighting_edges(self):
        pass

    def create_dwell_edges(self):
        pass

    def create_connector_edges(self):
        """Create the connector edges between each stop and the closest od."""
        transformer = pyproj.Transformer.from_crs(
            pyproj.CRS("EPSG:4326"), pyproj.CRS(self.local_crs), always_xy=True
        ).transform

        od_coords = self.od_vertices["coord"].apply(
            lambda coord: shapely.ops.transform(transformer, shapely.from_wkt(coord))
        )
        od_coords = np.array(list(od_coords.apply(lambda coord: (coord.x, coord.y))))

        stop_coords = self.stop_vertices["coord"].apply(
            lambda coord: shapely.ops.transform(transformer, shapely.from_wkt(coord))
        )
        stop_coords = np.array(list(stop_coords.apply(lambda coord: (coord.x, coord.y))))

        kdTree = cKDTree(od_coords)

        # query the kdTree for the closest (k=1) od for each stop in parallel (workers=-1)
        distance, index = kdTree.query(stop_coords, k=1, distance_upper_bound=np.inf, workers=self.num_threads)
        nearest_od = self.od_vertices.iloc[index]["node_id"].reset_index(drop=True)
        trav_time = pd.Series(distance * self.walking_speed, name="trav_time")
        self.connector_edges = pd.concat(
            [
                self.stop_vertices["stop_id"].reset_index(drop=True).rename("head_vert_id"),
                nearest_od.rename("tail_vert_id"),
                trav_time,
            ],
            axis=1,
        )

        self.connector_edges = pd.concat(
            [
                graph.connector_edges,
                graph.connector_edges.rename(
                    columns={"head_vert_id": "tail_vert_id", "tail_vert_id": "head_vert_id"}, copy=False
                ),
            ],
            copy=True,
        ).reset_index(drop=True)

        self.connector_edges["type"] = "connector"
        self.connector_edges["stop_id"] = None
        self.connector_edges["line_seg_idx"] = None
        self.connector_edges["freq"] = np.inf
        self.connector_edges["o_line_id"] = None
        self.connector_edges["d_line_id"] = None
        self.connector_edges["transfer_id"] = None

        return self.connector_edges

    def create_inner_stop_transfer_edges(self):
        stations = graph.stop_vertices[["stop_id", "parent_station"]]
        boarding = stations[stations["stop_id"].isin(graph.boarding_vertices["stop_id"])]
        alighting = stations[stations["stop_id"].isin(graph.alighting_vertices["stop_id"])]
        boarding_by_station = boarding.groupby(by="parent_station")
        alighting_by_station = alighting.groupby(by="parent_station")

        dwells = []
        transfers = []
        for station in np.intersect1d(boarding["parent_station"].values, alighting["parent_station"].values):
            boarding_stops = boarding_by_station.get_group(station)["stop_id"]
            alighting_stops = alighting_by_station.get_group(station)["stop_id"]

            for boarding_stop in boarding_stops:
                for alighting_stop in alighting_stops:
                    if boarding_stop == alighting_stop:
                        dwells.append([boarding_stop, alighting_stop, station])
                    else:
                        transfers.append([boarding_stop, alighting_stop, station])

        self.dwell_edges = pd.DataFrame(dwells, columns=["head_vert_id", "tail_vert_id", "station"])
        self.inner_transfer_edges = pd.DataFrame(transfers, columns=["head_vert_id", "tail_vert_id", "station"])
        # self.inner_transfer_edges["type"] = "transfer"
        # self.inner_transfer_edges["stop_id"] = None
        # self.inner_transfer_edges["line_seg_idx"] = None
        # # self.inner_transfer_edges["freq"] = np.inf
        # self.inner_transfer_edges["o_line_id"] = None
        # self.inner_transfer_edges["d_line_id"] = None
        # self.inner_transfer_edges["transfer_id"] = None

        return self.inner_transfer_edges, self.dwell_edges

    def create_outer_stop_transfer_edges(self):
        pass

    def create_transfer_edges(self):
        pass

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
