"""Create the graph used by public transport assignment algorithms.

Naming Conventions:
- a_node/b_node is head/tail vertex

TransitGraphBuilder Assumptions:
- opposite directions are not supported. In the GTFS files, this corresponds to direction_id from trips.txt (indicates the direction of travel for a trip),
- all times are expressed in seconds [s], all frequencies in [1/s], and
- headways are uniform for trips of the same pattern.

"""

import warnings

import numpy as np
import pandas as pd
import pyproj
import shapely
import shapely.ops
import json
import sqlite3

from aequilibrae.utils.geo_utils import haversine
from aequilibrae.project.database_connection import database_connection
from scipy.spatial import KDTree, minkowski_distance
from shapely.geometry import Point

from aequilibrae.paths import PathResults
from aequilibrae.context import get_active_project
from aequilibrae.paths import TransitGraph

SF_VERTEX_COLS = ["node_id", "node_type", "stop_id", "line_id", "line_seg_idx", "taz_id", "geometry"]
SF_EDGE_COLS = [
    "link_id",
    "link_type",
    "line_id",
    "stop_id",
    "line_seg_idx",
    "b_node",
    "a_node",
    "trav_time",
    "freq",
    "o_line_id",
    "d_line_id",
    "direction",
]


class TransitGraphBuilder:
    """Graph builder for the transit assignment Spiess & Florian algorithm.

    :Arguments:
        **public_transport_conn** (:obj:`sqlite3.Connection`): Connection to the ``public_transport.sqlite`` database.

        **period_id** (:obj:`int`): Period id for the period to be used. Preferred over start and end.

        **time_margin** (:obj:`int`): Time margin, extends the ``start`` and ``end`` times by ``time_margin`` ([``start``, ``end``] becomes [``start`` - ``time_margin``, ``end`` + ``time_margin``]), in order to include more trips when computing mean values. Defaults to ``0``.

        **projected_crs** (:obj:`str`): Projected CRS of the network, intended for more accurate distance calculations. Defaults to ``"EPSG:3857"``, Spherical Mercator.

        **num_threads** (:obj:`int`): Number of threads to be used where possible. Defaults to ``-1``, using all available threads.

        **seed** (:obj:`int`): Deprecated. No longer in use.

        **geometry_noise** (:obj:`bool`): Deprecated. No longer in use.

        **noise_coef** (:obj:`float`): Deprecated. No longer in use.

        **with_inner_stop_transfers** (:obj:`bool`): Whether to create transfer edges within parent stations. Defaults to ``False``.

        **with_outer_stop_transfers** (:obj:`bool`): Whether to create transfer edges between parent stations. Defaults to ``False``.

        **with_walking_edges** (:obj:`bool`): Whether to create walking edges between distinct stops of each station. Defaults to ``True``.

        **distance_upper_bound** (:obj:`float`): Upper bound on connector distance. Defaults to ``np.inf``.

        **blocking_centroid_flows** (:obj:`bool`): Whether to block flow through centroids. Defaults to ``True``.

        **max_connectors_per_zone** (:obj:`int`): Maximum connectors per zone. Defaults to ``-1`` for unlimited.
    """

    def __init__(
        self,
        public_transport_conn: sqlite3.Connection,
        period_id: int = 1,
        time_margin: int = 0,
        projected_crs: str = "EPSG:3857",
        num_threads: int = -1,
        seed: int = None,
        geometry_noise: bool = None,
        noise_coef: float = None,
        with_inner_stop_transfers: bool = False,
        with_outer_stop_transfers: bool = False,
        with_walking_edges: bool = True,
        distance_upper_bound: float = np.inf,
        blocking_centroid_flows: bool = True,
        connector_method: str = "nearest_neighbour",
        max_connectors_per_zone: int = -1,
    ):
        self.pt_conn = public_transport_conn
        self.project_conn: sqlite3.Connection = database_connection("project_database")

        self.period_id = period_id
        start, end = self.project_conn.execute(
            "SELECT period_start, period_end FROM periods WHERE period_id = ?;", [period_id]
        ).fetchall()[0]

        self.start = start - time_margin  # starting time of the selected time period
        self.end = end + time_margin  # ending time of the selected time period
        self.num_threads = num_threads

        # graph components
        # ----------------
        self.vertices = None
        self.edges = None
        self.__line_segments = None
        self.od_node_mapping = None

        # vertices
        self.__stop_vertices = None
        self.__boarding_vertices = None
        self.__alighting_vertices = None
        self.__od_vertices = None

        # edges
        self.__on_board_edges = None
        self.__dwell_edges = None
        self.__alighting_edges = None
        self.__boarding_edges = None
        self.__connector_edges = None
        self.__inner_stop_transfer_edges = pd.DataFrame()
        self.__outer_stop_transfer_edges = pd.DataFrame()
        self.__walking_edges = pd.DataFrame()

        self.global_crs = "EPSG:4326"
        self.__global_crs = pyproj.CRS(self.global_crs)
        self.projected_crs = projected_crs
        self.__projected_crs = pyproj.CRS(self.projected_crs)

        # longlat to projected CRS transformer
        self.transformer_g_to_p = pyproj.Transformer.from_crs(
            self.__global_crs, self.__projected_crs, always_xy=True
        ).transform

        self.transformer_p_to_g = pyproj.Transformer.from_crs(
            self.__projected_crs, self.__global_crs, always_xy=True
        ).transform

        if seed is not None or geometry_noise is not None or noise_coef is not None:
            warnings.warn(
                "the 'seed', 'geometry_noise', and 'noise_coef' arguments are depreciated and no longer in use. "
                "Duplicate geometries are allowed within the public transport database.",
                DeprecationWarning,
            )

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
        self.blocking_centroid_flows = blocking_centroid_flows
        self.connector_method = connector_method
        self.max_connectors_per_zone = max_connectors_per_zone

        self.__config_attrs = [
            "period_id",
            "projected_crs",
            # "uniform_dwell_time",
            # "alighting_penalty",
            # "a_tiny_time_duration",
            # "wait_time_factor",
            # "walk_time_factor",
            # "walking_speed",
            # "access_time_factor",
            # "egress_time_factor",
            "with_inner_stop_transfers",
            "with_outer_stop_transfers",
            "with_walking_edges",
            "distance_upper_bound",
            "blocking_centroid_flows",
            "connector_method",
            "max_connectors_per_zone",
        ]

    def add_zones(self, zones, from_crs: str = None):
        """Add zones as ODs.

        :Arguments:
            **zones** (:obj:`pd.DataFrame`): DataFrame containing the zoning information. Columns must include ``zone_id`` and ``geometry``.

            **from_crs** (:obj:`str`): The CRS of the ``geometry`` column of ``zones``. If not provided it's assumed that the geometry is already in ``self.projected_crs``. If provided, the geometry will be projected to ``self.projected_crs``. Defaults to ``None``.
        """
        if "zone_id" not in zones.columns or "geometry" not in zones.columns:
            raise KeyError("zone_id and geometry must be columns in zones")

        if zones.geometry.dtype is str or zones.geometry.dtype is bytes:
            geometry = shapely.from_wkt(zones.geometry.values)
        # Check if the supplied zones df is from geopandas without import geopandas.
        # We check __mro__ incase of inheritance. https://stackoverflow.com/a/63337375/14047443
        elif "GeometryDtype" in [t.__name__ for t in type(zones.geometry.dtype).__mro__] or all(
            isinstance(x, shapely.geometry.base.BaseGeometry) for x in zones.geometry
        ):
            geometry = zones.geometry.values
        else:
            raise TypeError("geometry is not a string, bytes, or shapely.Geometry instance")

        if from_crs:
            transformer = pyproj.Transformer.from_crs(
                pyproj.CRS(from_crs), self.__projected_crs, always_xy=True
            ).transform
        else:
            transformer = self.transformer_g_to_p
        geometry = [shapely.ops.transform(transformer, p) for p in geometry]
        centroids = shapely.centroid(geometry)

        self.zones = pd.DataFrame(
            {
                "zone_id": zones.zone_id.copy(deep=True).astype(str),
                "geometry": shapely.to_wkb([shapely.ops.transform(self.transformer_p_to_g, p) for p in geometry]),
                "centroids": shapely.to_wkb([shapely.ops.transform(self.transformer_p_to_g, p) for p in centroids]),
            }
        )

    def _create_line_segments(self):
        """Line segments correspond to segments between two successive stops for each line.

        For example if 2 lines, L1 and L2, are going from stop A to stop B, we have 2 line segments:
        - L1_AB
        - L2_AB

        line_segments table format:
            pattern_id  seq    from_stop      to_stop shortname         line_id  trav_time  headway      freq
        0  10001006000    0  10000000464  10000000462        T2  T2_10001006000      150.0    240.0  0.004167
        1  10001006000    1  10000000462  10000000459        T2  T2_10001006000      110.0    240.0  0.004167
        2  10001006000    2  10000000459  10000000160        T2  T2_10001006000      100.0    240.0  0.004167
        """

        # we select route links for the pattern_ids in the given time range
        sql = f"""SELECT distinct
            trips.pattern_id,
            route_links.seq,
            CAST(from_stop AS TEXT) from_stop,
            CAST(to_stop AS TEXT) to_stop
        FROM
            route_links
        INNER JOIN trips ON route_links.pattern_id = trips.pattern_id
        INNER JOIN trips_schedule ON trips.trip_id = trips_schedule.trip_id
        WHERE
            departure>={self.start}
            AND arrival<={self.end}"""
        route_links = pd.read_sql(
            sql=sql,
            con=self.pt_conn,
        )

        # create a routes table
        sql = "SELECT pattern_id, CAST(shortname AS TEXT) shortname FROM routes"
        routes = pd.read_sql(
            sql=sql,
            con=self.pt_conn,
        )
        # we create a line id by concatenating the route short name with the pattern_id
        routes["line_id"] = routes["shortname"] + "_" + routes["pattern_id"].astype(str)

        # we create the line segments by merging the route links with the routes
        # in order to have a route name on each segment
        self.__line_segments = pd.merge(route_links, routes, on="pattern_id", how="left")

        # we add the travel time and headway to each line segment
        self._add_mean_travel_time_to_segments()
        self._add_mean_headway_to_segments()

        # we compute the frequency from the headway
        self.__line_segments["freq"] = np.inf
        self.__line_segments.loc[self.__line_segments.headway > 0.0, "freq"] = (
            1.0 / self.__line_segments.loc[self.__line_segments.headway > 0.0, "headway"]
        )

    def _compute_segment_travel_time(self, time_filter=True):
        """Compute the mean travel time for each line segment.

        tt table format:
            pattern_id  seq     trav_time
        0   10001006000     1   114.545455
        1   10001006000     2   100.000000
        2   10001006000     3   180.000000

        :Arguments:
           **time_filter** (:obj:`bool`): If time_filter is True, the mean travel time is computed over the
           [start, end] time range, otherwise it is computed over all the available data (e.g. a whole day).
           Defaults to ``True``.

        :Returns:
           **tt** (:obj:`pd.DataFrame`): DataFrame containing the travel item for line segments.
        """

        if time_filter:
            sql = f"""SELECT trips_schedule.trip_id, trips_schedule.seq, trips_schedule.arrival,
                trips_schedule.departure, trips.pattern_id FROM trips_schedule LEFT JOIN trips
                ON trips_schedule.trip_id = trips.trip_id
                WHERE trips_schedule.departure>={self.start} AND trips_schedule.arrival<={self.end}"""
        else:
            sql = """SELECT trips_schedule.trip_id, trips_schedule.seq, trips_schedule.arrival,
                trips_schedule.departure, trips.pattern_id FROM trips_schedule LEFT JOIN trips
                ON trips_schedule.trip_id = trips.trip_id"""
        tt = pd.read_sql(sql=sql, con=self.pt_conn)

        # compute the travel time on the segments
        tt.sort_values(by=["pattern_id", "trip_id", "seq"], ascending=True, inplace=True)
        tt["last_departure"] = tt["departure"].shift(+1)
        tt["last_trip_id"] = tt["trip_id"].shift(+1)
        tt["last_pattern_id"] = tt["pattern_id"].shift(+1)
        tt["trav_time"] = tt["arrival"] - tt["last_departure"]
        tt.dropna(how="any", inplace=True)
        tt[["last_departure", "last_trip_id", "last_pattern_id"]] = tt[
            ["last_departure", "last_trip_id", "last_pattern_id"]
        ].astype(np.uint64)

        tt.loc[(tt.last_pattern_id != tt.pattern_id) | (tt.last_trip_id != tt.trip_id), "trav_time"] = np.nan
        tt.dropna(subset="trav_time", inplace=True)
        tt = tt.copy(deep=True)

        # tt.seq refers to the stop sequence index.
        # Because we computed the travel time between two stops, we are now dealing
        # with a segment sequence index.
        tt = tt.loc[tt.seq > 0]
        tt.seq -= 1

        # take the mean of the travel times computed among the trips of a pattern segment
        tt = tt[["pattern_id", "seq", "trav_time"]].groupby(["pattern_id", "seq"]).mean().reset_index(drop=False)

        return tt

    def _add_mean_travel_time_to_segments(self):
        """Add a column to the line segment table with a mean travel time.

        Because there might be missing values when computing the travel time on a small time range,
        we also compute the mean travel time over all the available data in order to fill these potential
        missing values.
        """
        tt = self._compute_segment_travel_time(time_filter=True)
        tt_full = self._compute_segment_travel_time(time_filter=False)
        tt_full.rename(columns={"trav_time": "trav_time_full"}, inplace=True)

        # Compute the mean travel time from the different trips corresponding to
        self.__line_segments = pd.merge(self.__line_segments, tt, on=["pattern_id", "seq"], how="left")
        self.__line_segments = pd.merge(self.__line_segments, tt_full, on=["pattern_id", "seq"], how="left")
        self.__line_segments.trav_time = self.__line_segments.trav_time.fillna(self.__line_segments.trav_time_full)
        self.__line_segments.drop("trav_time_full", axis=1, inplace=True)
        self.__line_segments.trav_time = self.__line_segments.trav_time.fillna(self.end - self.start)

    def _add_mean_headway_to_segments(self):
        """Compute the mean headway for each pattern and add it to the line segment table, as headway.

        When there is not enough information to compute the mean headway (e.g. a single trip), the headway is
        given the value of the time range length.
        """
        # start from the trip_schedule table
        sql = f"""SELECT trip_id, seq, arrival FROM trips_schedule
            WHERE departure>={self.start} AND arrival<={self.end}"""
        mh = pd.read_sql(sql=sql, con=self.pt_conn)

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
        mh["headway"] = mh["headway"].bfill()

        # take the min of the headways computed among the stops of a given trip
        mh = mh[["pattern_id", "trip_id", "headway"]].groupby("pattern_id").min().reset_index(drop=False)

        # compute the mean headway computed among the trips of a given pattern
        mh = mh[["pattern_id", "headway"]].groupby("pattern_id").mean().reset_index(drop=False)

        self.__line_segments = pd.merge(self.__line_segments, mh, on=["pattern_id"], how="left")

    def _create_stop_vertices(self):
        """Create stop vertices."""
        # select all stops
        sql = "SELECT CAST(stop_id AS TEXT) stop_id, ST_AsBinary(geometry) AS geometry FROM stops"
        stop_vertices = pd.read_sql(sql=sql, con=self.pt_conn)

        # filter stops that are used on the given time range
        stops_ids = pd.concat((self.__line_segments.from_stop, self.__line_segments.to_stop), axis=0).unique()
        stop_vertices = stop_vertices.loc[stop_vertices.stop_id.isin(stops_ids)]

        # uniform attributes
        stop_vertices["line_seg_idx"] = -1
        stop_vertices["node_type"] = "stop"

        self.__stop_vertices = stop_vertices

    def _create_boarding_vertices(self):
        """Create boarding vertices."""
        boarding_vertices = self.__line_segments[["line_id", "seq", "from_stop"]].copy(deep=True)
        boarding_vertices.rename(columns={"seq": "line_seg_idx", "from_stop": "stop_id"}, inplace=True)
        boarding_vertices = pd.merge(
            boarding_vertices, self.__stop_vertices[["stop_id", "geometry"]], on="stop_id", how="left"
        )

        # uniform attributes
        boarding_vertices["node_type"] = "boarding"

        self.__boarding_vertices = boarding_vertices

    def _create_alighting_vertices(self):
        """Create alighting vertices."""
        alighting_vertices = self.__line_segments[["line_id", "seq", "to_stop"]].copy(deep=True)
        alighting_vertices.rename(columns={"seq": "line_seg_idx", "to_stop": "stop_id"}, inplace=True)
        alighting_vertices = pd.merge(
            alighting_vertices, self.__stop_vertices[["stop_id", "geometry"]], on="stop_id", how="left"
        )

        # uniform attributes
        alighting_vertices["node_type"] = "alighting"

        self.__alighting_vertices = alighting_vertices

    def _create_od_vertices(self):
        """Create OD vertices from zones.

        If zones have not previously been added, add zones from the project.
        If ``self.blocking_centroid_flow`` is ``True``, a distinction is made between ``origin`` and ``destination`` vertices. Otherwise, they are both classified as ``od``.
        """
        if "zones" not in self.__dict__:
            project = get_active_project(True)
            self.add_zones(
                pd.DataFrame(
                    [(x.zone_id, x.geometry) for x in project.zoning.all_zones().values()],
                    columns=["zone_id", "geometry"],
                )
            )

        if self.blocking_centroid_flows:
            # we create both "origin" and "destination" nodes
            origin_vertices = self.zones[["zone_id", "centroids"]].rename(
                columns={"zone_id": "taz_id", "centroids": "geometry"}
            )

            # uniform attributes
            origin_vertices["node_type"] = "origin"
            origin_vertices["line_seg_idx"] = -1

            destination_vertices = origin_vertices.copy(deep=True)
            destination_vertices["node_type"] = "destination"

            od_vertices = pd.concat((origin_vertices, destination_vertices), axis=0)
        else:
            # we create only "od" nodes
            od_vertices = self.zones[["zone_id", "centroids"]].rename(
                columns={"zone_id": "taz_id", "centroids": "geometry"}
            )

            # uniform attributes
            od_vertices["node_type"] = "od"
            od_vertices["line_seg_idx"] = -1

        self.__od_vertices = od_vertices

    def create_od_node_mapping(self):
        """Build a dataframe mapping the centroid node ids with to transport assignment zone ids."""
        if self.blocking_centroid_flows:
            origin_nodes = self.vertices.loc[
                self.vertices.node_type == "origin",
                ["node_id", "taz_id"],
            ]
            origin_nodes.rename(columns={"node_id": "o_node_id"}, inplace=True)
            destination_nodes = self.vertices.loc[
                self.vertices.node_type == "destination",
                ["node_id", "taz_id"],
            ]
            destination_nodes.rename(columns={"node_id": "d_node_id"}, inplace=True)
            od_node_mapping = pd.merge(origin_nodes, destination_nodes, on="taz_id", how="left")[
                ["o_node_id", "d_node_id", "taz_id"]
            ]
        else:
            od_node_mapping = self.vertices.loc[
                self.vertices.node_type == "od",
                ["node_id", "taz_id"],
            ]
        self.od_node_mapping = od_node_mapping

    def _create_vertices(self):
        """Graph vertices creation as a dataframe.

        Vertices have the following attributes:
            - node_id (:obj:`int`),
            - node_type (:obj:`str`): Either 'stop', 'boarding', 'alighting', 'od', 'origin', or 'destination',
            - stop_id (:obj:`str`): Only applies to 'stop', 'boarding' and 'alighting' vertices,
            - line_id (:obj:`str`): Only applies to 'boarding' and 'alighting' vertices,
            - line_seg_idx (:obj:`int`): Only applies to 'boarding' and 'alighting' vertices,
            - taz_id (:obj:`str`): Only applies to 'origin', 'destination', and 'od' nodes,
            - geometry (:obj:`str`): Geometry object in WKB (well known binary).
        """
        self._create_line_segments()
        self._create_stop_vertices()
        self._create_boarding_vertices()
        self._create_alighting_vertices()
        self._create_od_vertices()

        # stack the dataframes on top of each other
        self.vertices = pd.concat(
            [
                self.__od_vertices,
                self.__stop_vertices,
                self.__boarding_vertices,
                self.__alighting_vertices,
            ],
            axis=0,
        )

        # reset index and copy it to column
        self.vertices.reset_index(drop=True, inplace=True)
        self.vertices["node_id"] = self.vertices.index + 1
        self.vertices = self.vertices[SF_VERTEX_COLS]
        self.create_od_node_mapping()

        # data types
        self.vertices.node_id = self.vertices.node_id.astype("int64")
        self.vertices.stop_id = self.vertices.stop_id.fillna("").astype(str)
        self.vertices.line_id = self.vertices.line_id.fillna("").astype(str)
        self.vertices.line_seg_idx = self.vertices.line_seg_idx.fillna(-1).astype("int64")
        self.vertices.taz_id = self.vertices.taz_id.fillna("").astype(str)

    def _create_on_board_edges(self):
        """Create on board edges."""
        on_board_edges = self.__line_segments[["line_id", "seq", "trav_time"]].copy(deep=True)
        on_board_edges.rename(columns={"seq": "line_seg_idx"}, inplace=True)

        # get tail vertex index
        on_board_edges = pd.merge(
            on_board_edges,
            self.vertices[self.vertices.node_type == "boarding"][["line_id", "line_seg_idx", "node_id"]],
            on=["line_id", "line_seg_idx"],
            how="left",
        )
        on_board_edges.rename(columns={"node_id": "b_node"}, inplace=True)

        # get head vertex index
        on_board_edges = pd.merge(
            on_board_edges,
            self.vertices[self.vertices.node_type == "alighting"][["line_id", "line_seg_idx", "node_id"]],
            on=["line_id", "line_seg_idx"],
            how="left",
        )
        on_board_edges.rename(columns={"node_id": "a_node"}, inplace=True)

        # uniform attributes
        on_board_edges["link_type"] = "on-board"
        on_board_edges["freq"] = np.inf
        on_board_edges["direction"] = 1

        self.__on_board_edges = on_board_edges

    def _create_boarding_edges(self):
        """Create boarding edges."""
        boarding_edges = self.__line_segments[["line_id", "seq", "from_stop", "freq"]].copy(deep=True)
        boarding_edges.rename(columns={"seq": "line_seg_idx", "from_stop": "stop_id"}, inplace=True)

        # get tail vertex index (stop vertex)
        boarding_edges = pd.merge(
            boarding_edges,
            self.vertices[self.vertices.node_type == "stop"][["stop_id", "node_id"]],
            on="stop_id",
            how="left",
        )
        boarding_edges.rename(columns={"node_id": "b_node"}, inplace=True)

        # get head vertex index (boarding vertex)
        boarding_edges = pd.merge(
            boarding_edges,
            self.vertices[self.vertices.node_type == "boarding"][["line_id", "line_seg_idx", "node_id"]],
            on=["line_id", "line_seg_idx"],
            how="left",
        )
        boarding_edges.rename(columns={"node_id": "a_node"}, inplace=True)

        # frequency update : line_freq / wait_time_factor
        wait_time_factor_inv = 1.0 / self.wait_time_factor
        boarding_edges["freq"] *= wait_time_factor_inv

        # uniform attributes
        boarding_edges["link_type"] = "boarding"
        boarding_edges["trav_time"] = 0.5 * self.uniform_dwell_time + self.a_tiny_time_duration
        boarding_edges["direction"] = 1

        self.__boarding_edges = boarding_edges

    def _create_alighting_edges(self):
        """Create alighting edges."""
        alighting_edges = self.__line_segments[["line_id", "seq", "to_stop"]].copy(deep=True)
        alighting_edges.rename(columns={"seq": "line_seg_idx", "to_stop": "stop_id"}, inplace=True)

        # get tail vertex index (alighting vertex)
        alighting_edges = pd.merge(
            alighting_edges,
            self.vertices[self.vertices.node_type == "alighting"][["line_id", "line_seg_idx", "node_id"]],
            on=["line_id", "line_seg_idx"],
            how="left",
        )
        alighting_edges.rename(columns={"node_id": "b_node"}, inplace=True)

        # get head vertex index (stop vertex)
        alighting_edges = pd.merge(
            alighting_edges,
            self.vertices[self.vertices.node_type == "stop"][["stop_id", "node_id"]],
            on="stop_id",
            how="left",
        )
        alighting_edges.rename(columns={"node_id": "a_node"}, inplace=True)

        # uniform attributes
        alighting_edges["link_type"] = "alighting"
        alighting_edges["freq"] = np.inf
        alighting_edges["trav_time"] = (
            0.5 * self.uniform_dwell_time + self.alighting_penalty + self.a_tiny_time_duration
        )
        alighting_edges["direction"] = 1

        self.__alighting_edges = alighting_edges

    def _create_dwell_edges(self):
        """Create dwell edges."""
        # we start by removing the first segment of each line
        dwell_edges = self.__line_segments.loc[self.__line_segments.seq != 0][["line_id", "from_stop", "seq"]]
        dwell_edges.rename(columns={"seq": "line_seg_idx"}, inplace=True)

        # we take the first stop of the segment
        dwell_edges["stop_id"] = dwell_edges.from_stop

        # head vertex index (boarding vertex)
        # boarding vertices of line segments [1:segment_count+1]
        dwell_edges = pd.merge(
            dwell_edges,
            self.vertices[self.vertices.node_type == "boarding"][["line_id", "stop_id", "node_id", "line_seg_idx"]],
            on=["line_id", "stop_id", "line_seg_idx"],
            how="left",
        )
        dwell_edges.rename(columns={"node_id": "a_node"}, inplace=True)

        # tail vertex index (alighting vertex)
        # aligthing vertices of line segments [0:segment_count]
        dwell_edges.line_seg_idx -= 1
        dwell_edges = pd.merge(
            dwell_edges,
            self.vertices[self.vertices.node_type == "alighting"][["line_id", "stop_id", "node_id", "line_seg_idx"]],
            on=["line_id", "stop_id", "line_seg_idx"],
            how="left",
        )
        dwell_edges.rename(columns={"node_id": "b_node"}, inplace=True)

        # clean-up
        dwell_edges.drop("from_stop", axis=1, inplace=True)

        # uniform values
        dwell_edges["line_seg_idx"] = -1
        dwell_edges["link_type"] = "dwell"
        dwell_edges["freq"] = np.inf
        dwell_edges["trav_time"] = self.uniform_dwell_time
        dwell_edges["direction"] = 1

        self.__dwell_edges = dwell_edges

    def _create_connector_edges(self, method=None, allow_missing_connections=True):
        """
        Create the connector edges between each stops and ODs.

        Nearest neighbour: Creates edges between every stop and its nearest OD.

        Overlapping regions: Creates edges between all stops that lying within the circle centred on each OD whose radius is the distance to the next nearest OD.

        :Arguments:
           **method** (:obj:`str`): Must either be "overlapping_regions", or "nearest_neighbour".
           Defaults to ``overlapping_regions``.

           **allow_missing_connections** (:obj:`bool`): Whether to allow missing connections or not.
           Defaults to ``True``.
        """
        if method is None:
            method = self.connector_method
        else:
            self.connector_method = method

        if method not in ["overlapping_regions", "nearest_neighbour"]:
            raise ValueError("method must be either 'overlapping_regions' or 'nearest_neighbour'")

        # Create access connectors
        # ========================

        # Select/copy the od vertices and project their geometry
        if self.blocking_centroid_flows:
            node_type = "origin"
        else:
            node_type = "od"
        od_vertices = self.vertices[self.vertices.node_type == node_type][["node_id", "taz_id", "geometry"]].copy(
            deep=True
        )

        od_vertices.reset_index(drop=True, inplace=True)
        od_geometries = od_vertices["geometry"].apply(
            lambda geometry: shapely.ops.transform(self.transformer_g_to_p, shapely.from_wkb(geometry))
        )
        od_geometries = np.array(list(od_geometries.apply(lambda geometry: (geometry.x, geometry.y))))

        # Select/copy the stop vertices and project their geometryinates
        stop_vertices = self.vertices[self.vertices.node_type == "stop"][["node_id", "stop_id", "geometry"]].copy(
            deep=True
        )
        stop_vertices.reset_index(drop=True, inplace=True)
        stop_geometries = stop_vertices["geometry"].apply(
            lambda geometry: shapely.ops.transform(self.transformer_g_to_p, shapely.from_wkb(geometry))
        )
        stop_geometries = np.array(list(stop_geometries.apply(lambda geometry: (geometry.x, geometry.y))))

        kdTree = KDTree(od_geometries)

        if method == "nearest_neighbour":
            # query the kdTree for the closest (k=1) od for each stop in parallel (workers=-1)
            distance, index = kdTree.query(
                stop_geometries, k=1, distance_upper_bound=self.distance_upper_bound, workers=self.num_threads
            )
            nearest_od = od_vertices.iloc[index][["node_id", "taz_id"]].reset_index(drop=True)
            trav_time = pd.Series(distance / self.walking_speed, name="trav_time")

            # access connectors
            access_connector_edges = pd.concat(
                [
                    stop_vertices[["stop_id", "node_id"]].reset_index(drop=True).rename(columns={"node_id": "a_node"}),
                    nearest_od.rename(columns={"node_id": "b_node"}),
                    trav_time,
                ],
                axis=1,
            )

        elif method == "overlapping_regions":
            # Construct a kdtree so we can lookup the 2nd closest OD to each OD (the first being itself)
            distance, _ = kdTree.query(
                od_geometries, k=[2], distance_upper_bound=self.distance_upper_bound, workers=self.num_threads
            )
            distance = distance.reshape(-1)

            # Construct a kdtree so we can query all the stops within the radius around each OD
            stop_kdTree = KDTree(stop_geometries)
            results = stop_kdTree.query_ball_point(od_geometries, distance, workers=self.num_threads)

            # access connectors

            # Build up a list of dataframes to concat, each dataframe corresponds to all connectors for a given OD
            connectors = []
            for i, verts in enumerate(results):
                distance = minkowski_distance(od_geometries[i], stop_geometries[verts])
                df = stop_vertices["node_id"].iloc[verts].to_frame()
                df["b_node"] = od_vertices.iloc[i]["node_id"]  # OD is tail node of access connector
                df["trav_time"] = distance / self.walking_speed
                connectors.append(df)
            access_connector_edges = pd.concat(connectors).rename(columns={"node_id": "a_node"}).reset_index(drop=True)

            if not allow_missing_connections:
                # Now we need to build up the edges for the stops without connectors
                missing = stop_vertices["node_id"].isin(access_connector_edges["a_node"])
                missing = missing[~missing].index

                distance, index = kdTree.query(
                    stop_geometries[missing], k=1, distance_upper_bound=np.inf, workers=self.num_threads
                )
                nearest_od = od_vertices["node_id"].iloc[index].reset_index(drop=True)
                trav_time = pd.Series(distance / self.walking_speed, name="trav_time")
                missing_edges = pd.concat(
                    [
                        stop_vertices["node_id"].iloc[missing].reset_index(drop=True).rename("a_node"),
                        nearest_od.rename("b_node"),
                        trav_time,
                    ],
                    axis=1,
                )

                access_connector_edges = pd.concat([access_connector_edges, missing_edges], axis=0)

        # uniform values
        access_connector_edges["link_type"] = "access_connector"
        access_connector_edges["line_seg_idx"] = -1
        access_connector_edges["freq"] = np.inf
        access_connector_edges["direction"] = 1

        if self.max_connectors_per_zone > 0:
            # max_connectors_per_zone connectors per zone with smallest travel time are selected
            access_connector_edges = access_connector_edges.sort_values(["b_node", "trav_time"])
            access_connector_edges = access_connector_edges.groupby("b_node").apply(
                lambda df: df.head(self.max_connectors_per_zone)
            )

        # Create egress connectors
        # ========================
        egress_connector_edges = access_connector_edges.copy(deep=True)
        egress_connector_edges.rename(columns={"a_node": "b_node", "b_node": "a_node"}, inplace=True)

        if self.blocking_centroid_flows:
            # we need to switch the head node to a destination node instead of an origin node
            egress_connector_edges = pd.merge(
                egress_connector_edges,
                self.od_node_mapping[["o_node_id", "d_node_id"]],
                left_on="a_node",
                right_on="o_node_id",
                how="left",
            )
            egress_connector_edges.drop(["a_node", "o_node_id"], axis=1, inplace=True)
            egress_connector_edges.rename(columns={"d_node_id": "a_node"}, inplace=True)

        # uniform values
        egress_connector_edges["link_type"] = "egress_connector"
        egress_connector_edges["line_seg_idx"] = -1
        egress_connector_edges["freq"] = np.inf
        egress_connector_edges["direction"] = 1

        # travel time update
        access_connector_edges.trav_time *= self.access_time_factor
        egress_connector_edges.trav_time *= self.egress_time_factor

        self.__connector_edges = pd.concat([access_connector_edges, egress_connector_edges], axis=0)

    def _create_inner_stop_transfer_edges(self):
        """Create transfer edges between distinct lines of each stop."""
        alighting = self.vertices[self.vertices.node_type == "alighting"][["stop_id", "line_id", "node_id"]].rename(
            columns={"line_id": "o_line_id", "node_id": "b_node"}
        )
        boarding = self.vertices[self.vertices.node_type == "boarding"][["stop_id", "line_id", "node_id"]].rename(
            columns={"line_id": "d_line_id", "node_id": "a_node"}
        )
        inner_stop_transfer_edges = pd.merge(alighting, boarding, on="stop_id", how="inner")

        # remove entries that have the same line as origin and destination
        inner_stop_transfer_edges = inner_stop_transfer_edges.loc[
            inner_stop_transfer_edges["o_line_id"] != inner_stop_transfer_edges["d_line_id"]
        ]

        # update the transfer edge frequency
        inner_stop_transfer_edges = pd.merge(
            inner_stop_transfer_edges,
            self.__line_segments[["from_stop", "line_id", "freq"]],
            left_on=["stop_id", "d_line_id"],
            right_on=["from_stop", "line_id"],
            how="left",
        )
        inner_stop_transfer_edges.drop(["from_stop", "line_id"], axis=1, inplace=True)

        # uniform attributes
        inner_stop_transfer_edges["line_seg_idx"] = -1
        inner_stop_transfer_edges["link_type"] = "inner_transfer"
        inner_stop_transfer_edges["direction"] = 1

        # frequency update : line_freq / wait_time_factor
        wait_time_factor_inv = 1.0 / self.wait_time_factor
        inner_stop_transfer_edges["freq"] *= wait_time_factor_inv

        # travel time update : dwell_time + alighting_penalty
        inner_stop_transfer_edges["trav_time"] = self.uniform_dwell_time + self.alighting_penalty

        # remove duplicates
        inner_stop_transfer_edges.drop_duplicates(inplace=True)

        self.__inner_stop_transfer_edges = inner_stop_transfer_edges

    def _create_outer_stop_transfer_edges(self):
        """Create transfer edges between distinct lines/stops of each station."""
        sql = """
        SELECT CAST(stop_id as TEXT) stop_id, CAST(parent_station as TEXT) parent_station FROM stops
        WHERE parent_station IS NOT NULL AND parent_station <> ''
        """
        stops = pd.read_sql(sql=sql, con=self.pt_conn)
        stations = stops.groupby("parent_station").size().to_frame("stop_count").reset_index(drop=False)

        # we only keep the stations which contain at least 2 stops
        stations = stations[stations["stop_count"] > 1]
        station_list = stations["parent_station"].values
        stops = stops[stops.parent_station.isin(station_list)]

        # load the aligthing vertices (tail of transfer edges)
        alighting = self.vertices[self.vertices.node_type == "alighting"][
            ["stop_id", "line_id", "node_id", "geometry"]
        ].rename(columns={"line_id": "o_line_id", "geometry": "o_geometry", "node_id": "b_node"})
        # add the station id
        alighting = pd.merge(alighting, stops, on="stop_id", how="inner")
        alighting.rename(columns={"stop_id": "o_stop_id"}, inplace=True)

        # load the boarding vertices (head of transfer edges)
        boarding = self.vertices[self.vertices.node_type == "boarding"][
            ["stop_id", "line_id", "node_id", "geometry"]
        ].rename(columns={"line_id": "d_line_id", "geometry": "d_geometry", "node_id": "a_node"})
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
            self.__line_segments[["from_stop", "line_id", "freq"]],
            left_on=["d_stop_id", "d_line_id"],
            right_on=["from_stop", "line_id"],
            how="left",
        )
        outer_stop_transfer_edges.drop(["o_stop_id", "d_stop_id", "from_stop", "line_id"], axis=1, inplace=True)

        # uniform attributes
        outer_stop_transfer_edges["line_seg_idx"] = -1
        outer_stop_transfer_edges["link_type"] = "outer_transfer"

        # frequency update : line_freq / wait_time_factor
        wait_time_factor_inv = 1.0 / self.wait_time_factor
        outer_stop_transfer_edges["freq"] *= wait_time_factor_inv

        # compute the walking time
        o_geometry = shapely.from_wkb(outer_stop_transfer_edges.o_geometry.values)
        d_geometry = shapely.from_wkb(outer_stop_transfer_edges.d_geometry.values)
        o_lon, o_lat = np.vectorize(lambda x: (x.x, x.y))(o_geometry)
        d_lon, d_lat = np.vectorize(lambda x: (x.x, x.y))(d_geometry)

        distance = haversine(o_lon, o_lat, d_lon, d_lat)

        outer_stop_transfer_edges["trav_time"] = distance / self.walking_speed
        outer_stop_transfer_edges["trav_time"] *= self.walk_time_factor
        outer_stop_transfer_edges["trav_time"] += self.alighting_penalty
        outer_stop_transfer_edges["direction"] = 1

        # cleanup
        outer_stop_transfer_edges.drop(
            ["o_geometry", "d_geometry"],
            axis=1,
            inplace=True,
        )

        # remove duplicates
        outer_stop_transfer_edges.drop_duplicates(inplace=True)

        self.__outer_stop_transfer_edges = outer_stop_transfer_edges

    def _create_walking_edges(self):
        """Create walking edges between distinct stops of each station."""

        sql = "SELECT COUNT(*) FROM stops WHERE parent_station IS NOT NULL AND parent_station <> ''"
        station_count = self.pt_conn.execute(sql).fetchone()[0]

        if station_count > 0:
            sql = """
            SELECT CAST(stop_id AS TEXT) stop_id, CAST(parent_station AS TEXT) parent_station FROM stops
            WHERE parent_station IS NOT NULL AND parent_station <> ''
            """
            stops = pd.read_sql(sql=sql, con=self.pt_conn)

            print(stops)
            stops.drop_duplicates(inplace=True)
            stations = stops.groupby("parent_station").size().to_frame("stop_count").reset_index(drop=False)

            # we only keep the stations which contain at least 2 stops
            stations = stations[stations["stop_count"] > 1]
            station_list = stations["parent_station"].values
            stops = stops[stops.parent_station.isin(station_list)]

            # tail vertex
            o_walking = self.vertices[self.vertices.node_type == "stop"][["stop_id", "node_id", "geometry"]].rename(
                columns={"geometry": "o_geometry", "node_id": "b_node"}
            )
            o_walking = pd.merge(o_walking, stops, on="stop_id", how="inner")
            o_walking.rename(columns={"stop_id": "o_stop_id"}, inplace=True)

            # head vertex
            d_walking = self.vertices[self.vertices.node_type == "stop"][["stop_id", "node_id", "geometry"]].rename(
                columns={"geometry": "d_geometry", "node_id": "a_node"}
            )
            d_walking = pd.merge(d_walking, stops, on="stop_id", how="inner")
            d_walking.rename(columns={"stop_id": "d_stop_id"}, inplace=True)

            walking_edges = pd.merge(o_walking, d_walking, on="parent_station", how="inner")

            # remove entries that share the same stop
            walking_edges = walking_edges.loc[walking_edges["o_stop_id"] != walking_edges["d_stop_id"]]
            walking_edges.drop("parent_station", axis=1, inplace=True)

            # uniform attributes
            walking_edges["line_seg_idx"] = -1
            walking_edges["link_type"] = "walking"
            walking_edges["freq"] = np.inf
            walking_edges["direction"] = 1

            # compute the walking time
            o_geometry = shapely.from_wkb(walking_edges.o_geometry.values)
            d_geometry = shapely.from_wkb(walking_edges.d_geometry.values)
            o_lon, o_lat = np.vectorize(lambda x: (x.x, x.y))(o_geometry)
            d_lon, d_lat = np.vectorize(lambda x: (x.x, x.y))(d_geometry)

            distance = haversine(o_lon, o_lat, d_lon, d_lat)

            walking_edges["trav_time"] = distance / self.walking_speed
            walking_edges["trav_time"] *= self.walk_time_factor
            walking_edges.loc[walking_edges["trav_time"] < self.a_tiny_time_duration, "trav_time"] = (
                self.a_tiny_time_duration
            )

            # cleanup
            walking_edges.drop(
                ["o_geometry", "d_geometry"],
                axis=1,
                inplace=True,
            )

            self.__walking_edges = walking_edges

    def _create_edges(self):
        """Graph edges creation as a Dataframe.

        Edges have the following attributes:
            - type (:obj:`str`): Either 'on-board', 'boarding', 'alighting', 'dwell', 'inner_transfer', 'outer_transfer',
              'access_connector', "egress_connector" or 'walking',
            - line_id (:obj:`str`): Only applies to 'on-board', 'boarding', 'alighting' and 'dwell' edges,
            - stop_id (:obj:`str`),
            - line_seg_idx (:obj:`int`): Only applies to 'on-board', 'boarding' and 'alighting' edges,
            - b_node (:obj:`int`),
            - a_node (:obj:`int`),
            - trav_time (:obj:`float`): Edge travel time,
            - freq (:obj:`float`),
            - o_line_id (:obj:`str`),
            - d_line_id (:obj:`str`),
            - transfer_id (:obj:`str`)
        """
        # create the graph edges
        self._create_on_board_edges()
        self._create_dwell_edges()
        self._create_boarding_edges()
        self._create_alighting_edges()
        self._create_connector_edges()
        if self.with_inner_stop_transfers:
            self._create_inner_stop_transfer_edges()
        if self.with_outer_stop_transfers:
            self._create_outer_stop_transfer_edges()
        if self.with_walking_edges:
            self._create_walking_edges()

        # stack the dataframes on top of each other
        self.edges = pd.concat(
            [
                self.__on_board_edges,
                self.__boarding_edges,
                self.__alighting_edges,
                self.__dwell_edges,
                self.__connector_edges,
                self.__inner_stop_transfer_edges,
                self.__outer_stop_transfer_edges,
                self.__walking_edges,
            ],
            axis=0,
        )

        # reset index and copy it to column
        self.edges.reset_index(drop=True, inplace=True)
        self.edges["link_id"] = self.edges.index + 1
        for col in SF_EDGE_COLS:
            if col not in self.edges:
                self.edges[col] = np.nan
        self.edges = self.edges[SF_EDGE_COLS]

        # data types
        self.edges.line_id = self.edges.line_id.fillna("").astype(str)
        self.edges.stop_id = self.edges.stop_id.fillna("").astype(str)
        self.edges.line_seg_idx = self.edges.line_seg_idx.fillna(-1).astype("int64")
        self.edges.b_node = self.edges.b_node.astype("int64")
        self.edges.a_node = self.edges.a_node.astype("int64")
        self.edges.trav_time = self.edges.trav_time.astype(float)
        self.edges.freq = self.edges.freq.astype(float)
        self.edges.o_line_id = self.edges.o_line_id.fillna("").astype(str)
        self.edges.d_line_id = self.edges.d_line_id.fillna("").astype(str)
        self.edges.direction = self.edges.direction.astype("int8")

    def create_graph(self):
        """Create the SF transit graph (vertices and edges)."""
        self._create_vertices()
        self._create_edges()

    def create_line_geometry(self, method="direct", graph="w"):
        """
        Create the LineString for each edge.

        The direct method creates a straight line between all points.

        The connect project match method uses the existing line geometry within the project to create more
        accurate line strings. It creates a line string that matches the path between the shortest path
        between the project nodes closest to either end of the access and egress connectors.

        Project graphs must be built for the "connector project match" method.

        :Arguments:
           **method** (:obj:`str`): Must be either ``"direct"`` or ``"connector project match"``.
           If method is ``"direct"``, ``graph`` argument is ignored.

           **graph** (:obj:`str`): Must be a key within ``project.network.graphs``.
        """
        if method not in ["direct", "connector project match"]:
            raise ValueError("method must be either 'direct' or 'connector project match'")

        self.edges["geometry"] = None

        if method == "direct":
            self.edges["geometry"] = [
                shapely.LineString(
                    (
                        shapely.from_wkb(self.vertices.at[row.a_node - 1, "geometry"]),
                        shapely.from_wkb(self.vertices.at[row.b_node - 1, "geometry"]),
                    )
                ).wkb
                for row in self.edges.itertuples()
            ]
        elif method == "connector project match":
            # Check validity of project and nodes database
            project = get_active_project(must_exist=True)
            warnings.warn(
                'In its current implementation, the "connector project match" method may take a while for large networks.'
            )

            nodes = project.network.nodes.data[["node_id", "geometry"]].set_index("node_id")
            links = project.network.links.data[["link_id", "geometry"]].set_index("link_id")

            if len(nodes) == 0:
                raise ValueError(
                    "No nodes found in the project database. Connector project matching requires an existing project network."
                )

            # Create indexes for access and egress connectors
            connector_rows = (self.edges.link_type == "access_connector") | (self.edges.link_type == "egress_connector")
            other_rows = ~connector_rows

            # Create line strings for non-access and egress connectors
            self.edges.loc[other_rows, "geometry"] = [
                shapely.LineString(
                    (
                        shapely.from_wkb(self.vertices.at[row.a_node - 1, "geometry"]),
                        shapely.from_wkb(self.vertices.at[row.b_node - 1, "geometry"]),
                    )
                ).wkb
                for row in self.edges[other_rows].itertuples()
            ]

            lines = self.__connector_project_match(connector_rows, project, nodes, links, graph)

            self.edges.loc[connector_rows, ("trav_time", "geometry")] = lines

    def __connector_project_match(self, connector_rows, project, nodes, links, graph_key):
        """Create line string geometry for ``connector_rows`` that matches the line strings in
        ``project.network.graphs[graph_key]``.

        :Arguments:
           **connector_rows** (:obj:`pd.Series`): Boolean series for the rows of ``self.edges`` to create line strings for.

           **project** (:obj:`Aequilibrae.project.Project`): Reference to the project to pull the graph from.

           **nodes** (:obj:`pd.DataFrame`): A Dataframe containing the project nodes. Must have columns ``geometry``, and an index of ``node_id``s.

           **links** (:obj:`pd.DataFrame`): A Dataframe containing the project links. Must have columns ``geometry``, and an index of ``link_id``s.

           **graph_key** (:obj:`str`): The key of the ``project.network.graphs`` graph to use for path finding.
        """
        # Create kdtree for fast nearest neighbour lookup on the project db nodes
        nodes["geometry"] = nodes["geometry"].apply(
            lambda geometry: shapely.ops.transform(self.transformer_g_to_p, geometry)
        )
        links["geometry"] = links["geometry"].apply(
            lambda geometry: shapely.ops.transform(self.transformer_g_to_p, geometry)
        )
        nodes_geometries = np.array(list(nodes["geometry"].apply(lambda geometry: (geometry.x, geometry.y))))
        kdtree = KDTree(nodes_geometries)

        # Prepare shortest path computation
        graph = project.network.graphs[graph_key]
        graph.set_graph("distance")
        res = PathResults()
        res.prepare(graph)

        # Loop over connect edges, query for the closest nodes in the project and create the relevant line string
        lines = []
        for row in self.edges[connector_rows].itertuples():
            # row.a_node - 1 because the node_ids are the index + 1
            start = shapely.ops.transform(
                self.transformer_g_to_p, shapely.from_wkb(self.vertices.at[row.a_node - 1, "geometry"])
            )
            end = shapely.ops.transform(
                self.transformer_g_to_p, shapely.from_wkb(self.vertices.at[row.b_node - 1, "geometry"])
            )

            _, ids = kdtree.query([[start.x, start.y], [end.x, end.y]], k=1)

            # If the ids for the closest nodes to the start and end are the same, then we make an edge between those 3
            # If they differ we compute the shortest path between them. If no path exists we use a straight between the start and end
            # Otherwise create a line string using the already existing link geometry.
            if ids[0] == ids[1]:
                line = shapely.LineString((start, nodes.iloc[ids[0]].geometry, end))
            else:
                res.compute_path(*nodes.iloc[ids].index.values)

                if res.path_nodes is not None:
                    line = shapely.ops.linemerge(
                        (
                            shapely.LineString((start, nodes.loc[res.path_nodes[0]].geometry)),
                            *links.loc[res.path].geometry,
                            shapely.LineString((nodes.loc[res.path_nodes[-1]].geometry, end)),
                        )
                    )
                else:
                    line = shapely.LineString((start, end))

            trav_time = line.length / self.walking_speed
            if row.link_type == "access_connector":
                trav_time *= self.access_time_factor
            else:  # row.link_type == "egress_connector"
                trav_time *= self.egress_time_factor

            lines.append((trav_time, shapely.ops.transform(self.transformer_p_to_g, line).wkb))
        return lines

    def create_additional_db_fields(self):
        """Create the additional required entries in the tables."""
        # This graph requires some additional tables and fields in order to store all our information.
        # Currently it mimics what we are storing in the df

        with self.pt_conn as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO link_types (link_type, link_type_id, description) VALUES (?, ?, ?)
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
                ],
            )

    def save_vertices(self, robust=None):
        """
        Write the vertices DataFrame to the public transport database.

        Within the database nodes may not exist at the exact same point in space, provide ``robust=True`` to move the nodes slightly.

        :Arguments:
           **robust** (:obj:`bool`): Deprecated. No longer in use.
        """
        if robust is not None:
            warnings.warn(
                "the 'robust' argument is depreciated and no longer in use. Duplicate geometries are allowed within the public transport database.",
                DeprecationWarning,
            )

        with self.pt_conn as conn:
            if conn.execute("SELECT node_id FROM nodes WHERE period_id=? LIMIT 1;", (self.period_id,)).fetchall():
                raise ValueError(
                    f"cannot save nodes into a database with existing nodes in the same period ({self.period_id})"
                )

            df = self.vertices[SF_VERTEX_COLS]
            df.loc[:, "period_id"] = self.period_id
            conn.executemany(
                f"""\
                INSERT INTO nodes ({",".join(SF_VERTEX_COLS)},modes,period_id)
                VALUES({",".join("?" * (len(SF_VERTEX_COLS) - 1))},GeomFromWKB(?, {self.__global_crs.to_epsg()}),"t",?);
                """,
                df.itertuples(index=False, name=None),
            )

    @staticmethod
    def remove_vertices(pt_conn: sqlite3.Connection, period_id: int):
        """
        Remove a transit graph's vertices from the public transport database specified by it's 'period_id'.

        :Arguments:
            **pt_conn** (:obj:`sqlite3.Connection`): Connection to the ``public_transport.sqlite`` database.
            **period_id** (:obj:`int`): ``period_id`` to remove.

        """
        with pt_conn as conn:
            conn.execute("DELETE FROM nodes where period_id=?", (period_id,))

    def save_edges(self, recreate_line_geometry=False):
        """
        Save the contents of self.edges to the public transport database.

        If no geometry for the edges is present or ``recreate_line_geometry`` is ``True``, direct lines will be created.

        :Arguments:
           **recreate_line_geometry** (:obj:`bool`): Whether to recreate the line strings for the edges as direct lines. Defaults to ``False``.
        """
        # We need to generate the geometry for each edge, this may take a bit
        if "geometry" not in self.edges.columns or recreate_line_geometry:
            self.create_line_geometry()

        with self.pt_conn as conn:
            if conn.execute("SELECT link_id FROM links WHERE period_id=? LIMIT 1;", (self.period_id,)).fetchall():
                raise ValueError("cannot save links into a database with existing links in the same period")

            df = self.edges[SF_EDGE_COLS + ["geometry"]]
            df.loc[:, "period_id"] = self.period_id
            conn.executemany(
                f"""\
                INSERT INTO links ({",".join(SF_EDGE_COLS)},geometry,modes,period_id)
                VALUES({",".join("?" * len(SF_EDGE_COLS))},GeomFromWKB(?, {self.__global_crs.to_epsg()}),"t",?);
                """,
                df.itertuples(index=False, name=None),
            )

    @staticmethod
    def remove_edges(pt_conn: sqlite3.Connection, period_id: int):
        """
        Remove a transit graph's edges from the public transport database specified by it's 'period_id'.

        :Arguments:
            **pt_conn** (:obj:`sqlite3.Connection`): Connection to the ``public_transport.sqlite`` database.
            **period_id** (:obj:`int`): ``period_id`` to remove.

        """
        with pt_conn as conn:
            conn.execute("DELETE FROM links WHERE period_id=?", (period_id,))

    def save_config(self):
        with self.project_conn as conn:
            sql = "INSERT OR REPLACE INTO transit_graph_configs (period_id,config) VALUES (?,?)"
            conn.execute(sql, [self.period_id, json.dumps(self.config)])

    @staticmethod
    def remove_config(conn: sqlite3.Connection, period_id: int):
        """
        Remove a transit graph configuration from the project database specified by it's 'period_id'.

        :Arguments:
            **conn** (:obj:`sqlite3.Connection`): Connection to the ``project.sqlite`` database.
            **period_id** (:obj:`int`): 'period_id' key for the 'transit_graph_configs' table.

        """
        with conn as conn:
            sql = "DELETE FROM transit_graph_configs WHERE period_id = ?"
            conn.execute(sql, [period_id])

    def save(self, robust=True):
        """Save the current graph to the public transport database.

        :Arguments:
            **robust** (:obj:`bool`): Deprecated. No longer in use.
        """
        self.create_additional_db_fields()
        self.save_vertices(robust=robust)
        self.save_edges()
        self.save_config()

    @classmethod
    def remove(cls, pt_conn: sqlite3.Connection, period_id: int):
        cls.remove_edges(pt_conn, period_id)
        cls.remove_vertices(pt_conn, period_id)
        try:
            conn = database_connection("project_database")
            cls.remove_config(conn, period_id)
        finally:
            conn.close()

    def to_transit_graph(self) -> TransitGraph:
        """Create an AequilibraE ``TransitGraph`` object from an SF graph builder."""

        # TODO: Better required link type detections
        # link_type_diff = set(self.edges.link_type.unique()) ^ {'access_connector', 'alighting', 'boarding', 'dwell', 'egress_connector', 'inner_transfer', 'on-board'}
        # if link_type_diff:
        #     raise ValueError(f"Not all required links have been created. Link types {link_type_diff} are missing.")

        g = TransitGraph(config=self.config, od_node_mapping=self.od_node_mapping)
        g.network = self.edges.copy(deep=True)
        g.cost = g.network.trav_time.values
        g.free_flow_time = g.network.trav_time.values

        g.network["id"] = g.network.link_id
        g.prepare_graph(
            self.vertices[
                (
                    (self.vertices.node_type == "origin")
                    if self.blocking_centroid_flows
                    else (self.vertices.node_type == "od")
                )
            ].node_id.values
        )
        g.set_graph("trav_time")
        g.set_blocked_centroid_flows(True)
        g.graph.__compressed_id__ = g.graph.__compressed_id__.astype("int32")

        return g

    @classmethod
    def from_db(cls, public_transport_conn, period_id: int, **kwargs):
        """
        Create a SF graph instance from an existing database save.

        Assumes the database was constructed with the provided save methods.
        No checks are performed to see if the provided arguments are compatible with the saved graph.

        All arguments are forwarded to the constructor.

        :Arguments:
           **public_transport_conn** (:obj:`sqlite3.Connection`): Connection to the 'public_transport.sqlite' database.
        """
        project_conn = database_connection("project_database")
        try:
            config = project_conn.execute(
                "SELECT config FROM transit_graph_configs WHERE period_id = ? LIMIT 1;", [period_id]
            ).fetchone()
        finally:
            project_conn.close()

        if config is None:
            raise ValueError(f"no transit graph configuration found for 'period_id={period_id}'")

        config = json.loads(config[0])
        config.update(kwargs)

        graph = cls(public_transport_conn, **config)

        # FIXME Load specific period_id graph from dynamic table
        graph.vertices = pd.read_sql_query(
            sql=f'SELECT {",".join(SF_VERTEX_COLS[:-1])}, ST_asBinary("geometry") as geometry FROM nodes WHERE period_id={period_id};',
            con=public_transport_conn,
        )
        graph.vertices.node_id = graph.vertices.node_id.astype("int64")
        graph.vertices.line_seg_idx = graph.vertices.line_seg_idx.astype("int64")

        graph.edges = pd.read_sql_query(
            sql=f'SELECT {",".join(SF_EDGE_COLS)}, ST_asBinary("geometry") as geometry FROM links WHERE period_id={period_id};',
            con=public_transport_conn,
        )
        graph.edges.line_id = graph.edges.line_id.fillna("").astype(str)
        graph.edges.stop_id = graph.edges.stop_id.fillna("").astype(str)
        graph.edges.line_seg_idx = graph.edges.line_seg_idx.astype("int64")
        graph.edges.b_node = graph.edges.b_node.astype("int64")
        graph.edges.a_node = graph.edges.a_node.astype("int64")
        graph.edges.direction = graph.edges.direction.astype("int8")

        graph.create_od_node_mapping()

        return graph

    def convert_demand_matrix_from_zone_to_node_ids(
        self, demand_matrix, o_zone_col="origin_zone_id", d_zone_col="destination_zone", demand_col="demand"
    ):
        """Convert a sparse demand matrix from ``zone_id``\'s to the corresponding ``node_id``\'s."""
        if self.blocking_centroid_flows:
            od_matrix = pd.merge(
                demand_matrix,
                self.od_node_mapping[["o_node_id", "taz_id"]],
                left_on=o_zone_col,
                right_on="taz_id",
            )[["o_node_id", d_zone_col, "demand"]]
            od_matrix = pd.merge(
                od_matrix,
                self.od_node_mapping[["d_node_id", "taz_id"]],
                left_on=d_zone_col,
                right_on="taz_id",
            )[["o_node_id", "d_node_id", "demand"]]
        else:
            od_matrix = pd.merge(
                demand_matrix,
                self.od_node_mapping[["node_id", "taz_id"]],
                left_on=o_zone_col,
                right_on="taz_id",
            )[["node_id", d_zone_col, "demand"]]
            od_matrix.rename(columns={"node_id": "o_node_id"}, inplace=True)
            od_matrix = pd.merge(
                od_matrix,
                self.od_node_mapping[["node_id", "taz_id"]],
                left_on=d_zone_col,
                right_on="taz_id",
            )[["o_node_id", "node_id", "demand"]]
            od_matrix.rename(columns={"node_id": "d_node_id"}, inplace=True)
        return od_matrix

    @property
    def config(self):
        return {k: self.__dict__[k] for k in self.__config_attrs}
