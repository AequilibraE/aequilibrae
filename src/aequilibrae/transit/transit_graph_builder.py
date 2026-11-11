"""Create the graph used by public transport assignment algorithms.

Naming Conventions:
- a_node/b_node is the tail/head vertex of an edge, as for road links

TransitGraphBuilder Assumptions:
- opposite directions are not supported. In the GTFS files, this corresponds to direction_id
  from trips.txt (indicates the direction of travel for a trip),
- all times are expressed in seconds [s], all frequencies in [1/s], and
- headways are uniform for trips of the same pattern.

The graph only holds physical attributes: run, dwell and walking times, walking distances, headways, trip counts,
frequencies and capacities. Each ``TransitClass`` turns them into the costs it assigns with (see ``TransitCosting``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import sqlite3
import warnings

import numpy as np
import pandas as pd
import pyproj
import shapely
from pandas.api.types import is_integer_dtype
from pandas.core.frame import DataFrame
from scipy.spatial import KDTree

from aequilibrae.paths import PathResults
from aequilibrae.paths.graph import TransitGraph
from aequilibrae.paths.transit_costing import TransitCosting
from aequilibrae.transit.service_periods import (
    TRIP_INCLUSION_RULES,
    period_duration,
    period_window,
    trips_in_period_sql,
)
from aequilibrae.utils.geo_utils import haversine

logger = logging.getLogger(__name__)

SF_VERTEX_COLS = ["node_id", "node_type", "stop_id", "line_id", "line_seg_idx", "taz_id", "geometry"]
SF_EDGE_COLS = [
    "link_id",
    "link_type",
    "line_id",
    "stop_id",
    "line_seg_idx",
    "segment_id",
    "a_node",
    "b_node",
    "run_time",
    "dwell_time",
    "walk_distance",
    "walk_time",
    "headway",
    "trip_count",
    "line_freq",
    "capacity",
    "direction",
]

# Value of each physical attribute on edges it does not apply to
EDGE_DEFAULTS = {
    "line_id": "",
    "stop_id": "",
    "line_seg_idx": -1,
    "run_time": 0.0,
    "dwell_time": 0.0,
    "walk_distance": 0.0,
    "walk_time": 0.0,
    "headway": 0.0,
    "trip_count": 0,
    "line_freq": np.inf,
    "capacity": 0.0,
    "direction": 1,
}


class _LegacyCostingAttribute:
    """A builder attribute that moved to ``TransitCosting``.

    Setting it warns, and changes the costing that classes on this builder's graphs default to.
    """

    def __init__(self, costing_field: str):
        self.costing_field = costing_field

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, builder, owner=None):
        if builder is None:
            return self
        return getattr(builder._legacy_costing or TransitCosting(), self.costing_field)

    def __set__(self, builder, value):
        warnings.warn(
            f"TransitGraphBuilder.{self.name} is deprecated and will be removed in 2.1. Pass "
            f"TransitCosting({self.costing_field}=...) to TransitClass instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        costing = builder._legacy_costing or TransitCosting()
        builder._legacy_costing = dataclasses.replace(costing, **{self.costing_field: float(value)})


class _RetiredAttribute:
    """A builder attribute that no longer has any effect."""

    def __init__(self, value, reason: str):
        self.value = value
        self.reason = reason

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, builder, owner=None):
        return self if builder is None else self.value

    def __set__(self, builder, value):
        warnings.warn(
            f"TransitGraphBuilder.{self.name} is deprecated, has no effect and will be removed in 2.1: {self.reason}",
            DeprecationWarning,
            stacklevel=2,
        )


class TransitGraphBuilder:
    """Graph builder for the transit assignment Spiess & Florian algorithm.

    The graph holds the physical supply of one period. ``TransitClass`` costs it for assignment.

    :Arguments:
        **project** (:obj:`Project` or :obj:`Scenario`): Project or scenario to build from. A project's active
        scenario is used, and the builder keeps using it if another scenario is activated later.

        **period_id** (:obj:`int`): Period id for the period to be used. Preferred over start
        and end.

        **time_margin** (:obj:`int`): Time margin, extends the start and end times by
        `time_margin` ([start, end] becomes [start - time_margin, end + time_margin]), in order
        to include more trips when computing mean values. Defaults to ``0``.

        **projected_crs** (:obj:`str`): Projected CRS used to find connector candidates. Defaults to the UTM zone
        of the stops' centroid. Reported distances are always geodesic.

        **num_threads** (:obj:`int`): Number of threads to be used where possible.
        Defaults to ``-1`` (using all available threads).

        **with_inner_stop_transfers** (:obj:`bool`): Deprecated. Transfers at a stop alight to the stop and board
        again, so no separate edges are needed. Defaults to ``False``.

        **with_outer_stop_transfers** (:obj:`bool`): Whether to create transfer edges between the distinct stops of
        each parent station. Defaults to ``False``.

        **with_walking_edges** (:obj:`bool`): Whether to create walking edges between distinct
        stops of each station. Defaults to ``True``. Station stop pairs get one edge, an outer transfer when
        ``with_outer_stop_transfers`` is set.

        **distance_upper_bound** (:obj:`float`): Upper bound on connector distance, in metres. Defaults to
        ``np.inf``.

        **blocking_centroid_flows** (:obj:`bool`): Whether to prevent OD connector-through traversal
        by splitting OD nodes into origin/destination roles. Defaults to ``True``.

        **connector_method** (:obj:`str`): Either ``nearest_neighbour`` or ``overlapping_regions``.

        **max_connectors_per_zone** (:obj:`int`): Maximum connectors per zone. Defaults to ``-1``
        for unlimited.

        **walking_speed** (:obj:`float`): Walking speed in metres per second. Defaults to ``1.0``.

        **trip_inclusion** (:obj:`str`): Rule assigning trips to the period: ``start``, ``end``, ``midpoint`` or
        ``any``, as for ``Transit.build_pt_preload``. Defaults to ``start``.
    """

    alighting_penalty = _LegacyCostingAttribute("alighting_penalty")
    wait_time_factor = _LegacyCostingAttribute("headway_fraction")
    walk_time_factor = _LegacyCostingAttribute("walk_weight")
    access_time_factor = _LegacyCostingAttribute("access_weight")
    egress_time_factor = _LegacyCostingAttribute("egress_weight")
    uniform_dwell_time = _RetiredAttribute(0.0, "dwell edges use the scheduled dwell at each stop")
    a_tiny_time_duration = _RetiredAttribute(0.0, "the assignment floors zero edge costs itself")

    def __init__(
        self,
        project,
        period_id: int = 1,
        time_margin: int = 0,
        projected_crs: str | None = None,
        num_threads: int = -1,
        *,
        with_inner_stop_transfers: bool = False,
        with_outer_stop_transfers: bool = False,
        with_walking_edges: bool = True,
        distance_upper_bound: float = np.inf,
        blocking_centroid_flows: bool = True,
        connector_method: str = "nearest_neighbour",
        max_connectors_per_zone: int = -1,
        walking_speed: float = 1.0,
        trip_inclusion: str = "start",
    ):
        self.project = project
        # Bind to the scenario now, so activating another one later does not change where this graph lives
        self.scenario = getattr(project, "scenario", project)

        self.period_id = period_id
        with self.scenario.db_connection as conn:
            start, end = period_window(conn, period_id)

        if trip_inclusion not in TRIP_INCLUSION_RULES:
            raise ValueError(f"trip_inclusion must be one of {TRIP_INCLUSION_RULES}")

        self.time_margin = time_margin
        self.start = start - time_margin  # starting time of the selected time period
        self.end = end + time_margin  # ending time of the selected time period
        self.period_duration = period_duration(self.start, self.end)
        self.num_threads = num_threads

        # graph components
        # ----------------
        self.vertices: pd.DataFrame | None = None
        self.edges: pd.DataFrame | None = None
        self.__line_segments: pd.DataFrame | None = None
        self.od_node_mapping: pd.DataFrame | None = None

        # vertices
        self.__stop_vertices: pd.DataFrame | None = None
        self.__boarding_vertices: pd.DataFrame | None = None
        self.__alighting_vertices: pd.DataFrame | None = None
        self.__od_vertices: pd.DataFrame | None = None

        # edges
        self.__on_board_edges: pd.DataFrame | None = None
        self.__dwell_edges: pd.DataFrame | None = None
        self.__alighting_edges: pd.DataFrame | None = None
        self.__boarding_edges: pd.DataFrame | None = None
        self.__connector_edges: pd.DataFrame | None = None
        self.__station_edges: pd.DataFrame | None = None

        self.global_crs = "EPSG:4326"
        self.__global_crs = pyproj.CRS(self.global_crs)
        self.projected_crs = projected_crs

        if with_inner_stop_transfers:
            warnings.warn(
                "'with_inner_stop_transfers' is deprecated and has no effect: transfers at a stop alight to the stop "
                "and board again.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Costing that classes on this builder's graphs default to, set only through the deprecated attributes
        self._legacy_costing: TransitCosting | None = None

        # graph parameters
        self.walking_speed = walking_speed
        self.trip_inclusion = trip_inclusion
        self.with_inner_stop_transfers = with_inner_stop_transfers
        self.with_outer_stop_transfers = with_outer_stop_transfers
        self.with_walking_edges = with_walking_edges
        self.distance_upper_bound = distance_upper_bound
        self.blocking_centroid_flows = blocking_centroid_flows
        self.connector_method = connector_method
        self.max_connectors_per_zone = max_connectors_per_zone

        # Every argument that shapes the graph, so a saved configuration rebuilds it
        self.__config_attrs = [
            "period_id",
            "time_margin",
            "projected_crs",
            "with_inner_stop_transfers",
            "with_outer_stop_transfers",
            "with_walking_edges",
            "distance_upper_bound",
            "blocking_centroid_flows",
            "connector_method",
            "max_connectors_per_zone",
            "walking_speed",
            "trip_inclusion",
        ]

    @property
    def projected_crs(self) -> str:
        """Projected CRS used to find connector candidates, resolved from the stops when not given."""
        if self.__projected_crs_name is None:
            self.__set_projected_crs(self._local_utm_crs())
        return self.__projected_crs_name

    @projected_crs.setter
    def projected_crs(self, value: str | None):
        self.__projected_crs_name = None
        if value is not None:
            self.__set_projected_crs(value)

    def __set_projected_crs(self, name: str):
        crs = pyproj.CRS(name)
        self.__projected_crs_name = name
        self.__projected_crs = crs
        # longlat to projected CRS transformer
        self.transformer_g_to_p = pyproj.Transformer.from_crs(self.__global_crs, crs, always_xy=True).transform
        self.transformer_p_to_g = pyproj.Transformer.from_crs(crs, self.__global_crs, always_xy=True).transform

    def _local_utm_crs(self) -> str:
        """Return the UTM zone of the centroid of all stops."""
        with self.scenario.transit_connection as conn:
            lon, lat = conn.execute("SELECT AVG(ST_X(geometry)), AVG(ST_Y(geometry)) FROM stops").fetchone()
        if lon is None:
            raise ValueError("cannot choose a projected CRS without stops; pass 'projected_crs'")
        zone = min(int((lon + 180.0) // 6.0) + 1, 60)
        return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"

    def add_zones(self, zones, from_crs: str | None = None):
        """Add zones as ODs.

        :Arguments:
            **zones** (:obj:`pd.DataFrame`): DataFrame containing the zoning information.
            Columns must include `zone_id` and `geometry`.

            **from_crs** (:obj:`str`, optional): The CRS of the `geometry` column of `zones`. Defaults to
            ``EPSG:4326``.
        """
        if "zone_id" not in zones.columns or "geometry" not in zones.columns:
            raise KeyError("zone_id and geometry must be columns in zones")
        elif not (is_integer_dtype(zones.zone_id.dtype) and zones.zone_id.min() > 0):
            raise ValueError("'zone_id' must consist of integers greater than 0")
        elif not zones.zone_id.is_unique:
            raise ValueError("'zone_id' must be unique")

        geo_values = zones.geometry.values
        if all(isinstance(x, str) for x in geo_values):
            geometry = shapely.from_wkt(geo_values)
        elif all(isinstance(x, bytes) for x in geo_values):
            geometry = shapely.from_wkb(geo_values)
        # Check if the supplied zones df is from geopandas without import geopandas.
        # We check __mro__ in case of inheritance. https://stackoverflow.com/a/63337375/14047443
        elif "GeometryDtype" in [t.__name__ for t in type(zones.geometry.dtype).__mro__] or all(
            isinstance(x, shapely.geometry.base.BaseGeometry) for x in geo_values
        ):
            geometry = geo_values
        else:
            raise TypeError("geometry is not a string, bytes, or shapely.Geometry instance")

        if from_crs:
            transformer = pyproj.Transformer.from_crs(
                pyproj.CRS(from_crs), self.__projected_crs_for_zones(), always_xy=True
            ).transform
        else:
            transformer = self.__to_projected()
        geometry = [shapely.transform(p, transformer, include_z=None, interleaved=False) for p in geometry]
        centroids = shapely.centroid(geometry)

        # Zones in zone_id order, which is the centroid order of the graph and its skims
        order = np.argsort(zones.zone_id.to_numpy(), kind="stable")
        self.zones = pd.DataFrame(
            {
                "zone_id": zones.zone_id.to_numpy()[order].astype(int),
                "geometry": shapely.to_wkb(
                    [
                        shapely.transform(geometry[i], self.transformer_p_to_g, include_z=None, interleaved=False)
                        for i in order
                    ]
                ),
                "centroids": shapely.to_wkb(
                    [
                        shapely.transform(centroids[i], self.transformer_p_to_g, include_z=None, interleaved=False)
                        for i in order
                    ]
                ),
            }
        )

    def __projected_crs_for_zones(self):
        self.projected_crs  # noqa: B018 resolves the CRS and its transformers
        return self.__projected_crs

    def __to_projected(self):
        self.projected_crs  # noqa: B018 resolves the CRS and its transformers
        return self.transformer_g_to_p

    def _period_trips_sql(self) -> str:
        return trips_in_period_sql(self.start, self.end, self.trip_inclusion)

    def _create_line_segments(self):
        """Line segments correspond to segments between two successive stops for each line.

        For example if 2 lines, L1 and L2, are going from stop A to stop B, we have 2 line segments:
        - L1_AB
        - L2_AB

        Every attribute comes from the complete trips the period includes: mean run times between stops, mean dwell
        at each stop, the number of trips and the period rate of distinct departures, which is the frequency the
        assignment waits with.

        line_segments table format:
            pattern_id  seq    from_stop      to_stop shortname         line_id  run_time  headway  line_freq ...
        0  10001006000    0  10000000464  10000000462        T2  T2_10001006000     150.0    240.0   0.004167
        1  10001006000    1  10000000462  10000000459        T2  T2_10001006000     110.0    240.0   0.004167
        """
        schedule_sql = f"""
            SELECT trips_schedule.trip_id, trips.pattern_id, trips_schedule.seq, trips_schedule.arrival,
                   trips_schedule.departure
            FROM trips_schedule INNER JOIN trips ON trips_schedule.trip_id = trips.trip_id
            WHERE trips_schedule.trip_id IN ({self._period_trips_sql()})"""
        route_links_sql = """SELECT pattern_id, seq, CAST(from_stop AS TEXT) from_stop, CAST(to_stop AS TEXT) to_stop
            FROM route_links"""
        routes_sql = "SELECT pattern_id, CAST(shortname AS TEXT) shortname, total_capacity FROM routes"

        with self.scenario.transit_connection as conn:
            schedule = pd.read_sql(sql=schedule_sql, con=conn)
            route_links = pd.read_sql(sql=route_links_sql, con=conn)
            routes = pd.read_sql(sql=routes_sql, con=conn)

        schedule = schedule.sort_values(["pattern_id", "trip_id", "seq"]).reset_index(drop=True)
        schedule["dwell"] = self.__guard_negative(
            schedule.departure - schedule.arrival, "departure before arrival at a stop", schedule
        )
        # Run time of the segment starting at each stop, from departure there to arrival at the next stop
        next_arrival = schedule.groupby("trip_id").arrival.shift(-1)
        schedule["run_time"] = self.__guard_negative(
            next_arrival - schedule.departure, "arrival before departure from the previous stop", schedule
        )

        run_times = schedule.dropna(subset="run_time").groupby(["pattern_id", "seq"]).run_time.mean()
        dwell_times = schedule.groupby(["pattern_id", "seq"]).dwell.mean()

        # Duplicate trips (same pattern and departure) add capacity, but not frequency
        departures = schedule.groupby("trip_id", sort=False).agg(
            pattern_id=("pattern_id", "first"), departure=("departure", "first")
        )
        patterns = departures.groupby("pattern_id").agg(
            trip_count=("departure", "size"),
            services=("departure", "nunique"),
            first_departure=("departure", "min"),
            last_departure=("departure", "max"),
        )
        patterns["headway"] = (patterns.last_departure - patterns.first_departure) / (patterns.services - 1)
        # A single departure has no headway, and is taken to run once in the period
        patterns.loc[patterns.services < 2, "headway"] = self.period_duration
        patterns["line_freq"] = patterns.services / self.period_duration
        patterns = patterns[["trip_count", "headway", "line_freq"]].reset_index()

        segments = route_links.merge(patterns, on="pattern_id", how="inner")
        segments = segments.merge(routes, on="pattern_id", how="left")
        segments = segments.merge(
            run_times.rename("run_time"), left_on=["pattern_id", "seq"], right_index=True, how="left"
        )
        segments = segments.merge(
            dwell_times.rename("from_dwell"), left_on=["pattern_id", "seq"], right_index=True, how="left"
        )
        to_dwell = dwell_times.rename("to_dwell").reset_index()
        to_dwell["seq"] -= 1
        segments = segments.merge(to_dwell, on=["pattern_id", "seq"], how="left")

        missing = segments.run_time.isna()
        if missing.any():
            logger.warning(f"{int(missing.sum())} route link(s) have no scheduled run time and take zero time")
        segments[["run_time", "from_dwell", "to_dwell"]] = segments[["run_time", "from_dwell", "to_dwell"]].fillna(0.0)

        # we create a line id by concatenating the route short name with the pattern_id
        segments["line_id"] = segments["shortname"] + "_" + segments["pattern_id"].astype(str)
        segments["segment_id"] = segments["pattern_id"].astype(str) + ":" + segments["seq"].astype(str)
        segments["capacity"] = segments.trip_count * segments.total_capacity.fillna(0.0)

        self.__line_segments = segments.sort_values(["pattern_id", "seq"]).reset_index(drop=True)

    @staticmethod
    def __guard_negative(values: pd.Series, problem: str, schedule: pd.DataFrame) -> pd.Series:
        negative = values < 0
        if negative.any():
            trips = schedule.loc[negative, "trip_id"].unique()
            logger.warning(f"{int(negative.sum())} stop event(s) have {problem}, taken as zero, on trips {trips[:10]}")
        return values.mask(negative, 0.0)

    def _create_stop_vertices(self):
        """Create stop vertices."""
        # select all stops
        sql = "SELECT CAST(stop_id AS TEXT) stop_id, ST_AsBinary(geometry) AS geometry FROM stops"
        with self.scenario.transit_connection as conn:
            stop_vertices = pd.read_sql(sql=sql, con=conn)

        # filter stops that are used on the given time range
        assert self.__line_segments is not None
        stops_ids = pd.concat((self.__line_segments.from_stop, self.__line_segments.to_stop), axis=0).unique()
        stop_vertices = stop_vertices.loc[stop_vertices.stop_id.isin(stops_ids)]

        # uniform attributes
        stop_vertices["line_seg_idx"] = -1
        stop_vertices["taz_id"] = -1
        stop_vertices["node_type"] = "stop"

        self.__stop_vertices = stop_vertices

    def _create_boarding_vertices(self):
        """Create boarding vertices."""
        assert self.__line_segments is not None
        boarding_vertices = self.__line_segments[["line_id", "seq", "from_stop"]].copy(deep=True)
        boarding_vertices = boarding_vertices.rename(columns={"seq": "line_seg_idx", "from_stop": "stop_id"})
        assert self.__stop_vertices is not None
        boarding_vertices = pd.merge(
            boarding_vertices, self.__stop_vertices[["stop_id", "geometry"]], on="stop_id", how="left"
        )

        # uniform attributes
        boarding_vertices["taz_id"] = -1
        boarding_vertices["node_type"] = "boarding"

        self.__boarding_vertices = boarding_vertices

    def _create_alighting_vertices(self):
        """Create alighting vertices."""
        assert self.__line_segments is not None
        assert self.__stop_vertices is not None
        alighting_vertices = self.__line_segments[["line_id", "seq", "to_stop"]].copy(deep=True)
        alighting_vertices = alighting_vertices.rename(columns={"seq": "line_seg_idx", "to_stop": "stop_id"})
        alighting_vertices = pd.merge(
            alighting_vertices, self.__stop_vertices[["stop_id", "geometry"]], on="stop_id", how="left"
        )

        # uniform attributes
        alighting_vertices["taz_id"] = -1
        alighting_vertices["node_type"] = "alighting"

        self.__alighting_vertices = alighting_vertices

    def _create_od_vertices(self):
        """Create OD vertices from zones.

        If zones have not previously been added, add zones from the project.
        If ``self.blocking_centroid_flows`` is ``True``, a distinction is made between
        ``origin`` and ``destination`` vertices. Otherwise, they are both classified as ``od``.
        """
        if "zones" not in self.__dict__:
            self.add_zones(
                pd.DataFrame(
                    [(zone.zone_id, zone.geometry) for zone in self.scenario.network.zones],
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
        od_vertices["stop_id"] = ""
        od_vertices["line_id"] = ""

        self.__od_vertices = od_vertices

    def create_od_node_mapping(self):
        """Build a dataframe mapping the centroid node ids with to transport assignment zone
        ids, in zone order."""
        assert self.vertices is not None
        if self.blocking_centroid_flows:
            origin_nodes = self.vertices.loc[
                self.vertices.node_type == "origin",
                ["node_id", "taz_id"],
            ]
            origin_nodes = origin_nodes.rename(columns={"node_id": "o_node_id"})
            destination_nodes = self.vertices.loc[
                self.vertices.node_type == "destination",
                ["node_id", "taz_id"],
            ]
            destination_nodes = destination_nodes.rename(columns={"node_id": "d_node_id"})
            od_node_mapping = pd.merge(origin_nodes, destination_nodes, on="taz_id", how="left")[
                ["o_node_id", "d_node_id", "taz_id"]
            ]
        else:
            od_node_mapping = self.vertices.loc[
                self.vertices.node_type == "od",
                ["node_id", "taz_id"],
            ]
        self.od_node_mapping = od_node_mapping.sort_values("taz_id").reset_index(drop=True)

    def _create_vertices(self):
        """Graph vertices creation as a dataframe.

        Vertices have the following attributes:
            - node_id (:obj:`int`): Equal to the ``taz_id`` for centroids/origins. Contiguous
              after,
            - node_type (:obj:`str`): Either 'stop', 'boarding', 'alighting', 'od', 'origin',
              or 'destination',
            - stop_id (:obj:`str`): Only applies to 'stop', 'boarding' and 'alighting' vertices,
            - line_id (:obj:`str`): Only applies to 'boarding' and 'alighting' vertices,
            - line_seg_idx (:obj:`int`): Only applies to 'boarding' and 'alighting' vertices,
            - taz_id (:obj:`int`): Only applies to 'origin', 'destination', and 'od' nodes,
              ``-1`` elsewhere,
            - geometry (:obj:`str`): Geometry object in WKB (well known binary).
        """
        self._create_line_segments()
        self._create_stop_vertices()
        self._create_boarding_vertices()
        self._create_alighting_vertices()
        self._create_od_vertices()

        # stack the dataframes on top of each other
        assert self.__od_vertices is not None
        assert self.__stop_vertices is not None
        assert self.__boarding_vertices is not None
        assert self.__alighting_vertices is not None
        self.vertices: DataFrame = pd.concat(
            [
                self.__od_vertices,
                self.__stop_vertices,
                self.__boarding_vertices,
                self.__alighting_vertices,
            ],
            axis=0,
        )

        # reset index and copy it to column
        self.vertices = self.vertices.reset_index(drop=True)

        o_vertices = self.__od_vertices[self.__od_vertices.node_type.isin(["origin", "od"])]
        node_id_min = o_vertices.taz_id.max() + 1
        self.vertices["node_id"] = np.hstack(
            (
                o_vertices.taz_id.to_numpy(),
                np.arange(node_id_min, node_id_min + len(self.vertices) - len(o_vertices)),
            )
        )

        self.vertices = self.vertices[SF_VERTEX_COLS]
        self.create_od_node_mapping()

        # data types
        self.vertices.node_id = self.vertices.node_id.astype("int64")
        self.vertices.stop_id = self.vertices.stop_id.astype("string").fillna("").astype(str)
        self.vertices.line_id = self.vertices.line_id.astype("string").fillna("").astype(str)
        self.vertices.line_seg_idx = self.vertices.line_seg_idx.fillna(-1).astype("int64")
        self.vertices.taz_id = self.vertices.taz_id.astype("int64")

    def __line_vertices(self, node_type: str) -> pd.DataFrame:
        assert self.vertices is not None
        return self.vertices.loc[self.vertices.node_type == node_type, ["line_id", "line_seg_idx", "node_id"]]

    def __stop_nodes(self) -> pd.DataFrame:
        assert self.vertices is not None
        return self.vertices.loc[self.vertices.node_type == "stop", ["stop_id", "node_id"]]

    def _create_on_board_edges(self):
        """Create on board edges, from a segment's boarding vertex to its alighting vertex."""
        assert self.__line_segments is not None
        on_board_edges = self.__line_segments[["line_id", "seq", "segment_id", "run_time", "capacity"]].rename(
            columns={"seq": "line_seg_idx"}
        )
        on_board_edges = on_board_edges.merge(self.__line_vertices("boarding"), on=["line_id", "line_seg_idx"])
        on_board_edges = on_board_edges.rename(columns={"node_id": "a_node"})
        on_board_edges = on_board_edges.merge(self.__line_vertices("alighting"), on=["line_id", "line_seg_idx"])
        on_board_edges = on_board_edges.rename(columns={"node_id": "b_node"})

        on_board_edges["link_type"] = "on-board"
        self.__on_board_edges = on_board_edges

    def _create_boarding_edges(self):
        """Create boarding edges, from a stop to the boarding vertex of each segment leaving it."""
        assert self.__line_segments is not None
        columns = ["line_id", "seq", "segment_id", "from_stop", "from_dwell", "headway", "trip_count", "line_freq"]
        boarding_edges = self.__line_segments[columns].rename(
            columns={"seq": "line_seg_idx", "from_stop": "stop_id", "from_dwell": "dwell_time"}
        )
        boarding_edges = boarding_edges.merge(self.__stop_nodes(), on="stop_id").rename(columns={"node_id": "a_node"})
        boarding_edges = boarding_edges.merge(self.__line_vertices("boarding"), on=["line_id", "line_seg_idx"])
        boarding_edges = boarding_edges.rename(columns={"node_id": "b_node"})

        boarding_edges["link_type"] = "boarding"
        self.__boarding_edges = boarding_edges

    def _create_alighting_edges(self):
        """Create alighting edges, from the alighting vertex of each segment to the stop it reaches."""
        assert self.__line_segments is not None
        alighting_edges = self.__line_segments[["line_id", "seq", "segment_id", "to_stop", "to_dwell"]].rename(
            columns={"seq": "line_seg_idx", "to_stop": "stop_id", "to_dwell": "dwell_time"}
        )
        alighting_edges = alighting_edges.merge(self.__line_vertices("alighting"), on=["line_id", "line_seg_idx"])
        alighting_edges = alighting_edges.rename(columns={"node_id": "a_node"})
        alighting_edges = alighting_edges.merge(self.__stop_nodes(), on="stop_id").rename(columns={"node_id": "b_node"})

        alighting_edges["link_type"] = "alighting"
        self.__alighting_edges = alighting_edges

    def _create_dwell_edges(self):
        """Create dwell edges, for riders staying on board at a stop, with the scheduled dwell there."""
        # we start by removing the first segment of each line
        assert self.__line_segments is not None
        dwell_edges = self.__line_segments.loc[
            self.__line_segments.seq != 0, ["line_id", "seq", "segment_id", "from_stop", "from_dwell"]
        ].rename(columns={"seq": "line_seg_idx", "from_stop": "stop_id", "from_dwell": "dwell_time"})

        # tail: alighting vertex of the previous segment, head: boarding vertex of this one
        previous = self.__line_vertices("alighting").assign(line_seg_idx=lambda df: df.line_seg_idx + 1)
        dwell_edges = dwell_edges.merge(previous, on=["line_id", "line_seg_idx"]).rename(columns={"node_id": "a_node"})
        dwell_edges = dwell_edges.merge(self.__line_vertices("boarding"), on=["line_id", "line_seg_idx"])
        dwell_edges = dwell_edges.rename(columns={"node_id": "b_node"})

        dwell_edges["link_type"] = "dwell"
        self.__dwell_edges = dwell_edges

    def _create_connector_edges(self, method=None, allow_missing_connections=True):
        """
        Create the connector edges between each stops and ODs.

        Nearest neighbour: Creates edges between every stop and its nearest OD.

        Overlapping regions: Creates edges between all stops that lying within the circle
        centred on each OD whose radius is the distance to the next nearest OD.

        With either method every zone also connects to its nearest stop, so no zone is left without a connector
        unless ``distance_upper_bound`` excludes it. Candidates are searched in ``projected_crs``, while distances
        are geodesic.

        :Arguments:
           **method** (:obj:`str`): Must either be `overlapping_regions`, or `nearest_neighbour`.
           Defaults to `overlapping_regions`.

           **allow_missing_connections** (:obj:`bool`): Whether to allow stops without connectors under
           `overlapping_regions`, or to connect each to its nearest OD. Defaults to ``True``.
        """
        if method is None:
            method = self.connector_method
        else:
            self.connector_method = method

        if method not in ["overlapping_regions", "nearest_neighbour"]:
            raise ValueError("method must be either 'overlapping_regions' or 'nearest_neighbour'")

        assert self.vertices is not None
        node_type = "origin" if self.blocking_centroid_flows else "od"
        zones = self.vertices[self.vertices.node_type == node_type][["node_id", "geometry"]].reset_index(drop=True)
        stops = self.vertices[self.vertices.node_type == "stop"][["node_id", "stop_id", "geometry"]]
        stops = stops.reset_index(drop=True)

        zone_lonlat = shapely.get_coordinates(shapely.from_wkb(zones.geometry.to_numpy()))
        stop_lonlat = shapely.get_coordinates(shapely.from_wkb(stops.geometry.to_numpy()))
        project = pyproj.Transformer.from_crs(self.__global_crs, self.projected_crs, always_xy=True)
        zone_xy = np.column_stack(project.transform(zone_lonlat[:, 0], zone_lonlat[:, 1]))
        stop_xy = np.column_stack(project.transform(stop_lonlat[:, 0], stop_lonlat[:, 1]))

        zone_tree = KDTree(zone_xy)
        stop_tree = KDTree(stop_xy)
        pairs = []  # (stop index, zone index)

        if method == "nearest_neighbour":
            # the closest OD of each stop
            _, nearest_zone = zone_tree.query(stop_xy, k=1, workers=self.num_threads)
            pairs.append(np.column_stack((np.arange(len(stops)), nearest_zone)))

        elif method == "overlapping_regions":
            # Each OD reaches the stops closer than its nearest neighbouring OD, within the distance bound
            if len(zones) > 1:
                radius = zone_tree.query(zone_xy, k=[2], workers=self.num_threads)[0].reshape(-1)
            else:
                radius = np.full(len(zones), np.inf)
            radius = np.minimum(radius, self.distance_upper_bound)
            # scipy's ball query needs a finite radius, and twice the extent of all points reaches every stop
            extent = 2.0 * np.ptp(np.vstack((zone_xy, stop_xy)), axis=0).max() + 1.0
            radius = np.where(np.isfinite(radius), radius, extent)
            members = stop_tree.query_ball_point(zone_xy, radius, workers=self.num_threads)
            for zone, stop_indices in enumerate(members):
                pairs.append(
                    np.column_stack((np.asarray(stop_indices, dtype=np.int64), np.full(len(stop_indices), zone)))
                )

            if not allow_missing_connections:
                connected = np.unique(np.concatenate([p[:, 0] for p in pairs])) if pairs else np.array([], int)
                missing = np.setdiff1d(np.arange(len(stops)), connected)
                if len(missing):
                    _, nearest_zone = zone_tree.query(stop_xy[missing], k=1, workers=self.num_threads)
                    pairs.append(np.column_stack((missing, nearest_zone)))

        # Every zone connects to at least its nearest stop
        _, nearest_stop = stop_tree.query(zone_xy, k=1, workers=self.num_threads)
        pairs.append(np.column_stack((nearest_stop, np.arange(len(zones)))))

        pairs = np.unique(np.vstack(pairs).astype(np.int64), axis=0)
        stop_idx, zone_idx = pairs[:, 0], pairs[:, 1]
        distance = haversine(
            stop_lonlat[stop_idx, 0], stop_lonlat[stop_idx, 1], zone_lonlat[zone_idx, 0], zone_lonlat[zone_idx, 1]
        )

        access_connector_edges = pd.DataFrame(
            {
                "a_node": zones.node_id.to_numpy()[zone_idx],
                "b_node": stops.node_id.to_numpy()[stop_idx],
                "stop_id": stops.stop_id.to_numpy()[stop_idx],
                "walk_distance": distance,
            }
        )
        access_connector_edges = access_connector_edges[
            access_connector_edges.walk_distance <= self.distance_upper_bound
        ]
        access_connector_edges = access_connector_edges.sort_values(["a_node", "walk_distance", "b_node"])

        if self.max_connectors_per_zone > 0:
            # max_connectors_per_zone connectors per zone with smallest travel time are selected
            access_connector_edges = access_connector_edges.groupby("a_node").head(self.max_connectors_per_zone)

        unconnected = np.setdiff1d(zones.node_id.to_numpy(), access_connector_edges.a_node.to_numpy())
        if len(unconnected):
            logger.warning(f"{len(unconnected)} zone(s) have no stop within the distance bound: {unconnected[:10]}")

        access_connector_edges = access_connector_edges.reset_index(drop=True)
        access_connector_edges["walk_time"] = access_connector_edges.walk_distance / self.walking_speed
        access_connector_edges["segment_id"] = access_connector_edges.stop_id
        access_connector_edges["link_type"] = "access_connector"

        # Egress connectors run from the stop to the destination node of the zone
        egress_connector_edges = access_connector_edges.rename(columns={"a_node": "b_node", "b_node": "a_node"})
        assert self.od_node_mapping is not None
        if self.blocking_centroid_flows:
            destinations = self.od_node_mapping.set_index("o_node_id").d_node_id
            egress_connector_edges["b_node"] = destinations.loc[egress_connector_edges.b_node].to_numpy()
        egress_connector_edges["link_type"] = "egress_connector"

        self.__connector_edges = pd.concat([access_connector_edges, egress_connector_edges], axis=0)

    def _create_station_edges(self, link_type: str):
        """Create edges between the distinct stops of each parent station, walking from one stop to the other."""
        sql = (
            "SELECT CAST(stop_id AS TEXT) stop_id, CAST(parent_station AS TEXT) parent_station FROM stops "
            "WHERE parent_station IS NOT NULL AND parent_station <> ''"
        )
        with self.scenario.transit_connection as conn:
            stations = pd.read_sql(sql=sql, con=conn).drop_duplicates()

        assert self.vertices is not None
        stops = self.vertices[self.vertices.node_type == "stop"][["stop_id", "node_id", "geometry"]]
        stops = stops.merge(stations, on="stop_id", how="inner")

        edges = stops.merge(stops, on="parent_station", suffixes=("_o", "_d"))
        edges = edges[edges.stop_id_o != edges.stop_id_d]

        o_lonlat = shapely.get_coordinates(shapely.from_wkb(edges.geometry_o.to_numpy()))
        d_lonlat = shapely.get_coordinates(shapely.from_wkb(edges.geometry_d.to_numpy()))
        distance = haversine(o_lonlat[:, 0], o_lonlat[:, 1], d_lonlat[:, 0], d_lonlat[:, 1])

        self.__station_edges = pd.DataFrame(
            {
                "a_node": edges.node_id_o.to_numpy(),
                "b_node": edges.node_id_d.to_numpy(),
                "segment_id": (edges.stop_id_o + ":" + edges.stop_id_d).to_numpy(),
                "walk_distance": distance,
                "walk_time": distance / self.walking_speed,
                "link_type": link_type,
            }
        )

    def _create_edges(self):
        """Graph edges creation as a Dataframe.

        Edges have the following attributes:
            - link_type (:obj:`str`): Either 'on-board', 'boarding', 'alighting', 'dwell',
              'outer_transfer', 'access_connector', "egress_connector" or 'walking',
            - line_id (:obj:`str`): Only applies to 'on-board', 'boarding', 'alighting' and
              'dwell' edges,
            - stop_id (:obj:`str`): The stop of boarding, alighting, dwell and connector edges,
            - line_seg_idx (:obj:`int`): Segment of 'on-board', 'boarding', 'alighting' and 'dwell' edges,
            - segment_id (:obj:`str`): Physical element the edge belongs to: ``pattern_id:seq`` for line edges,
              the stop for connectors and ``from_stop:to_stop`` for walks between stops,
            - a_node (:obj:`int`): Tail vertex,
            - b_node (:obj:`int`): Head vertex,
            - run_time (:obj:`float`): Mean scheduled run time of 'on-board' edges,
            - dwell_time (:obj:`float`): Mean scheduled dwell at the stop of 'dwell', 'boarding' and 'alighting'
              edges,
            - walk_distance (:obj:`float`): Geodesic walking distance, in metres,
            - walk_time (:obj:`float`): Walking distance over walking speed,
            - headway (:obj:`float`): Scheduled headway between departures at the line's first stop, on 'boarding'
              edges,
            - trip_count (:obj:`int`): Trips of the line in the period, on 'boarding' edges,
            - line_freq (:obj:`float`): Distinct departures over the period length on 'boarding' edges, infinite
              elsewhere,
            - capacity (:obj:`float`): Capacity of all the period's trips on 'on-board' edges.
        """
        # create the graph edges
        self._create_on_board_edges()
        self._create_dwell_edges()
        self._create_boarding_edges()
        self._create_alighting_edges()
        self._create_connector_edges()

        assert self.__on_board_edges is not None
        assert self.__boarding_edges is not None
        assert self.__alighting_edges is not None
        assert self.__dwell_edges is not None
        assert self.__connector_edges is not None

        edges: list[DataFrame] = [
            self.__on_board_edges,
            self.__boarding_edges,
            self.__alighting_edges,
            self.__dwell_edges,
            self.__connector_edges,
        ]

        # Stop pairs of a station get one edge, an outer transfer when both kinds are requested
        if self.with_outer_stop_transfers or self.with_walking_edges:
            self._create_station_edges("outer_transfer" if self.with_outer_stop_transfers else "walking")
            assert self.__station_edges is not None
            edges.append(self.__station_edges)

        # stack the dataframes on top of each other
        self.edges = pd.concat(edges, axis=0)

        # reset index and copy it to column
        self.edges = self.edges.reset_index(drop=True)
        self.edges["link_id"] = self.edges.index + 1
        for col, default in EDGE_DEFAULTS.items():
            self.edges[col] = self.edges[col].fillna(default) if col in self.edges else default
        self.edges = self.edges[SF_EDGE_COLS]
        self._set_edge_dtypes(self.edges)

    @staticmethod
    def _set_edge_dtypes(edges: pd.DataFrame):
        edges["line_id"] = edges.line_id.astype("string").fillna("").astype(str)
        edges["stop_id"] = edges.stop_id.astype("string").fillna("").astype(str)
        edges["segment_id"] = edges.segment_id.astype("string").fillna("").astype(str)
        edges["line_seg_idx"] = edges.line_seg_idx.fillna(-1).astype("int64")
        edges["a_node"] = edges.a_node.astype("int64")
        edges["b_node"] = edges.b_node.astype("int64")
        edges["trip_count"] = edges.trip_count.fillna(0).astype("int64")
        edges["direction"] = edges.direction.astype("int8")
        for col in ["run_time", "dwell_time", "walk_distance", "walk_time", "headway", "line_freq", "capacity"]:
            edges[col] = edges[col].astype(float)
        edges["line_freq"] = edges.line_freq.fillna(np.inf)

    def create_graph(self):
        """Create the SF transit graph (vertices and edges)."""

        all_present = True
        tables = ["stops", "routes", "route_links", "trips", "trips_schedule"]
        with self.scenario.transit_connection as conn:
            for table in tables:
                sql = f"SELECT COUNT(*) FROM {table}"
                all_present &= conn.execute(sql).fetchone()[0] > 0

        if not all_present:
            raise ValueError("cannot build a transit graph without a GTFS import")

        # FIXME: change these methods that create verticies and edges to standalone functions or static
        #  methods, and then call them here, and change in-place methods to non-inplace that return the dataframes
        self._create_vertices()
        self._create_edges()

    def _vertex_points(self) -> pd.Series:
        """Vertex geometries by node_id."""
        assert self.vertices is not None
        return pd.Series(shapely.from_wkb(self.vertices.geometry.to_numpy()), index=self.vertices.node_id.to_numpy())

    def create_line_geometry(self, method="direct", graph="w"):
        """
        Create the LineString for each edge, drawn from its tail to its head.

        The direct method creates a straight line between all points.

        The connect project match method uses the existing line geometry within the project to
        create more accurate line strings. It creates a line string that matches the path
        between the shortest path between the project nodes closest to either end of the access
        and egress connectors. Their walking distance and time become the geodesic length of that line.

        The "connector project match" method prepares its graph from the project network.

        :Arguments:
           **method** (:obj:`str`): Must be either `direct` or `connector project match`.
           If method is `direct`, `graph` argument is ignored.

           **graph** (:obj:`str`): Network mode for the path computation.
        """
        if method not in ["direct", "connector project match"]:
            raise ValueError("method must be either 'direct' or 'connector project match'")

        assert self.edges is not None
        assert self.vertices is not None
        points = self._vertex_points()

        def direct(rows):
            tails = shapely.get_coordinates(points.loc[self.edges.a_node[rows]].to_numpy())
            heads = shapely.get_coordinates(points.loc[self.edges.b_node[rows]].to_numpy())
            return shapely.to_wkb(shapely.linestrings(np.stack((tails, heads), axis=1)))

        if method == "direct":
            self.edges["geometry"] = direct(np.ones(len(self.edges), dtype=bool))
        elif method == "connector project match":
            # Check validity of project and nodes database
            logger.warning(
                'In its current implementation, the "connector project match" method may take '
                "a while for large networks."
            )

            nodes = self.scenario.network.nodes.data[["node_id", "geometry"]].set_index("node_id")
            links = self.scenario.network.links.data[["link_id", "geometry"]].set_index("link_id")

            if len(nodes) == 0:
                raise ValueError(
                    "No nodes found in the project database. Connector project matching "
                    "requires an existing project network."
                )

            # Create indexes for access and egress connectors
            connector_rows = (self.edges.link_type == "access_connector") | (self.edges.link_type == "egress_connector")
            other_rows = ~connector_rows

            self.edges["geometry"] = None
            self.edges.loc[other_rows, "geometry"] = direct(other_rows.to_numpy())

            lines = self.__connector_project_match(connector_rows, points, nodes, links, graph)
            distance = np.array([length for length, _ in lines])

            self.edges.loc[connector_rows, "geometry"] = [line for _, line in lines]
            self.edges.loc[connector_rows, "walk_distance"] = distance
            self.edges.loc[connector_rows, "walk_time"] = distance / self.walking_speed

    def __connector_project_match(self, connector_rows, points, nodes, links, graph_key):
        """Create line string geometry for `connector_rows` that matches the line strings in
        the graph for ``graph_key``.

        :Arguments:
           **connector_rows** (:obj:`pd.Series`): Boolean series for the rows of ``self.edges``
           to create line strings for.

           **points** (:obj:`pd.Series`): Vertex geometries by node_id.

           **nodes** (:obj:`pd.DataFrame`): A Dataframe containing the project nodes.
           Must have columns `geometry`, and an index of `node_id`\'s.

           **links** (:obj:`pd.DataFrame`): A Dataframe containing the project links.
           Must have columns `geometry`, and an index of `link_id`\'s.

           **graph_key** (:obj:`str`): The network mode to use
           for path finding.

        :Returns:
            **lines** (:obj:`list`): Geodesic length and WKB line string of each connector.
        """
        to_projected = self.__to_projected()
        # Create kdtree for fast nearest neighbour lookup on the project db nodes
        nodes["geometry"] = nodes["geometry"].apply(
            lambda geometry: shapely.transform(geometry, to_projected, include_z=None, interleaved=False)
        )
        links["geometry"] = links["geometry"].apply(
            lambda geometry: shapely.transform(geometry, to_projected, include_z=None, interleaved=False)
        )
        nodes_geometries = np.array(list(nodes["geometry"].apply(lambda geometry: (geometry.x, geometry.y))))
        kdtree = KDTree(nodes_geometries)

        # Prepare shortest path computation
        graph = self.scenario.network.build_graph(mode=graph_key, cost="distance")

        # Loop over connect edges, query for the closest nodes in the project and create the relevant line string
        lines = []
        assert self.edges is not None
        for row in self.edges[connector_rows].itertuples():
            start = shapely.transform(points.loc[row.a_node], to_projected, include_z=None, interleaved=False)
            end = shapely.transform(points.loc[row.b_node], to_projected, include_z=None, interleaved=False)

            _, ids = kdtree.query([[start.x, start.y], [end.x, end.y]], k=1)

            # If the ids for the closest nodes to the start and end are the same, then we make
            # an edge between those 3. If they differ we compute the shortest path between them.
            # If no path exists we use a straight between the start and end. Otherwise create a
            # line string using the already existing link geometry.
            if ids[0] == ids[1]:
                line = shapely.LineString((start, nodes.iloc[ids[0]].geometry, end))
            else:
                res = PathResults(graph, *nodes.iloc[ids].index.values)

                if res.path_nodes is not None:
                    line = chain_line(start, links.loc[res.path].geometry, end)
                else:
                    line = shapely.LineString((start, end))

            line = shapely.transform(line, self.transformer_p_to_g, include_z=None, interleaved=False)
            lines.append((geodesic_length(line), line.wkb))
        return lines

    def create_additional_db_fields(self, conn: sqlite3.Connection | None = None):
        """
        Create the additional required entries in the tables.

        :Arguments:
            **pt_conn** (:obj:`sqlite.Connection`): Optional PT connection to use
        """
        # This graph requires some additional tables and fields in order to store all our
        # information. Currently it mimics what we are storing in the df

        with conn or self.scenario.transit_connection as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO link_types (link_type, link_type_id, description)
                VALUES (?, ?, ?)
                """,
                [
                    ("on-board", "o", "From boarding to alighting"),
                    ("boarding", "b", "From stop to boarding"),
                    ("alighting", "a", "From alighting to stop"),
                    ("dwell", "d", "From alighting to boarding"),
                    ("access_connector", "A", ""),
                    ("egress_connector", "e", ""),
                    (
                        "inner_transfer",
                        "t",
                        "Transfer edge within station, from alighting to boarding",
                    ),
                    (
                        "outer_transfer",
                        "T",
                        "Transfer edge between the stops of a station, from stop to stop",
                    ),
                    ("walking", "w", "Walking, from stop or walking to stop or walking"),
                ],
            )

    def save_vertices(self, robust=None, conn: sqlite3.Connection | None = None):
        """
        Write the vertices DataFrame to the public transport database.

        Within the database nodes may not exist at the exact same point in space, provide
        ``robust=True`` to move the nodes slightly.

        :Arguments:
            **robust** (:obj:`bool`): Deprecated. No longer in use.

            **pt_conn** (:obj:`sqlite.Connection`): Optional PT connection to use
        """
        if robust is not None:
            warnings.warn(
                "the 'robust' argument is depreciated and no longer in use. Duplicate "
                "geometries are allowed within the public transport database.",
                DeprecationWarning,
                stacklevel=2,
            )

        with conn or self.scenario.transit_connection as conn:
            if conn.execute("SELECT node_id FROM nodes WHERE period_id=? LIMIT 1;", (self.period_id,)).fetchall():
                raise ValueError(
                    f"cannot save nodes into a database with existing nodes in the same period ({self.period_id})"
                )

            assert self.vertices is not None
            df = self.vertices[SF_VERTEX_COLS]
            conn.executemany(
                f"""\
                INSERT INTO nodes ({",".join(SF_VERTEX_COLS)},modes,period_id)
                VALUES({",".join("?" * (len(SF_VERTEX_COLS) - 1))},
                GeomFromWKB(?, {self.__global_crs.to_epsg()}),'t',{self.period_id});
                """,
                df.itertuples(index=False, name=None),
            )

    @staticmethod
    def remove_vertices(pt_conn: sqlite3.Connection, period_id: int):
        """
        Remove a transit graph's vertices from the public transport database specified by
        it's `period_id`.

        :Arguments:
            **pt_conn** (:obj:`sqlite3.Connection`): Connection to the
            ``public_transport.sqlite`` database.

            **period_id** (:obj:`int`): ``period_id`` to remove.

        """
        with pt_conn as conn:
            conn.execute("DELETE FROM nodes where period_id=?", (period_id,))

    def save_edges(self, recreate_line_geometry=False, conn: sqlite3.Connection | None = None):
        """
        Save the contents of self.edges to the public transport database.

        If no geometry for the edges is present or `recreate_line_geometry` is ``True``,
        direct lines will be created.

        :Arguments:
           **recreate_line_geometry** (:obj:`bool`): Whether to recreate the line strings for
           the edges as direct lines. Defaults to ``False``.

           **pt_conn** (:obj:`sqlite.Connection`): Optional PT connection to use
        """
        # We need to generate the geometry for each edge, this may take a bit
        assert self.edges is not None
        if "geometry" not in self.edges.columns or recreate_line_geometry:
            self.create_line_geometry()

        with conn or self.scenario.transit_connection as conn:
            if conn.execute("SELECT link_id FROM links WHERE period_id=? LIMIT 1;", (self.period_id,)).fetchall():
                raise ValueError("cannot save links into a database with existing links in the same period")

            df = self.edges[SF_EDGE_COLS + ["geometry"]].copy()
            df["stop_id"] = df.stop_id.astype(object)
            df.loc[df.stop_id == "", "stop_id"] = None
            conn.executemany(
                f"""\
                INSERT INTO links ({",".join(SF_EDGE_COLS)},geometry,modes,period_id)
                VALUES({",".join("?" * len(SF_EDGE_COLS))},
                GeomFromWKB(?, {self.__global_crs.to_epsg()}),'t',{self.period_id});
                """,
                df.itertuples(index=False, name=None),
            )

    @staticmethod
    def remove_edges(pt_conn: sqlite3.Connection, period_id: int):
        """
        Remove a transit graph's edges from the public transport database specified by
        it's `period_id`.

        :Arguments:
            **pt_conn** (:obj:`sqlite3.Connection`): Connection to the
            ``public_transport.sqlite`` database.

            **period_id** (:obj:`int`): `period_id` to remove.

        """
        with pt_conn as conn:
            conn.execute("DELETE FROM links WHERE period_id=?", (period_id,))

    def save_config(self, conn: sqlite3.Connection | None = None):
        with conn or self.scenario.db_connection as conn:
            sql = "INSERT OR REPLACE INTO transit_graph_configs (period_id,config) VALUES (?,?)"
            conn.execute(sql, [self.period_id, json.dumps(self.config)])

    @staticmethod
    def remove_config(conn: sqlite3.Connection, period_id: int):
        """
        Remove a transit graph configuration from the project database specified by
        it's `period_id`.

        :Arguments:
            **conn** (:obj:`sqlite3.Connection`): Connection to the ``project.sqlite`` database.

            **period_id** (:obj:`int`): `period_id` key for the `transit_graph_configs` table.

        """
        with conn as conn:
            sql = "DELETE FROM transit_graph_configs WHERE period_id = ?"
            conn.execute(sql, [period_id])

    def save(
        self,
        robust=None,
        pt_conn: sqlite3.Connection | None = None,
        project_conn: sqlite3.Connection | None = None,
    ):
        """Save the current graph to the public transport database.

        :Arguments:
            **robust** (:obj:`bool`): Deprecated. No longer in use.

            **pt_conn** (:obj:`sqlite.Connection`): Optional PT connection to use

            **project_conn** (:obj:`sqlite.Connection`): Optional project connection to use
        """
        if robust is not None:
            warnings.warn(
                "the 'robust' argument is deprecated and no longer in use. Duplicate geometries are allowed "
                "within the public transport database.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.create_additional_db_fields(conn=pt_conn)
        self.save_vertices(conn=pt_conn)
        self.save_edges(conn=pt_conn)
        self.save_config(conn=project_conn)

    @classmethod
    def remove(
        cls,
        pt_conn: sqlite3.Connection,
        project_conn: sqlite3.Connection,
        period_id: int,
    ):
        cls.remove_edges(pt_conn, period_id)
        cls.remove_vertices(pt_conn, period_id)
        cls.remove_config(project_conn, period_id)

    @property
    def provenance(self) -> dict:
        """What a result needs to record to trace it back to this graph."""
        return {
            "scenario": getattr(self.scenario, "name", None),
            "period_id": self.period_id,
            "graph_config": self.config,
            "graph_config_hash": config_hash(self.config),
        }

    def to_transit_graph(self) -> TransitGraph:
        """Create an AequilibraE ``TransitGraph`` object from an SF graph builder."""
        assert self.od_node_mapping is not None
        assert self.vertices is not None
        assert self.edges is not None
        g = TransitGraph(
            network=self.edges,
            centroids=self.vertices[
                (
                    (self.vertices.node_type == "origin")
                    if self.blocking_centroid_flows
                    else (self.vertices.node_type == "od")
                )
            ].node_id.values,
            od_node_mapping=self.od_node_mapping,
            a_node_field="a_node",
            b_node_field="b_node",
            config=self.config,
            provenance=self.provenance,
            costing=self._legacy_costing,
        )

        return g

    @classmethod
    def from_db(cls, project, period_id: int, **kwargs):
        """
        Create a SF graph instance from an existing database save.

        Assumes the database was constructed with the provided save methods.
        No checks are performed to see if the provided arguments are compatible with the saved
        graph.

        All arguments are forwarded to the constructor.

        :Arguments:
           **project** (:obj:`Project` or :obj:`Scenario`): AequilbraE project or scenario to use.

           **period_id** (:obj:`int`): Period ID to use.
        """
        scenario = getattr(project, "scenario", project)
        with scenario.db_connection as project_conn:
            config = project_conn.execute(
                "SELECT config FROM transit_graph_configs WHERE period_id = ? LIMIT 1;",
                [period_id],
            ).fetchone()

        if config is None:
            raise ValueError(f"no transit graph configuration found for 'period_id={period_id}'")

        config = json.loads(config[0])
        config.update(kwargs)

        with warnings.catch_warnings():
            # The saved configuration repeats deprecated arguments, which warned when the graph was built
            warnings.simplefilter("ignore", DeprecationWarning)
            graph = cls(project, **config)

        with scenario.transit_connection as public_transport_conn:
            graph.vertices = pd.read_sql_query(
                sql=(
                    f"SELECT {','.join(SF_VERTEX_COLS[:-1])}, "
                    f'ST_asBinary("geometry") as geometry FROM nodes '
                    f"WHERE period_id={period_id};"
                ),
                con=public_transport_conn,
            )

            graph.edges = pd.read_sql_query(
                sql=(
                    f"SELECT {','.join(SF_EDGE_COLS)}, "
                    f'ST_asBinary("geometry") as geometry FROM links '
                    f"WHERE period_id={period_id};"
                ),
                con=public_transport_conn,
            )

        graph.vertices.node_id = graph.vertices.node_id.astype("int64")
        graph.vertices.line_seg_idx = graph.vertices.line_seg_idx.astype("int64")
        graph.vertices.stop_id = graph.vertices.stop_id.astype("string").fillna("").astype(str)
        graph.vertices.line_id = graph.vertices.line_id.astype("string").fillna("").astype(str)

        cls._set_edge_dtypes(graph.edges)

        graph.create_od_node_mapping()

        return graph

    def convert_demand_matrix_from_zone_to_node_ids(
        self,
        demand_matrix,
        o_zone_col="origin_zone_id",
        d_zone_col="destination_zone",
        demand_col="demand",
    ):
        """Convert a sparse demand matrix from ``zone_id``\'s to the corresponding
        ``node_id``\'s."""
        assert self.od_node_mapping is not None
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
            od_matrix = od_matrix.rename(columns={"node_id": "o_node_id"})
            od_matrix = pd.merge(
                od_matrix,
                self.od_node_mapping[["node_id", "taz_id"]],
                left_on=d_zone_col,
                right_on="taz_id",
            )[["o_node_id", "node_id", "demand"]]
            od_matrix = od_matrix.rename(columns={"node_id": "d_node_id"})
        return od_matrix

    @property
    def config(self):
        return {k: getattr(self, k) for k in self.__config_attrs}


def config_hash(config: dict) -> str:
    """Hash of a graph configuration, identifying the settings a saved graph was built with."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def chain_line(start, lines, end) -> shapely.LineString:
    """Join lines in path order into one line from ``start`` to ``end``, reversing any drawn the other way."""
    coords = [shapely.get_coordinates(start)[0]]
    for line in lines:
        points = shapely.get_coordinates(line)
        if np.hypot(*(points[-1] - coords[-1])) < np.hypot(*(points[0] - coords[-1])):
            points = points[::-1]
        coords.extend(points)
    coords.append(shapely.get_coordinates(end)[0])
    return shapely.LineString(coords)


def geodesic_length(line_lonlat) -> float:
    """Length in metres of a line in longitude and latitude, measured as the walking edges are."""
    coords = shapely.get_coordinates(line_lonlat)
    if len(coords) < 2:
        return 0.0
    return float(haversine(coords[:-1, 0], coords[:-1, 1], coords[1:, 0], coords[1:, 1]).sum())
