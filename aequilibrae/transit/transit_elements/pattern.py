from sqlite3 import Connection
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.ops
from shapely.geometry import LineString
from shapely.ops import transform

from aequilibrae.transit.functions.get_srid import get_srid
from .basic_element import BasicPTElement
from .link import Link
from .mode_correspondence import mode_corresp

DEAD_END_RUN = 40


class Pattern(BasicPTElement):
    """
    Represents a stop pattern for a particular route, as defined in GTFS.
    """

    def __init__(self, route_id, gtfs_feed) -> None:
        """
        :Arguments:
            *route_id* (:obj:`str`): route ID for which this stop pattern belongs

            *gtfs_feed* (:obj:`Geo`): Parent feed object
        """
        self.pattern_hash = ""
        self.pattern_id = -1
        self.route_id = route_id
        self.route = ""
        self.agency_id = None
        self.longname = ""
        self.shortname = ""
        self.description = ""
        self.pce = 2.0
        self.seated_capacity = None
        self.total_capacity = None
        self.__srid = get_srid()
        self.__geolinks = gtfs_feed.geo_links
        self.__logger = gtfs_feed.logger

        self.__feed = gtfs_feed
        # For map matching
        self.raw_shape: LineString = None
        self._stop_based_shape: LineString = None
        self.shape: LineString = None
        self.route_type: int = None
        self.links: List[Link] = []
        self.network_candidates = []
        self.full_path: List[int] = []
        self.fpath_dir: List[int] = []
        self.pattern_mapping = pd.DataFrame([])
        self.stops = []
        self.__map_matching_error = {}

        self.__graph = None
        self.__res = None
        self.__curr_net_nodes_from_stops = []
        self.__net_links_from_stops = []
        self.__net_nodes_from_stops = []
        self.__map_matched = False
        self.shape_length = -1

    def save_to_database(self, conn: Connection, commit=True) -> None:
        """Saves the pattern to the routes table"""

        shp = self.best_shape()
        geo = None if shp is None else shp.wkb

        data = [
            self.pattern_id,
            self.route_id,
            self.route,
            self.agency_id,
            self.shortname,
            self.longname,
            self.description,
            self.route_type,
            self.pce,
            self.seated_capacity,
            self.total_capacity,
            geo,
            self.__srid,
        ]

        sql = """insert into routes (pattern_id, route_id, route, agency_id, shortname, longname, description, route_type, pce,
                         seated_capacity, total_capacity, geometry) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ST_Multi(GeomFromWKB(?, ?)));"""
        conn.execute(sql, data)

        if self.pattern_mapping.shape[0]:
            sqlgeo = """insert into pattern_mapping (pattern_id, seq, link, dir, geometry)
                        values (?, ?, ?, ?, GeomFromWKB(?, ?));"""
            sql = "insert into pattern_mapping (pattern_id, seq, link, dir) values (?, ?, ?, ?);"

            if "wkb" in self.pattern_mapping.columns:
                cols = ["pattern_id", "seq", "link_id", "dir", "wkb", "srid"]
                data = self.pattern_mapping[cols].to_records(index=False)
                conn.executemany(sqlgeo, data)
            else:
                data = self.pattern_mapping[["pattern_id", "seq", "link_id", "dir"]].to_records()
                conn.executemany(sql, data)
        if commit:
            conn.commit()

    def best_shape(self) -> LineString:
        """Gets the best version of shape available for this pattern"""
        if self.shape is None:
            return self._stop_based_shape if self.raw_shape is None else self.raw_shape
        return self.shape

    def map_match(self):
        """Map matches the route into the network, considering its appropriate shape.

        Part of the map-matching process is to find the network links corresponding the pattern's
        raw shape, so that method will be called in case it has not been called before.

        The basic algorithm behind the map-matching algorithm is described in https://doi.org/10.3141%2F2646-08

        In a nutshell, we compute the shortest path between the nodes corresponding to the links to which
        stops were geographically matched, for each pair of identified links.

        We do not consider links that are in perfect sequence, as we found that it introduces severe issues when
        stops are close to intersections without clear upsides.

        When issues are found, we remove the stops in the immediate vicinity of the issue and attempt new
        path finding. The First and last stops/corresponding links are always kept.

        If an error was found, (record for it will show in the log), it is stored within the object.

        """
        if self.__map_matched:
            return
        self.__map_matched = True
        self.__logger.debug(f"Map-matching pattern ID {self.pattern_id}")

        if not self.__feed.map_matchers:
            self.__feed.builds_map_matchers()
        if self.route_type not in mode_corresp or mode_corresp[self.route_type] not in self.__feed.map_matchers:
            return

        df = pd.DataFrame({"stop_id": [stop.stop_id for stop in self.stops],
                           "geometry": [stop.geo for stop in self.stops]})
        map_matcher = self.__feed.map_matchers[mode_corresp[self.route_type]]  # type: RouteMapMatcher

        stops = gpd.GeoDataFrame(df, geometry="geometry", crs=f"EPSG:{self.__srid}").to_crs(map_matcher.crs)

        route_shape = self.raw_shape
        if route_shape is not None:
            route_shape = transform(self.__feed.mm_transformer.transform, route_shape)
        df = map_matcher.map_match_route(stops, route_shape, self.pattern_id)

        if df.shape[0] == 0:
            self.__logger.warning(f"Could not rebuild path for pattern {self.pattern_id}")
            return

        self.shape = shapely.ops.transform(self.__feed.mm_transform_rev.transform, map_matcher.assemble_shape(df))
        self.__build_pattern_mapping(df)
        self.__logger.debug(f"Map-matched pattern {self.pattern_id}")

    def __build_pattern_mapping(self, df):
        # We find what is the position along routes that we have for each stop and make sure they are always growing
        df_net = df.assign(seq=np.arange(df.shape[0]), pattern_id=self.pattern_id, srid=4326)

        df_net = df_net.merge(self.__geolinks[["link_id", "geometry"]], on="link_id", how="inner")
        df_net.sort_values("seq", inplace=True)
        df_net = gpd.GeoDataFrame(df_net, geometry="geometry", crs=self.__geolinks.crs)
        df_net = df_net.assign(wkb=df_net.geometry.to_wkb()).drop(columns=["geometry"])
        self.pattern_mapping = df_net[["pattern_id", "seq", "link_id", "wkb", "dir", "srid"]]
