from sqlite3 import Connection
from typing import List, Tuple, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from aequilibrae.paths import PathResults
from aequilibrae.transit.functions.get_srid import get_srid
from .basic_element import BasicPTElement
from .link import Link
from .mode_correspondence import mode_correspondence

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
        self.__mm_fail_position = -1
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
        shp = self._stop_based_shape if self.raw_shape is None else self.raw_shape
        return shp

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

        if not self.__feed.graphs:
            self.__feed.builds_link_graphs_with_broken_stops()
        if self.route_type not in mode_correspondence or mode_correspondence[self.route_type] not in self.__feed.graphs:
            return

        self.__map_matching_error.clear()
        df = self.__map_matching_complete_path_building()
        if df.shape[0] == 0:
            self.__logger.warning(f"Could not rebuild path for pattern {self.pattern_id}")
            return
        self.full_path = df.link_id.to_list()
        self.fpath_dir = df.dir.to_list()
        self.__assemble__mm_shape(df)
        self.__build_pattern_mapping()
        self.__logger.info(f"Map-matched pattern {self.pattern_id}")

    # TODO: consider improving the link selection for discount applying an overlay and use a cost proportional to the
    # link length in the route (raw_shape) buffer.
    def __graph_discount(self):
        buff = gpd.GeoSeries(self.raw_shape, crs="EPSG:4326").to_crs(3857).buffer(20).geometry
        gdf = gpd.GeoDataFrame(geometry=buff.to_crs(4326), crs=self.__geolinks.crs)
        gdf = self.__geolinks.overlay(gdf, how="intersection")

        gdf = gdf.loc[gdf.modes.str.contains(mode_correspondence[self.route_type])]
        return gdf.link_id.tolist()

    def __map_matching_complete_path_building(self):
        mode_ = mode_correspondence[self.route_type]
        # We get the graph for our job
        graph = self.__feed.graphs[mode_]
        empty_frame = pd.DataFrame([])

        # We search for disconnected stops:
        candidate_stops = list(self.stops)
        stop_node_idxs = [stop.__map_matching_id__[self.route_type] for stop in candidate_stops]

        node0 = graph.network.a_node[~graph.network.a_node.isin(graph.centroids)].min()
        connected_stops = []

        res = PathResults()
        res.prepare(graph)
        res1 = PathResults()
        res1.prepare(graph)

        for i, stop in enumerate(candidate_stops):
            node_o = stop.__map_matching_id__[self.route_type]
            self.__logger.debug(f"Computing paths between {node_o} and {node0}")
            res.compute_path(node_o, int(node0), early_exit=False)
            # Get skims, as proxy for connectivity, for all stops other than the origin
            other_nodes = stop_node_idxs[:i] + stop_node_idxs[i + 1 :]
            dest_skim = res.skims[other_nodes, 0]
            if dest_skim.min() < 1.0e308:
                candidate_stops = candidate_stops[i:]
                connected_stops = [stop for i, stop in enumerate(candidate_stops[1:]) if dest_skim[i] < 1.0e308]
                connected_stops = [candidate_stops[0]] + connected_stops
                break

        if not connected_stops:
            self.__logger.critical(f"Route completely disconnected. {self.route}/{self.route_id}")
            return empty_frame

        graph.cost = np.array(graph.graph.distance)
        likely_links = self.__graph_discount()
        graph.cost[graph.graph.original_id.abs().isin(likely_links)] *= 0.1

        fstop = connected_stops[0]

        if len(connected_stops) == 1:
            return empty_frame

        if len(connected_stops) == 2:
            nstop = connected_stops[1].__map_matching_id__[self.route_type]
            self.__logger.debug(f"Computing paths between {fstop.__map_matching_id__[self.route_type]} and {nstop}")
            res.compute_path(fstop.__map_matching_id__[self.route_type], int(nstop), early_exit=True)
            if res.milepost is None:
                return empty_frame
            pdist = list(res.milepost[1:-1] - res.milepost[:-2])[1:]
            plnks = list(res.path[1:-1])
            pdirecs = list(res.path_link_directions[1:-1])
            return self.__build_path_df(graph, pdirecs, pdist, plnks)

        path_links = []
        path_directions = []
        path_distances = []
        start = fstop.__map_matching_id__[self.route_type]
        for idx, tstop in enumerate(connected_stops[1:]):
            end = tstop.__map_matching_id__[self.route_type]

            not_last = idx + 2 <= len(connected_stops) - 1

            if not_last:
                following_stop = connected_stops[idx + 2]
                n_end = following_stop.__map_matching_id__[self.route_type]
            self.__logger.debug(f"Computing paths between {start} and {end}")
            res.compute_path(start, int(end), early_exit=True)
            connection_candidates = graph.network[graph.network.a_node == end].b_node.values
            min_cost = np.inf
            access_node = -1
            follow_val = 0
            for connec in connection_candidates:
                if connec == start:
                    continue
                if not_last:
                    res1.compute_path(int(connec), int(n_end), early_exit=True)
                    if res1.milepost is None:
                        continue
                    follow_val = res1.milepost[-1]
                estimate = follow_val + res.skims[connec, 0]
                if estimate < min_cost:
                    min_cost = estimate
                    access_node = connec
            if access_node >= 0:
                res.update_trace(int(access_node))
                shift = 1 if not_last else 0
                if len(res.path) <= 1 + shift:
                    # Stop connectors only
                    continue

                if not_last:
                    path_links.extend(list(res.path[:-1]))
                    path_directions.extend(list(res.path_link_directions[:-1]))
                    path_distances.extend(list(res.milepost[1:] - res.milepost[:-1])[1:])
                else:
                    path_links.extend(list(res.path[:]))
                    path_directions.extend(list(res.path_link_directions[:]))
                    path_distances.extend(list(res.milepost[1:] - res.milepost[:-1])[:])
            else:
                self.__logger.debug(f"Failed path computation when map-matching {self.pattern_id}")
                return empty_frame
            start = res.path_nodes[-2] if len(res.path_nodes) > 3 else start

        # Connection to the last stop
        return self.__build_path_df(graph, path_directions, path_distances, path_links)

    def __build_path_df(self, graph, path_directions, path_distances, path_links):
        corresp = pd.DataFrame(graph.network[["link_id", "original_id"]])
        if not path_links:
            return pd.DataFrame({"link_id": [], "dir": []})
        result = pd.DataFrame(
            {
                "link_id": path_links[1:],
                "direction": path_directions[1:],
                "sequence": np.arange(len(path_links) - 1),
                "distance": path_distances[1:],
            }
        )
        df = result.merge(corresp, on="link_id", how="left")
        df.sort_values(by=["sequence"], inplace=True)  # We just guarantee that we haven't messed up anything
        df = df[(df.original_id.shift(-1) != df.original_id) | (df.direction.shift(-1) != df.direction)]

        crit1 = df.original_id.shift(1) != df.original_id
        crit2 = df.original_id.shift(-1) != df.original_id
        df = df[(crit1 & crit2) | (df.distance > DEAD_END_RUN)]

        df = df[["original_id", "direction"]]
        df.columns = ["link_id", "dir"]
        df.loc[df.link_id > 0, "dir"] = 1
        df.loc[df.link_id < 0, "dir"] = -1
        df.reset_index(drop=True, inplace=True)
        has_issues = True
        while has_issues:
            # We eliminate multiple backs-and-forth on links
            has_issues = False
            for i in range(0, df.shape[0] - 2):
                if df.loc[i : i + 2, "link_id"].abs().unique().shape[0] == 1:
                    df.drop(index=[i, i + 1], inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    has_issues = True
                    break
        return df

    def __assemble__mm_shape(self, df: pd.DataFrame):
        shape = []  # type: List[Tuple[float, float]]

        for _, rec in df.iterrows():
            line_geo = self.__geolinks.loc[self.__geolinks.link_id == abs(rec.link_id)].geometry.values[0]
            coords = list(line_geo.coords)[::-1] if rec.link_id < 0 else list(line_geo.coords)
            data = coords[1:] if shape else coords
            shape.extend(data)
        self.shape = LineString(shape)

    def get_error(self, what_to_get="culprit") -> Optional[tuple]:
        """Returns information on the area of the network a map-matching error occurred

        :Arguments:
           *what_to_get* (:obj:`str`): The object you want returned. Options are 'culprit' and 'partial_path'

        :Returns:
        """
        if not self.__map_matching_error:
            self.__logger.debug("No map-matching error recorded for this pattern")
            return None

        if what_to_get not in self.__map_matching_error:
            return None
        return self.__map_matching_error[what_to_get]

    def __build_pattern_mapping(self):
        # We find what is the position along routes that we have for each stop and make sure they are always growing
        self.pattern_mapping = pd.DataFrame(
            {"seq": np.arange(len(self.full_path)), "link_id": np.abs(self.full_path), "dir": self.fpath_dir}
        )
        self.pattern_mapping = self.pattern_mapping.assign(pattern_id=self.pattern_id, srid=4326)
        links_with_geo = self.__geolinks.assign(wkb=self.__geolinks.geometry.to_wkb())
        links_with_geo = links_with_geo[["link_id", "wkb"]]

        self.pattern_mapping = self.pattern_mapping.merge(links_with_geo, on="link_id", how="left")
