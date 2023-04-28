import math
from sqlite3 import Connection
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import shapely.wkb
from shapely.geometry import LineString, Polygon
from shapely.ops import substring

from aequilibrae.log import logger
from aequilibrae.transit.functions.get_srid import get_srid
from aequilibrae.utils.geo_index import GeoIndex
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
        Args:
            *pattern_id* (:obj:`str`): Pre-computed ID for this pattern
            *geotool* (:obj:`Geo`): Suite of geographic utilities.
        """
        self.pattern_hash = ""
        self.pattern_id = -1
        self.route_id = route_id
        self.route = ""
        self.agency_id = None
        self.longname = ""
        self.shortname = ""
        self.description = ""
        self.seated_capacity = None
        self.total_capacity = None
        self.__srid = get_srid()
        self.__geotool = gtfs_feed.geotool
        self.__geolinks = self.__geotool.network.links.data
        self.__logger = logger

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
        self.pattern_mapping = []
        self.stops = []
        self.__map_matching_error = {}

        self.__graph = None
        self.__res = None
        self.__curr_net_nodes_from_stops = []
        self.__net_links_from_stops = []
        self.__net_nodes_from_stops = []
        self.__mm_fail_position = -1
        self.__map_matched = False
        self.__match_quality = None
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
            self.seated_capacity,
            self.total_capacity,
            geo,
            self.__srid,
        ]

        sql = """insert into routes (pattern_id, route_id, route, agency_id, shortname, longname, description, route_type, seated_capacity,
                         total_capacity, geometry) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ST_Multi(GeomFromWKB(?, ?)));"""
        conn.execute(sql, data)

        if self.pattern_mapping and self.shape:
            sqlgeo = """insert into pattern_mapping (pattern_id, seq, link, dir, geometry)
                        values (?, ?, ?, ?, GeomFromWKB(?, ?));"""
            sql = """insert into pattern_mapping (pattern_id, seq, link, dir)
                                                  values (?, ?, ?, ?);"""

            for record in self.pattern_mapping:
                if record[-1] is None:
                    conn.execute(sql, record[:-1])
                else:
                    geo = shapely.wkb.loads(record[-1])
                    if isinstance(geo, LineString):
                        conn.execute(sqlgeo, record + [self.__srid])
                    else:
                        conn.execute(sql, record[:-1])

    def best_shape(self) -> LineString:
        """Gets the best version of shape available for this pattern"""
        shp = self._stop_based_shape if self.raw_shape is None else self.raw_shape
        shp = shp if self.shape is None else self.shape
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
        self.__logger.debug(f"Map-matching pattern ID {self.pattern_id}")

        if not self.__feed.graphs:
            self.__feed.builds_link_graphs_with_broken_stops()

        if not mode_correspondence[self.route_type] in self.__feed.graphs:
            return

        self.__feed.path_store.add_graph(
            self.__feed.graphs[mode_correspondence[self.route_type]], mode_correspondence[self.route_type]
        )

        self.__map_matching_error.clear()
        df = self.__map_matching_complete_path_building()
        if df.shape[0] == 0:
            logger.warning(f"Could not rebuild path for pattern {self.pattern_id}")
            return
        self.full_path = df.link_id.to_list()
        self.fpath_dir = df.dir.to_list()
        self.__assemble__mm_shape(df)
        self.__build_pattern_mapping()
        logger.info(f"Map-matched pattern {self.pattern_id}")

    def __graph_discount(self, connected_stops):
        self.__geolinks = self.__geolinks[self.__geolinks.modes.str.contains(mode_correspondence[self.route_type])]
        link_idx = GeoIndex()
        for _, row in self.__geolinks.iterrows():
            link_idx.insert(feature_id=row.link_id, geometry=row.geometry)
        links = set()
        for stop in connected_stops:
            links.update(list(link_idx.nearest(stop.geo, 3)))
        buffer = self.best_shape().buffer(40)  # type: Polygon

        return [
            lnk
            for lnk in links
            if buffer.contains(self.__geolinks.loc[self.__geolinks.link_id == lnk].geometry.values[0])
        ]

    def __map_matching_complete_path_building(self):
        mode_ = mode_correspondence[self.route_type]
        # We get the graph for our job
        graph = self.__feed.graphs[mode_]
        empty_frame = pd.DataFrame([])

        # We search for disconnected stops:
        candidate_stops = [stop for stop in self.stops]
        stop_node_idxs = [stop.___map_matching_id__[self.route_type] for stop in candidate_stops]

        node0 = graph.network.a_node[~graph.network.a_node.isin(graph.centroids)].min()
        connected_stops = []
        for i, stop in enumerate(candidate_stops):
            node_o = stop.___map_matching_id__[self.route_type]
            logger.debug(f"Computing paths between {node_o} and {node0}")
            res = self.__feed.path_store.get_path_results(node_o, mode_)
            res.update_trace(int(node0))

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
        likely_links = self.__graph_discount(connected_stops)
        graph.cost[graph.graph.original_id.abs().isin(likely_links)] *= 0.1

        fstop = connected_stops[0]

        if len(connected_stops) == 1:
            return empty_frame

        if len(connected_stops) == 2:
            nstop = connected_stops[1].___map_matching_id__[self.route_type]
            logger.debug(f"Computing paths between {fstop.___map_matching_id__[self.route_type]} and {nstop}")
            res = self.__feed.path_store.get_path_results(fstop.___map_matching_id__[self.route_type], mode_)
            res.update_trace(int(nstop))
            if res.milepost is None:
                return empty_frame
            pdist = list(res.milepost[1:-1] - res.milepost[:-2])[1:]
            plnks = list(res.path[1:-1])
            pdirecs = list(res.path_link_directions[1:-1])
            return self.__build_path_df(graph, pdirecs, pdist, plnks)

        path_links = []
        path_directions = []
        path_distances = []
        start = fstop.___map_matching_id__[self.route_type]
        for idx, tstop in enumerate(connected_stops[1:]):
            end = tstop.___map_matching_id__[self.route_type]

            not_last = idx + 2 <= len(connected_stops) - 1

            if not_last:
                following_stop = connected_stops[idx + 2]
                n_end = following_stop.___map_matching_id__[self.route_type]
            logger.debug(f"Computing paths between {start} and {end}")
            res = self.__feed.path_store.get_path_results(start, mode_)
            res.update_trace(int(end))
            connection_candidates = graph.network[graph.network.a_node == end].b_node.values
            min_cost = np.inf
            access_node = -1
            follow_val = 0
            for connec in connection_candidates:
                if connec == start:
                    continue
                if not_last:
                    res1 = self.__feed.path_store.get_path_results(connec, mode_)
                    res1.update_trace(int(n_end))
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
        df.loc[df.link_id > 0, "dir"] = 0
        df.loc[df.link_id < 0, "dir"] = 1
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
        """
           Returns information on the  area of the network a map-matching error occurred

        Args:

           *what_to_get* (:obj:`str`): The object you want returned. Options are 'culprit' and 'partial_path'

           :return:
        """
        if not self.__map_matching_error:
            self.__logger.debug("No map-matching error recorded for this pattern")
            return None

        if what_to_get not in self.__map_matching_error:
            return None
        return self.__map_matching_error[what_to_get]

    def __build_pattern_mapping(self):
        # We find what is the position along routes that we have for each stop and make sure they are always growing
        self.pattern_mapping = []
        segments = [LineString([pt1, pt2]) for pt1, pt2 in zip(self.shape.coords, self.shape.coords[1:])]
        seg_len = [seg.length for seg in segments]
        distances = []

        min_idx_so_far = 0
        for i, stop in enumerate(self.stops):
            d = [round(stop.geo.distance(line), 1) for line in segments]
            idx = d[min_idx_so_far:].index(min(d[min_idx_so_far:])) + min_idx_so_far
            idx = idx if i > 0 else 0
            projection = segments[idx].project(stop.geo) + sum(seg_len[:idx])
            distances.append(projection)
            min_idx_so_far = idx

        # the list needs to be monotonically increasing
        if not all(x <= y for x, y in zip(distances, distances[1:])):
            for i, (x, y) in enumerate(zip(distances, distances[1:])):
                distances[i + 1] = distances[i] if y < x else distances[i + 1]

            logger.warning(
                f"""Pattern {self.pattern_id} has a map-matching pattern resulting in backwards movement.
                            It was fixed, but it should be checked."""
            )

        distances[-1] = distances[-1] if self.shape.length > distances[-1] else self.shape.length - 0.00001

        # We make sure we don't have projections going beyond the length we will accumulate
        # This is only required because of numerical precision issues
        tot_broken_length = sum(
            [self.__geolinks.loc[self.__geolinks.link_id == abs(x)].geometry.values[0].length for x in self.full_path]
        )
        distances = [min(x, tot_broken_length) for x in distances]
        pid = self.pattern_id
        stop_pos = 0
        cum_dist = 0
        index = 0
        for link_pos, link_id in enumerate([abs(x) for x in self.full_path]):
            direc = self.fpath_dir[link_pos]
            link_geo = self.__geolinks.loc[self.__geolinks.link_id == link_id].geometry.values[0]
            link_geo = link_geo if direc == 0 else LineString(link_geo.coords[::-1])
            if distances[stop_pos] > cum_dist + link_geo.length:
                # If we have no stops falling right in this link
                dt = [pid, index, link_id, direc, link_geo.wkb]
                self.pattern_mapping.append(dt)
                index += 1
                cum_dist += link_geo.length
                continue

            start_point = cum_dist
            end_point = cum_dist + link_geo.length
            while cum_dist + link_geo.length >= distances[stop_pos]:
                milepost = distances[stop_pos]
                wkb = None if stop_pos == 0 else substring(self.shape, start_point, milepost).wkb
                stop = self.stops[stop_pos]
                dt = [pid, index, link_id, direc, wkb]
                self.pattern_mapping.append(dt)
                index += 1

                stop_pos += 1
                if stop_pos >= len(distances):
                    break
                start_point = milepost

            cum_dist += link_geo.length

            if start_point != end_point:
                wkb = substring(self.shape, start_point, end_point).wkb
                self.pattern_mapping.append([pid, index, link_id, direc, wkb])
                index += 1

            if stop_pos >= len(distances):
                break
        self.__compute_match_quality()

    def __compute_match_quality(self):
        dispersion = [(self.shape.distance(stop.geo) * math.pi * 6371000 / 180) for stop in self.stops]
        dispersion = sum([x * x for x in dispersion]) / len(self.stops)
        self.__match_quality = dispersion
