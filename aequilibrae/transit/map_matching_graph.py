import hashlib
from contextlib import closing
from copy import deepcopy
import math
from os.path import isfile, join
from tempfile import gettempdir
import importlib.util as iutil
from ..utils import WorkerThread

import numpy as np
import pandas as pd
import shapely.wkt
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.ops import substring

from aequilibrae.log import logger
from aequilibrae.project.database_connection import database_connection
from aequilibrae.transit.constants import DRIVING_SIDE
from aequilibrae.project.zoning import GeoIndex
from aequilibrae.transit.transit_elements import mode_correspondence
from aequilibrae.transit.functions.compute_line_bearing import compute_line_bearing

spec = iutil.find_spec("PyQt5")
pyqt = spec is not None
if pyqt:
    from PyQt5.QtCore import pyqtSignal

GRAPH_VERSION = 1
CONNECTOR_SPEED = 1


class MMGraph(WorkerThread):
    """Build specialized map-matching graphs. Not designed to be used by the final user"""

    if pyqt:
        signal = pyqtSignal(object)

    def __init__(self, lib_gtfs, mtmm):
        WorkerThread.__init__(self, None)
        self.geotool = lib_gtfs.geotool
        self.stops = lib_gtfs.gtfs_data.stops
        self.lib_gtfs = lib_gtfs
        self.__mtmm = mtmm
        self._idx = None
        self.max_link_id = -1
        self.max_node_id = -1
        self.mode = ""
        self.modename = ""
        self.mode_id = -1
        self.__mode = ""
        self.__df_file = ""
        self.__agency = lib_gtfs.gtfs_data.agency.agency
        self.__centroids_file = ""
        self.__mm_graph_file = ""
        self.node_corresp = []
        self.__all_links = {}
        self.distance_to_project = -1
        self.df = pd.DataFrame([])
        self.logger = logger

    def build_graph_with_broken_stops(self, mode_id: int, distance_to_project=200):
        """Build the graph for links for a certain mode while splitting the closest links at stops' projection

        :Arguments:
            **mode_id** (:obj:`int`): Mode ID for which we will build the graph for
            **distance_to_project** (:obj:`float`, `Optional`): Radius search for links to break at the stops. Defaults to 50m
        """
        self.logger.debug(f"Called build_graph_with_broken_stops for mode_id={mode_id}")
        self.mode_id = mode_id
        self.distance_to_project = distance_to_project
        self.__mode = mode_correspondence[self.mode_id]
        self.__mm_graph_file = join(gettempdir(), f"map_matching_graph_{self.__agency}_{self.__mode}.csv")
        modename = self.geotool.network.modes.get(self.__mode).mode_name

        with closing(database_connection("network")) as conn:
            get_qry = f"""Select link_id, a_node, b_node, max(speed_ab, speed_ba) speed,
                          distance, ST_AsText(geometry) wkt from links
                          WHERE INSTR(links.modes, "{self.__mode}")>0 AND direction>=0
                          UNION ALL
                          Select  link_id * -1 , b_node a_node, a_node b_node, max(speed_ab, speed_ba) speed,
                          distance, ST_AsText(ST_Reverse(geometry)) wkt from links
                          WHERE INSTR(links.modes, "{self.__mode}")>0 AND direction<=0;"""

            self.logger.debug("  Reading links from physical network")
            self.df = pd.read_sql(get_qry, conn)

        if not self.df.shape[0]:
            from aequilibrae.paths import Graph

            return Graph()

        # We do some wrangling to save the graph to disk, in case we need to run this more than once
        authvalue = hashlib.md5()
        authvalue.update(str(distance_to_project).encode())
        authvalue.update(str(GRAPH_VERSION).encode())
        authvalue.update("".join([str(x) for x in self.stops.keys()]).encode())
        authvalue.update("".join([str(x) for x in self.df.link_id.values]).encode())
        authvalue.update(modename.encode())
        graph_hash = authvalue.hexdigest()

        self.__df_file = join(gettempdir(), f"{graph_hash}_df.zip")
        self.__centroids_file = join(gettempdir(), f"{graph_hash}_centroids.zip")
        if isfile(self.__df_file) and isfile(self.__centroids_file):
            return self.__build_graph_from_cache()

        return self.__build_graph_from_scratch()

    def __build_graph_from_cache(self):
        self.logger.info(f"Loading map-matching graph from disk for mode_id={self.mode_id}")
        net = pd.read_csv(self.__df_file)
        centroid_corresp = pd.read_csv(self.__centroids_file)
        centroids = np.copy(centroid_corresp.centroid_id.values)
        centroid_corresp.set_index("node_id", inplace=True)
        for stop in self.stops.values():
            stop.___map_matching_id__[self.mode_id] = centroid_corresp.loc[stop.stop_id, "centroid_id"]
        return self.__graph_from_broken_net(centroids, net)

    def __build_graph_from_scratch(self):
        self.logger.info(f"Creating map-matching graph from scratch for mode_id={self.mode_id}")
        self.df = self.df.assign(original_id=self.df.link_id, is_connector=0, geo=np.nan)
        self.df.loc[:, "geo"] = self.df.wkt.apply(shapely.wkt.loads)
        self.df.loc[self.df.link_id < 0, "link_id"] = self.df.link_id * -1 + self.df.link_id.max() + 1
        # We make sure all link IDs are in proper order

        self.max_link_id = self.df.link_id.max() + 1
        self.max_node_id = self.df[["a_node", "b_node"]].max().max() + 1
        # Build initial index
        if pyqt:
            self.signal.emit(["start", "secondary", self.df.shape[0], f"Indexing links - {self.__mode}", self.__mtmm])
        self._idx = GeoIndex()
        for counter, (_, record) in enumerate(self.df.iterrows()):
            if pyqt:
                self.signal.emit(["update", "secondary", counter + 1, f"Indexing links - {self.__mode}", self.__mtmm])
            self._idx.insert(feature_id=record.link_id, geometry=record.geo)
        # We will progressively break links at stops' projection
        # But only on the right side of the link (no boarding at the opposing link's side)
        centroids = []
        self.node_corresp = []
        if pyqt:
            self.signal.emit(["start", "secondary", len(self.stops), f"Breaking links - {self.__mode}", self.__mtmm])
        self.df = self.df.assign(direction=1, free_flow_time=np.inf, wrong_side=0, closest=1, to_remove=0)
        self.__all_links = {rec.link_id: rec for _, rec in self.df.iterrows()}
        for counter, (stop_id, stop) in enumerate(self.stops.items()):
            if pyqt:
                self.signal.emit(["update", "secondary", counter + 1, f"Breaking links - {self.__mode}", self.__mtmm])
            stop.___map_matching_id__[self.mode_id] = self.max_node_id
            self.node_corresp.append([stop_id, self.max_node_id])
            centroids.append(stop.___map_matching_id__[self.mode_id])
            self.max_node_id += 1
            self.connect_node(stop)
        self.df = pd.concat([pd.DataFrame(rec).transpose() for rec in self.__all_links.values()])

        self.df = self.df[self.df.to_remove == 0]
        fltr = self.df.speed > 0
        self.df.loc[fltr, "free_flow_time"] = self.df.distance[fltr] / self.df.speed[fltr]

        # gets around AequilibraE bug
        # https://github.com/AequilibraE/aequilibrae/issues/307
        max_node_id = self.df[["a_node", "b_node"]].max().max()
        rec = deepcopy(self.df.iloc[[self.df.index.values[0]]])
        rec.a_node = max_node_id + 1
        rec.b_node = max_node_id
        rec.link_id = self.df.link_id.max() + 1
        rec.direction = 1
        self.df = pd.concat([self.df, rec], ignore_index=True)
        # End of upstream bug treatment

        cols = [
            "link_id",
            "a_node",
            "b_node",
            "direction",
            "free_flow_time",
            "distance",
            "is_connector",
            "closest",
            "original_id",
        ]
        net = self.df[cols]
        # Caches the graph outputs
        net.reset_index(inplace=True, drop=True)
        net = net.drop_duplicates(subset=["a_node", "b_node"])
        net.to_csv(self.__df_file, index=False)
        cols.append("wkt")
        self.df[cols].to_csv(self.__mm_graph_file, index=False)
        pd.DataFrame(self.node_corresp, columns=["node_id", "centroid_id"]).to_csv(self.__centroids_file)
        return self.__graph_from_broken_net(centroids, net)

    def connect_node(self, stop) -> None:
        list_nearest = list(self._idx.nearest(stop.geo, 30))

        is_closest = 1
        conn_found = 0
        distances = []
        for lid in list_nearest:
            lgeo = self.__all_links[lid].geo
            distances.append(stop.geo.distance(lgeo) * math.pi * 6371000 / 180)

        # Sort by distance to the stop
        nearest = pd.DataFrame({"dist": distances, "link_id": list_nearest}).sort_values(by="dist")
        criterium = min(5 * nearest.dist.min(), self.distance_to_project)
        criterium = criterium if nearest.dist.min() < 250 else nearest.dist.min()
        nearest = nearest[nearest.dist <= max(criterium, nearest.dist.min())].link_id.tolist()

        for counter, link_id in enumerate(nearest):
            wrong_side = 0
            link = self.__all_links[link_id]

            link_geo = link.geo  # Linestring

            # We disregard links beyond the threshold, but maintain the closest link to ensure connectivity

            if link_geo.boundary.is_empty:
                first = Point([link_geo.coords.xy[0][0], link_geo.coords.xy[1][0]])
                last = Point([link_geo.coords.xy[0][-1], link_geo.coords.xy[1][-1]])
            else:
                first = link_geo.boundary.geoms[0]
                last = link_geo.boundary.geoms[1]

            proj_point = link_geo.project(stop.geo)
            corr_proj = proj_point * math.pi * 6371000 / 180
            break_point = link_geo.interpolate(proj_point)
            connector_geo = LineString([stop.geo, break_point])
            conn_length = connector_geo.length * math.pi * 6371000 / 180

            if conn_length > 0:
                p = break_point if corr_proj > 0 else last
                az_link = compute_line_bearing((first.x, first.y), (p.x, p.y))
                az_connector = compute_line_bearing((stop.geo.x, stop.geo.y), (break_point.x, break_point.y))
                if (az_link - az_connector) * DRIVING_SIDE < 0:
                    wrong_side = 1  # We are on the WRONG side

            if corr_proj <= 1.0:  # If within one meter of the end of the link, let's go with the existing node
                break_point = first
                intersec_node = link.a_node
            elif corr_proj >= (link_geo.length * math.pi * 6371000 / 180) - 1.0:
                break_point = last
                intersec_node = link.b_node
            else:
                link.to_remove = 1
                self._idx.delete(link_id, link_geo)
                intersec_node = self.max_node_id
                self.max_node_id += 1

                # Create the first portion of the link
                fp = deepcopy(link)
                fp.link_id = self.max_link_id
                fp.b_node = intersec_node
                fp.to_remove = 0
                fp.geo = substring(link_geo, 0, proj_point)
                fp.wkt = fp.geo.wkt
                fp.distance = fp.geo.length * math.pi * 6371000 / 180
                self._idx.insert(fp.link_id, fp.geo)
                self.max_link_id += 1
                self.__all_links[fp.link_id] = fp

                # Create the second portion of the link
                lp = deepcopy(link)
                lp.link_id = self.max_link_id
                lp.a_node = intersec_node
                lp.geo = substring(link_geo, proj_point, link_geo.length)
                lp.wkt = lp.geo.wkt
                lp.distance = lp.geo.length * math.pi * 6371000 / 180
                lp.to_remove = 0
                self._idx.insert(lp.link_id, lp.geo)
                self.max_link_id += 1
                self.__all_links[lp.link_id] = lp

            # Add the connector
            connector_geo = LineString([stop.geo, break_point])
            connector = deepcopy(link)
            connector.link_id = self.max_link_id
            connector.original_id = -1
            connector.a_node = stop.___map_matching_id__[self.mode_id]
            connector.b_node = intersec_node
            connector.wrong_side = wrong_side
            connector.direction = 0
            connector.to_remove = 0
            connector.geo = connector_geo
            connector.wkt = connector_geo.wkt
            connector.distance = connector_geo.length * math.pi * 6371000 / 180
            connector.is_connector = 1  # We make sure that the closest connector cannot be turned off
            connector.closest = is_closest
            self.max_link_id += 1
            is_closest = 0
            conn_found += 1
            self.__all_links[connector.link_id] = connector

    def __graph_from_broken_net(self, centroids, net):
        from aequilibrae.paths import Graph

        net_data = pd.DataFrame(
            {
                "distance": net.distance.astype(float),
                "direction": net.direction.astype(np.int8),
                "a_node": net.a_node.astype(int),
                "b_node": net.b_node.astype(int),
                "link_id": net.link_id.astype(int),
                "is_connector": net.is_connector.astype(int),
                "original_id": net.original_id.astype(int),
                "closest": net.closest.astype(int),
                "free_flow_time": net.free_flow_time.astype(float),
            }
        )

        fltr = net.is_connector == 1
        net_data.loc[fltr, "distance"] = 1.2 * (net_data[fltr].distance ** 1.3)  # Already penalize it a bit
        net_data.loc[fltr, "free_flow_time"] = ((net_data[fltr].distance / 1000) / CONNECTOR_SPEED) * 60

        g = Graph()
        g.network_ok = True
        g.status = "OK"
        g.network = net_data
        g.prepare_graph(np.array(centroids))
        g.set_graph("free_flow_time")
        g.set_skimming(["distance"])
        g.set_blocked_centroid_flows(True)
        return g

    def finished(self):
        if pyqt:
            self.signal.emit(["finished_building_mm_graph_procedure"])
