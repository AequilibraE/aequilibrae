import geopandas as gpd
import numpy as np
import pandas as pd
from aequilibrae.paths import Graph
from aequilibrae.transit.functions.breaking_links_for_stop_access import split_links_at_stops
from aequilibrae.utils.geo_utils import metre_crs_for_gdf
from aequilibrae.utils.interface.worker_thread import WorkerThread


class MMGraph(WorkerThread):
    """Build specialized map-matching graphs. Not designed to be used by the final user"""

    def __init__(self, link_gdf: gpd.GeoDataFrame, nodes_gdf: gpd.GeoDataFrame, stops_gdf: gpd.GeoDataFrame,
                 distance_to_project=50):
        WorkerThread.__init__(self, None)

        utm_zone = metre_crs_for_gdf(link_gdf)
        self.links = self.__rename_geo(link_gdf).to_crs(utm_zone)
        self.stops = self.__rename_geo(stops_gdf).to_crs(utm_zone)
        self.nodes = self.__rename_geo(nodes_gdf).to_crs(utm_zone)

        self.dist_thresh = distance_to_project
        self.node_corresp = []
        self.__all_links = {}
        self.graph = Graph()

    @staticmethod
    def __rename_geo(gdf):
        if gdf.active_geometry_name != "geometry":
            return gdf.rename_geometry("geometry")
        return gdf

    def build_graph_with_broken_stops(self):
        """Build the graph for links for a certain mode while splitting the closest links at stops' projection

        :Arguments:
            **mode_id** (:obj:`int`): Mode ID for which we will build the graph for

            **distance_to_project** (:obj:`float`, *Optional*): Radius search for links to break at the stops.
            Defaults to 50m
        """
        self.logger.debug(f"Called build_graph_with_broken_stops for mode_id={mode_id}")

        if not self.links.shape[0]:
            return Graph()

        self.__build_graph_from_scratch()
        return self.graph

    def __build_graph_from_scratch(self):
        self.logger.debug(f"Creating map-matching graph")

        broken_links, new_nodes = split_links_at_stops(self.links, self.stops, self.dist_thresh)

        # To build connectors, let's get all nodes together
        # and connect stops to them within the threshold distance
        nodes = pd.concat([self.nodes[["node_id", "geometry"]], new_nodes], ignore_index=True)

        stops = self.stops[["stop_id", "geometry"]]
        joined = stops.sjoin_nearest(nodes, how="left", distance_col="dist_to_node", max_distance=self.dist_thresh)

        geos = joined[["node_id"]].merge(nodes[["node_id", "geometry"]], on="node_id", how="left")[["geometry"]]
        connector_geo = joined[["stop_id", "geometry"]].shortest_line(geos).set_crs(self.links.crs)

        df = joined[["stop_id", "node_id"]].rename(columns={"node_id": "a_node", "stop_id": "b_node"})
        connectors = gpd.GeoDataFrame(df, geometry=connector_geo)

        min_speed = min(self.links.speed_ab.min(), self.links.speed_ba.min())
        connectors = connectors.assign(direction=0, link_id=np.arange(df.shape[0]) + 1 + self.links.link_id.max(),
                                       is_connector=1, speed_ab=min_speed, speed_ba=min_speed,
                                       distance=1.2 * (connectors.connector_geo.length ** 1.3))

        connectors = connectors.assign(time_ab=connectors.distance / connectors.speed_ab,
                                       time_ba=connectors.distance / connectors.speed_ba)

        net_data = pd.concat([broken_links, connectors], ignore_index=True)
        net_gdf = gpd.GeoDataFrame(net_data, geometry="geometry", crs=self.links.crs)
        self.__graph_from_broken_net(net_gdf)

    def __graph_from_broken_net(self, net_data):
        self.graph.network = net_data
        self.graph.prepare_graph(np.array(self.stops.stop_id.values))
        self.graph.set_graph("distance")
        self.graph.set_skimming(["distance", "time"])
        self.graph.set_blocked_centroid_flows(True)
