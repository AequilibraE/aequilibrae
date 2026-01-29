import logging
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import LineString

from aequilibrae.paths import Graph

DEAD_END_RUN = 40


class RouteMapMatcher:
    def __init__(self, graph: Graph, reverse_graph: Optional[Graph] = None, check_connectivity=True):
        self.graph = graph
        self.crs = graph.network.crs
        self.reverse_graph = reverse_graph or graph.reverse()
        self.check_connectivity = check_connectivity

    def map_match_route(self, route_stops: gpd.GeoDataFrame, route_shape: Optional[LineString] = None):
        path_directions, path_links = self._build_full_path_on_broken_links(route_stops, route_shape)
        return self._build_path_df(path_directions, path_links)

    def _build_full_path_on_broken_links(self, route_stops: gpd.GeoDataFrame, route_shape: Optional[LineString] = None):
        # It assumes that both the graph, stops AND route shape are in the same CRS

        if route_stops.shape[0] <= 1:
            return [], []

        # If the route shape is not defined, we build it from the stops as a sequence of line segments
        if route_shape is None:
            route_shape = LineString(route_stops.geometry.tolist())

        if self.check_connectivity:
            # We check if all the stops are connected:
            centroids = self.graph.centroids
            self.graph.prepare_graph(centroids=route_stops.index.to_numpy())
            skims = self.graph.compute_skims()
            self.graph.prepare_graph(centroids=centroids)
            if skims.results.skims.distance.max() >= 1.0e308:
                self.__logger.critical("Route is not completely connected.")
                return [], []

        # We discount the likely links for this route to favor them in the map-matching
        self.graph.cost = np.array(self.graph.graph[self.graph.cost_field])
        likely_links = self.__graph_discount(route_shape)
        for g in [self.graph, self.reverse_graph]:
            g.cost[(g.graph.link_id.isin(likely_links)) & (g.graph.is_connector == 0)] *= 0.1

        current_stop = int(route_stops.index.iat[0])

        res = self.graph.compute_path(current_stop, int(route_stops.index.iat[-1]))

        if route_stops.shape[0] == 2:
            if res.milepost is None:
                return [], []
            plnks = list(res.path[1:-1])
            pdirecs = list(res.path_link_directions[1:-1])
            return plnks, pdirecs

        access_links = self.graph.network[self.graph.network.a_node.isin(route_stops.index)]
        path_links, path_directions = [], []

        is_first = True
        for i in range(1, route_stops.shape[0]):
            next_stop = int(route_stops.index[i])
            is_not_last = i < route_stops.shape[0] - 1

            # Get the next stop for look-ahead path estimation (if not at the end)
            following_stop = int(route_stops.index[i + 1]) if is_not_last else None
            logging.debug(f"Computing path from node {current_stop} to stop {next_stop}")

            connection_candidates = access_links[access_links.a_node == next_stop].b_node.values

            idx_ = 1 if is_first else 0
            is_first = False

            # If this is the last stop, then we just compute the path directly
            if not is_not_last:
                res.compute_path(current_stop, next_stop, early_exit=True)
                path_links.extend(list(res.path[idx_:-1]))
                path_directions.extend(list(res.path_link_directions[idx_:-1]))
                continue

            # Let's see if we can reach the following stop while going through the next one
            # Whis would save us some path computation
            res.compute_path(current_stop, following_stop, early_exit=True)
            indices_in_a = np.where(np.isin(connection_candidates, res.path_nodes))[0]
            if indices_in_a.shape[0] > 0:
                # We found a candidate that is already in the path to the following stop
                best_access_node = connection_candidates[indices_in_a[-1]]
            else:
                # Find the best network node to exit this stop from
                # We evaluate all connector links from this stop and choose the one
                # that minimizes total cost (current path + estimated cost to next stop)
                # That is MUCH easier done with the reserve graph, though

                res.update_trace(next_stop)
                indices = np.where(np.isin(res.path_nodes, connection_candidates))[0]
                first_leg_costs = res.milepost[indices]

                res_reverse = self.reverse_graph.compute_path(following_stop, next_stop)
                indices = np.where(np.isin(res_reverse.path_nodes, connection_candidates))[0]
                second_leg_costs = res_reverse.milepost[indices]

                best_idx = np.argmin(first_leg_costs + second_leg_costs)
                best_access_node = connection_candidates[best_idx]

            # Update the trace to end at the best access node
            res.update_trace(best_access_node)

            # Determine how many links to include
            # Skip connectors at start and (if not last stop) at end
            path_links.extend(list(res.path[idx_:]))
            path_directions.extend(list(res.path_link_directions[idx_:]))

            current_stop = best_access_node

        return path_directions, path_links

    def _build_path_df(self, path_directions, path_links) -> pd.DataFrame:
        """Builds a cleaned DataFrame of link IDs and directions from raw path data.

        Performs post-processing to:
        - Filter out short back-and-forth movements

        Args:
            graph: AequilibraE graph with network correspondence
            path_directions: List of link traversal directions
            path_links: List of internal link IDs

        Returns:
            DataFrame with columns ['link_id', 'dir']
        """

        if not path_links:
            return pd.DataFrame({"link_id": [], "dir": []})

        corresp = pd.DataFrame(self.graph.network[["link_id", "original_id", "distance"]])

        # Build initial result, skipping the first (connector) link
        result = pd.DataFrame(
            {
                "link_id": path_links[1:],
                "direction": path_directions[1:],
                "sequence": np.arange(len(path_links) - 1),
            }
        )

        # Map internal link IDs to original network IDs
        df = result.merge(corresp, on="link_id", how="left")
        df.sort_values(by=["sequence"], inplace=True)

        # Remove consecutive links with same original_id and direction
        # (these are internal graph subdivisions of the same physical link)
        df = df[(df.original_id.shift(-1) != df.original_id) | (df.direction.shift(-1) != df.direction)]

        # Filter out isolated short segments (likely noise or dead-end detours)
        # Keep a link if it differs from both neighbors OR if it's long enough
        crit_differs_prev = df.original_id.shift(1) != df.original_id
        crit_differs_next = df.original_id.shift(-1) != df.original_id
        df = df[(crit_differs_prev & crit_differs_next) | (df.distance > DEAD_END_RUN)]

        # Prepare final output with direction based on link sign
        df.sort_values(by=["sequence"], inplace=True)
        df = df[["original_id", "direction"]]

        # Eliminate back-and-forth patterns on the same link
        # e.g., [A, B, A] -> [A] when all three have the same absolute link_id
        has_issues = True
        while has_issues:
            has_issues = False
            for i in range(0, df.shape[0] - 2):
                # Check if three consecutive rows are all the same link
                if df.loc[i: i + 2, "link_id"].abs().unique().shape[0] == 1:
                    df.drop(index=[i, i + 1], inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    has_issues = True
                    break
        return df

    def __graph_discount(self, route_shape: LineString) -> list:
        """
        Finds network links within a buffer of the route shape.

        These links are candidates for cost discounting during map-matching.

        Args:
            route_shape: LineString geometry of the route
            geolinks: GeoDataFrame of network links with 'link_id' column

        Returns:
            List of link_ids that intersect the buffered route shape
        """
        geolinks = self.graph.network[["link_id", "geometry"]]

        # Create a 20-meter buffer around the route shape
        buff = gpd.GeoSeries(route_shape, crs=geolinks.crs).buffer(20)
        gdf = gpd.GeoDataFrame(geometry=buff)

        # Find all links that intersect this buffer
        return gdf.sjoin(geolinks, how="inner", predicate="contains").link_id.tolist()

    def __assemble__mm_shape(self, df: pd.DataFrame):
        shape = []  # type: List[Tuple[float, float]]

        for _, rec in df.iterrows():
            line_geo = self.__geolinks.loc[self.__geolinks.link_id == abs(rec.link_id)].geometry.values[0]
            coords = list(line_geo.coords)[::-1] if rec.link_id < 0 else list(line_geo.coords)
            data = coords[1:] if shape else coords
            shape.extend(data)
        self.shape = LineString(shape)
