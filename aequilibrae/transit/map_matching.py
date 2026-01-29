import geopandas as gpd
import pandas as pd
from aequilibrae.paths import Graph
from aequilibrae.paths import PathResults
from shapely import LineString


def map_match_route(graph: Graph, route_stops: gpd.GeoDataFrame, route_shape: LineString, check_connectivity=True):
    # It assumes that both the graph, stops AND route shape are in the same CRS

    if route_stops.shape[0] <= 1:
        return pd.DataFrame([])

    # If the route shape is not defined, we build it from the stops as a sequence of line segments
    if route_shape is None:
        route_shape = LineString(route_stops.geometry.tolist())

    if check_connectivity:
        # We check if all the stops are connected:
        centroids = graph.centroids
        graph.prepare_graph(centroids=route_stops.index.to_numpy())
        skims = graph.compute_skims()
        if skims.results.skims.distance.max() >= 1.0e308:
            self.__logger.critical(f"Route is not completely connected.")
            return pd.DataFrame([])
        graph.prepare_graph(centroids=centroids)

    # We discount the likely links for this route to favor them in the map-matching
    graph.cost = np.array(graph.graph[graph.cost_field])
    likely_links = graph_discount(route_shape, graph.network)
    graph.cost[(graph.graph.link_id.isin(likely_links)) & (graph.graph.is_connector == 0)] *= 0.1

    fstop = route_stops.index.iat[0]

    if len(connected_stops) == 2:
        res = graph.compute_path(fstop, route_stops.index.iat[1])
        if res.milepost is None:
            return pd.DataFrame([])
        pdist = list(res.milepost[1:-1] - res.milepost[:-2])[1:]
        plnks = list(res.path[1:-1])
        pdirecs = list(res.path_link_directions[1:-1])
        return build_path_df(graph, pdirecs, pdist, plnks)

    path_links, path_directions, path_distances = [], [], []
    res.compute_path()
    for stop_id, stop in route_stops.loc[1:, :].iterrows():
        end = tstop.__map_matching_id__[self.route_type]

        not_last = idx + 2 <= len(connected_stops) - 1

        if not_last:
            following_stop = connected_stops[idx + 2]
            n_end = following_stop.__map_matching_id__[self.route_type]
        self.__logger.debug(f"Computing paths between {start} and {end}")
        res.compute_path(fstop, int(end), early_exit=True)
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
            return pd.DataFrame([])
        start = res.path_nodes[-2] if len(res.path_nodes) > 3 else start

    return build_path_df(graph, path_directions, path_distances, path_links)


def build_path_df(graph, path_directions, path_distances, path_links):
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
            if df.loc[i: i + 2, "link_id"].abs().unique().shape[0] == 1:
                df.drop(index=[i, i + 1], inplace=True)
                df.reset_index(drop=True, inplace=True)
                has_issues = True
                break
    return df


def __assemble__mm_shape(df: pd.DataFrame):
    shape = []  # type: List[Tuple[float, float]]

    for _, rec in df.iterrows():
        line_geo = self.__geolinks.loc[self.__geolinks.link_id == abs(rec.link_id)].geometry.values[0]
        coords = list(line_geo.coords)[::-1] if rec.link_id < 0 else list(line_geo.coords)
        data = coords[1:] if shape else coords
        shape.extend(data)
    self.shape = LineString(shape)


def graph_discount(route_shape: LineString, geolinks: gpd.GeoDataFrame) -> list:
    buff = gpd.GeoSeries(route_shape, crs=geolinks.crs).buffer(20).geometry
    gdf = gpd.GeoDataFrame(geometry=buff)
    return geolinks.sjoin(gdf, how='inner', predicate="intersects").link_id.tolist()
