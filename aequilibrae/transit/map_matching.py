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

    fstop = int(route_stops.index.iat[0])

    res = graph.compute_path(fstop, int(route_stops.index.iat[-1]))
    res1 = graph.compute_path(fstop, int(route_stops.index.iat[-1]))

    if len(connected_stops) == 2:
        if res.milepost is None:
            return pd.DataFrame([])
        pdist = list(res.milepost[1:-1] - res.milepost[:-2])[1:]
        plnks = list(res.path[1:-1])
        pdirecs = list(res.path_link_directions[1:-1])
        return build_path_df(graph, pdirecs, pdist, plnks)

    access_links = graph.network[graph.network.a_node.isin(route_stops.index)]
    path_links, path_directions, path_distances = [], [], []
    for i in range(1, route_stops.shape[0]):
        current_stop = int(route_stops.index[i])
        is_not_last = i < route_stops.shape[0] - 1

        # Get the next stop for look-ahead path estimation (if not at the end)
        next_stop = int(route_stops.index[i + 1]) if is_not_last else None
        logger.debug(f"Computing path from node {current_start} to stop {current_stop}")

        # Compute path from current position to this stop
        res.compute_path(current_start, current_stop, early_exit=True)

        # Find the best network node to exit this stop from
        # We evaluate all connector links from this stop and choose the one
        # that minimizes total cost (current path + estimated cost to next stop)
        connection_candidates = access_links[access_links.a_node == current_stop].b_node.values

        min_cost = np.inf
        best_access_node = -1

        for candidate_node in connection_candidates:
            candidate_node = int(candidate_node)

            # Skip if candidate is where we came from
            if candidate_node == current_start:
                continue

            # Estimate cost: current path cost + look-ahead cost to next stop
            follow_cost = 0
            if is_not_last and next_stop is not None:
                res1.compute_path(candidate_node, next_stop, early_exit=True)
                if res1.milepost is None:
                    # Can't reach next stop from this candidate, skip it
                    continue
                follow_cost = res1.milepost[-1]

            # Get cost from origin to this candidate node
            # The skim contains cost to reach each node from the origin
            path_cost_to_candidate = res.skims[candidate_node, 0]
            total_estimate = follow_cost + path_cost_to_candidate

            if total_estimate < min_cost:
                min_cost = total_estimate
                best_access_node = candidate_node

        # If we found a valid exit node, extract the path
        if best_access_node >= 0:
            # Update the trace to end at the best access node
            res.update_trace(best_access_node)

            # Determine how many links to include
            # Skip connectors at start and (if not last stop) at end
            shift = 1 if is_not_last else 0

            if len(res.path) <= 1 + shift:
                # Path consists only of connector links, skip to next iteration
                continue

            # Extract and accumulate path components
            if is_not_last:
                # Exclude the final connector link (will be included in next iteration)
                path_links.extend(list(res.path[:-1]))
                path_directions.extend(list(res.path_link_directions[:-1]))
                # Calculate link distances from cumulative mileposts
                path_distances.extend(list(res.milepost[1:] - res.milepost[:-1])[:-1])
            else:
                # Last segment: include all links
                path_links.extend(list(res.path[:]))
                path_directions.extend(list(res.path_link_directions[:]))
                path_distances.extend(list(res.milepost[1:] - res.milepost[:-1])[:])

            # Update start position for next iteration
            # Use the second-to-last node in the path (before the access node)
            if len(res.path_nodes) > 3:
                current_start = res.path_nodes[-2]
            # Otherwise keep current_start as is

        else:
            # No valid path found through any access node
            logger.debug(f"Failed path computation during map-matching at stop index {i}")
            return pd.DataFrame(columns=["link_id", "dir"])

    return build_path_df(graph, path_directions, path_distances, path_links)


def build_path_df(graph, path_directions, path_distances, path_links) -> pd.DataFrame:
    """Builds a cleaned DataFrame of link IDs and directions from raw path data.

    Performs post-processing to:
    - Map internal link IDs to original network IDs
    - Remove consecutive duplicate links
    - Filter out short back-and-forth movements

    Args:
        graph: AequilibraE graph with network correspondence
        path_directions: List of link traversal directions
        path_distances: List of distances for each link
        path_links: List of internal link IDs

    Returns:
        DataFrame with columns ['link_id', 'dir']
    """
    corresp = pd.DataFrame(graph.network[["link_id", "original_id"]])

    if not path_links:
        return pd.DataFrame({"link_id": [], "dir": []})

    # Build initial result, skipping the first (connector) link
    result = pd.DataFrame(
        {
            "link_id": path_links[1:],
            "direction": path_directions[1:],
            "sequence": np.arange(len(path_links) - 1),
            "distance": path_distances[1:] if len(path_distances) > 1 else [],
        }
    )

    # Map internal link IDs to original network IDs
    df = result.merge(corresp, on="link_id", how="left")
    df.sort_values(by=["sequence"], inplace=True)

    # Remove consecutive links with same original_id and direction
    # (these are internal graph subdivisions of the same physical link)
    df = df[(df.original_id.shift(-1) != df.original_id) | (df.direction.shift(-1) != df.direction)]

    # Filter out isolated short segments (likely GPS noise or dead-end detours)
    # Keep a link if it differs from both neighbors OR if it's long enough
    crit_differs_prev = df.original_id.shift(1) != df.original_id
    crit_differs_next = df.original_id.shift(-1) != df.original_id
    df = df[(crit_differs_prev & crit_differs_next) | (df.distance > DEAD_END_RUN)]

    # Prepare final output with direction based on link sign
    df = df[["original_id", "direction"]].copy()
    df.columns = ["link_id", "dir"]
    df.loc[df.link_id > 0, "dir"] = 1
    df.loc[df.link_id < 0, "dir"] = -1
    df.reset_index(drop=True, inplace=True)

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


def graph_discount(route_shape: LineString, geolinks: gpd.GeoDataFrame) -> list:
    """
    Finds network links within a buffer of the route shape.

    These links are candidates for cost discounting during map-matching.

    Args:
        route_shape: LineString geometry of the route
        geolinks: GeoDataFrame of network links with 'link_id' column

    Returns:
        List of link_ids that intersect the buffered route shape
    """
    # Create a 20-meter buffer around the route shape
    buff = gpd.GeoSeries(route_shape, crs=geolinks.crs).buffer(20)
    gdf = gpd.GeoDataFrame(geometry=buff)

    # Find all links that intersect this buffer
    return geolinks.sjoin(gdf, how="inner", predicate="intersects").link_id.tolist()