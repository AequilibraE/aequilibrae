import logging
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import Point
from shapely.ops import substring


def split_links_at_stops(
    stops: gpd.GeoDataFrame, links: gpd.GeoDataFrame, tolerance: float = 50
) -> List[gpd.GeoDataFrame]:
    """
    Breaks links at the closest points for the nodes (transit stops) nearby, within a certain
    tolerance. New nodes are created at split points with IDs continuing from the maximum of the
    existing `a_node` and `b_node` values.

    Args:
        stops: GeoDataFrame of points
        links: GeoDataFrame of linestrings. Must contain "a_node" and "b_node" columns.
        tolerance: Search radius.

    Returns:
        List[GeoDataFrame]: A list containing two GeoDataFrames:
            - broken_links: The provided links where some have been split at stop locations.
              Contains updated "a_node", "b_node", and "geometry". Original link attributes are
              preserved (duplicated for splits).
            - new_nodes: Point geometries representing the new nodes created at split locations,
              with IDs starting from `start_node_id`.
    """

    start_node_id = max(links.a_node.max(), links.b_node.max(), stops.stop_id.max()) + 1

    # --- Step 0: Make sure we are operating in metres  ---
    for df in [stops, links]:
        assert df.crs.axis_info[0].unit_name.lower() in ["metre", "meter"], (
            "Both GeoDataFrames must be in a CRS with metre/meter units."
        )

    # --- Step 1: Matching Points to Lines (Same as before) ---
    # Buffer and intersection
    buffered_points = stops.copy()
    buffered_points["orig_geometry"] = buffered_points.geometry
    buffered_points = buffered_points.set_geometry(buffered_points.geometry.buffer(tolerance))

    matches_within = gpd.sjoin(buffered_points, links, how="inner", predicate="intersects")
    matches_within = (
        matches_within.set_geometry("orig_geometry")
        .drop(columns=["geometry"])
        .rename(columns={"orig_geometry": "geometry"})
    )

    # Handle orphans
    matched_indices = matches_within.index.unique()
    orphans_mask = ~stops.index.isin(matched_indices)

    df_result = matches_within[["geometry", "index_right"]].copy()

    if orphans_mask.any():
        orphans = stops.loc[orphans_mask]
        nearest_orphans = gpd.sjoin_nearest(orphans, links, how="left", distance_col="dist_to_line")
        nearest_orphans = nearest_orphans[~nearest_orphans.index.duplicated(keep="first")]
        df_result = pd.concat([df_result, nearest_orphans[["geometry", "index_right"]]])

    # Merge line info
    # We need geometry and node IDs
    df_result = df_result.merge(
        links[["geometry", "a_node", "b_node"]], left_on="index_right", right_index=True, suffixes=("", "_line")
    )

    # Calculate distance along line (linear reference)
    line_geoms = df_result["geometry_line"].values
    point_geoms = df_result["geometry"].values

    # Use vectorized project if available, otherwise apply
    dist_along = shapely.line_locate_point(line_geoms, point_geoms)

    df_result["dist_along"] = dist_along
    df_result["line_length"] = df_result.geometry_line.length

    # --- Step 2: Prepare Splits ---

    # Filter out splits that are effectively at the start or end of the link
    # 1 metre is effectively "right there"
    epsilon = 1
    valid_splits = df_result[
        (df_result["dist_along"] > epsilon) & (df_result["dist_along"] < (df_result["line_length"] - epsilon))
    ].copy()

    if valid_splits.empty:
        logging.debug("No valid splits found within tolerance? The map-matching will likely fail.")
        return [links.copy(), gpd.GeoDataFrame([], columns=["node_id", "geometry"], crs=links.crs)]

    # Identify unique split locations per link
    # A point maps to (link_id, distance).
    # Multiple points might map to the same location (approx).
    # We round distance to avoid segments shorter than 1 metre
    valid_splits["splt_lc"] = valid_splits["dist_along"].round(0)

    # Get unique split nodes needed: (link_id, rounded_distance)
    unique_splits = valid_splits[["index_right", "splt_lc"]].drop_duplicates().sort_values(["index_right", "splt_lc"])

    # Assign new Node IDs
    # We create a mapping: (link_id, splt_lc) -> new_node_id
    num_new_nodes = len(unique_splits)
    new_node_ids = np.arange(start_node_id, start_node_id + num_new_nodes)
    unique_splits["new_node_id"] = new_node_ids

    # --- Step 3: Perform Methodical Splitting ---

    # We need to process ONLY the links that have splits.
    links_to_split_indices = unique_splits["index_right"].unique()

    # Separate links that are untouched
    untouched_links = links.drop(links_to_split_indices)

    links_to_split = links.loc[links_to_split_indices]

    new_segments = []

    # Iterate over links that need splitting
    # (Grouping by link to handle multiple splits per link)
    grouped_splits = unique_splits.groupby("index_right")

    new_points = []

    for link_idx, group_df in grouped_splits:
        original_link = links_to_split.loc[link_idx]
        geo = original_link.geometry
        original_a = original_link["a_node"]
        original_b = original_link["b_node"]

        # Sort splits by distance
        splits = group_df.sort_values("splt_lc")

        # Define chain of nodes: [Original_A, New_1, New_2, ..., Original_B]
        nodes = [original_a] + splits["new_node_id"].tolist() + [original_b]

        # Define cut points: [0.0, d1, d2, ..., Length]
        cuts = [0.0] + splits["splt_lc"].tolist() + [geo.length]

        # Create segments
        for i in range(len(cuts) - 1):
            start_dist = cuts[i]
            end_dist = cuts[i + 1]

            # Extract geometry substring
            seg_geo = substring(geo, start_dist, end_dist)

            # Create a copy of the original row data
            new_row = original_link.copy()
            new_row["geometry"] = seg_geo
            new_row["a_node"] = nodes[i]
            new_row["b_node"] = nodes[i + 1]
            new_segments.append(new_row)

            # Record the new points, as we will need them to create the connectors
            # For the last segment, we don"t need to add a point, as we already have the original b_node
            if nodes[i + 1] != original_b:
                new_points.append([nodes[i + 1], Point(tuple(seg_geo.coords[-1]))])

    # Combine everything
    new_segments_gdf = gpd.GeoDataFrame(new_segments, crs=links.crs)

    broken_links = pd.concat([untouched_links, new_segments_gdf], ignore_index=True)

    broken_links = broken_links.assign(original_id=broken_links.link_id)
    broken_links = broken_links.assign(link_id=np.zeros(broken_links.shape[0]) + 1)
    broken_links["distance"] = broken_links.geometry.length

    new_points_gdf = gpd.GeoDataFrame(new_points, columns=["node_id", "geometry"], crs=links.crs)

    return [broken_links, new_points_gdf]
