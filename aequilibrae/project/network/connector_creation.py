import logging
import time
from sqlite3 import Connection
from typing import Optional, Union

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree
from shapely.geometry import LineString, Polygon

from aequilibrae.utils.db_utils import commit_and_close

INFINITE_CAPACITY = 99999

logger = logging.getLogger(__name__)


def connector_creation(
    zone_id: int,
    mode_id: str,
    network,
    proj_nodes,
    proj_links,
    link_types="",
    connectors=1,
    conn: Optional[Connection] = None,
    delimiting_area: Polygon = None,
):
    if len(mode_id) > 1:
        raise Exception("We can only add centroid connectors for one mode at a time")

    with conn or network.project.db_connection as connec:
        logger = network.project.logger
        if sum(connec.execute("select count(*) from nodes where node_id=?", [zone_id]).fetchone()) == 0:
            logger.warning("This centroid does not exist. Please create it first")
            return

        sql = "select count(*) from links where a_node=? and instr(modes,?) > 0"
        if connec.execute(sql, [zone_id, mode_id]).fetchone()[0] > 0:
            logger.warning("Mode is already connected")
            return

    centroid = proj_nodes.query("node_id == @zone_id")  # type: gpd.GeoDataFrame
    centroid = centroid.rename(columns={"node_id": "zone_id"})[["zone_id", "geometry"]]

    nodes = proj_nodes.query("is_centroid != 1 and modes.str.contains(@mode_id)", engine="python")

    if len(link_types) > 0:
        nodes = nodes[nodes.link_types.str.contains("|".join(list(link_types)))]

    if delimiting_area is not None:
        nodes = nodes[nodes.geometry.within(delimiting_area)]

    if nodes.empty:
        zone_id = centroid["zone_id"].values[0]
        logger.warning(f"No nodes found for centroid {zone_id} (mode {mode_id} and link types {link_types})")
        return

    joined = nodes[["node_id", "geometry"]].sjoin_nearest(centroid, distance_col="distance_connector")
    joined = joined.nsmallest(connectors, "distance_connector")

    # Check if link with a/b nodes exists to avoid unnecessary repetition
    centr_geo = centroid.geometry.values[0]
    links = network.links
    query = """(a_node==@zone_id | b_node==@zone_id) & (a_node==@rec.node_id | b_node==@rec.node_id) & link_type=='centroid_connector'"""
    for _, rec in joined.iterrows():
        link_exist = proj_links.query(query)
        if link_exist.empty:
            link = links.new()
            link.geometry = LineString([centr_geo, rec.geometry])
            link.modes = mode_id
            link.direction = 0
            link.link_type = "centroid_connector"
            link.name = f"centroid connector zone {zone_id}"
            link.capacity_ab = INFINITE_CAPACITY
            link.capacity_ba = INFINITE_CAPACITY
            link.save(conn)
        else:
            link = links.get(link_exist.link_id.values[0])
            link.add_mode(mode_id)
            link.save(conn)

    if not joined.empty:
        logger.info(f"{joined.shape[0]} new centroid connectors for mode {mode_id} added for centroid {zone_id}")


def bulk_connector_creation(
    conn: Connection,
    project_nodes: gpd.GeoDataFrame,
    project_links: gpd.GeoDataFrame,
    project_zones: gpd.GeoDataFrame,
    modes: list[str],
    k_connectors: int = 1,
    limit_to_zone: bool = True,
    distance_upper_bound: float = float("inf"),
    projected_crs: Union[str, int, None] = None,
):
    """
    Creates or updates centroid connectors between zone centroids and network nodes.

    This function generates k-nearest neighbour connections from each zone centroid to nearby
    network nodes that support the specified transport modes. It can either limit connections
    to nodes within the same zone or find the globally nearest nodes.

    :Arguments:
        **conn** (:obj:`Connection`): Database connection for executing SQL operations.

        **project_nodes** (:obj:`gpd.GeoDataFrame`): GeoDataFrame containing network nodes
        with columns including node_id, is_centroid, modes, and geometry.

        **project_links** (:obj:`gpd.GeoDataFrame`): GeoDataFrame containing network links
        with columns including link_id, a_node, b_node, modes, direction, and link_type.

        **project_zones** (:obj:`gpd.GeoDataFrame`): GeoDataFrame containing zone polygons
        with columns including zone_id and geometry.

        **modes** (:obj:`list[str]`): List of transport mode strings to create connectors for.

        **k_connectors** (:obj:`int`, `Optional`): Number of nearest neighbour connections
        to create per centroid. Defaults to 1.

        **limit_to_zone** (:obj:`bool`, `Optional`): If True, only connects to nodes within
        the same zone. If False, finds globally nearest nodes. Defaults to True.

        **distance_upper_bound** (:obj:`float`, `Optional`): Maximum distance for connections.
        Defaults to infinity.

        **projected_crs** (:obj:`str | int`, `Optional`): Coordinate reference system for
        distance calculations. If None, uses the CRS from the input data.
    """
    assert project_links.crs == project_nodes.crs == project_zones.crs, "Mismatched CRS"
    assert modes, "Modes must be provided"

    centroid_mask = project_nodes.is_centroid == 1
    centroids = project_nodes.loc[centroid_mask, ["node_id", "geometry"]]
    nodes = project_nodes.loc[~centroid_mask, ["node_id", "modes", "geometry"]]

    assert (
        project_zones["zone_id"].isin(centroids["node_id"]).all()
    ), "All provided zones must have their corresponding centroid provided"

    connectors = []
    for mode in modes:
        if limit_to_zone:
            df = k_nearest_in_zone(
                k_connectors,
                project_zones,
                centroids,
                nodes[nodes.modes.str.contains(mode)],
                distance_upper_bound=distance_upper_bound,
                crs=projected_crs if projected_crs is not None else centroids.crs,
            )
        else:
            df = k_nearest(
                k_connectors,
                centroids,
                nodes[nodes.modes.str.contains(mode)],
                distance_upper_bound=distance_upper_bound,
                crs=projected_crs if projected_crs is not None else centroids.crs,
            )

        connectors.append(df.assign(modes=mode))

    # We now need to combine the modes of the all node pairs, but we only care about the set of modes. Sorting is to
    # provide some sort of standard ordering.
    connectors = pd.concat(connectors).groupby(["a_node", "b_node"]).modes.apply(normalise_mode_strings).reset_index()
    if connectors.empty:
        raise ValueError(
            f"No connects found for any modes ({modes}), ensure the modes are correct and the distance "
            "bound matches the units of the CRS"
        )

    # We need to find out which connectors already exist so we can update the links instead of inserting them.
    centroid_connectors = project_links[project_links.link_type == "centroid_connector"]
    centroid_connectors["modes"] = centroid_connectors["modes"].apply(normalise_mode_strings)
    existing_connectors = centroid_connectors.merge(
        connectors.assign(connector_index=connectors.index), on=["a_node", "b_node"], how="inner"
    )[["link_id", "direction", "modes_x", "modes_y", "connector_index"]]

    # We'll drop any of the existing connectors from our new connectors dataframe.
    connectors = connectors.drop(existing_connectors["connector_index"])

    if (uni_directional_connectors := (existing_connectors["direction"] != 0)).any():
        logger.warning(
            f"Found {uni_directional_connectors.sum()} non-bidirectional existing centroid connectors that overlap "
            "with connectors that would be have created. These connectors will not be updated. link_ids: "
            f"{existing_connectors.loc[uni_directional_connectors, 'link_id'].to_list()}"
        )
        existing_connectors = existing_connectors[~uni_directional_connectors]

    # Find the connectors we need to update, we then find the union of the mode strings. These are the links we will
    # update with the new modes. We will not change anything else about them.
    existing_connectors = existing_connectors[existing_connectors["modes_x"] != existing_connectors["modes_y"]]
    existing_connectors = existing_connectors[["link_id"]].assign(
        modes=existing_connectors[["modes_x", "modes_y"]].sum(axis=1).apply(normalise_mode_strings)
    )

    if existing_connectors.empty and connectors.empty:
        logger.info("No new connectors to create nor any to update")
        return

    logger.info(f"Creating {len(connectors)} new connectors")
    logger.info(f"Updating {len(existing_connectors)} existing connectors' modes")

    # We now need to form the geometry of our connectors
    connectors = (
        connectors.merge(centroids, left_on="a_node", right_on="node_id", how="left")
        .drop(columns="node_id")
        .merge(nodes[["node_id", "geometry"]], left_on="b_node", right_on="node_id", how="left")
        .drop(columns="node_id")
    )
    connectors["geometry"] = connectors.apply(
        lambda row: LineString((row.geometry_x, row.geometry_y)), axis=1, result_type="reduce"
    )
    connectors = gpd.GeoDataFrame(
        connectors.drop(columns=["geometry_x", "geometry_y"]), geometry="geometry", crs=project_links.crs
    )

    # Links need new link_ids so just use the max + 1 as a starting point
    max_link_id = project_links.link_id.max()
    connectors["link_id"] = np.arange(max_link_id + 1, max_link_id + 1 + len(connectors))

    existing_connectors_sql = "UPDATE links SET modes=? WHERE link_id=?"
    new_connectors_sql = f"""
    INSERT INTO links
    (link_id, a_node, b_node, modes, direction, link_type, capacity_ab, capacity_ba, name, geometry)
    VALUES(?,?,?,?,0,"centroid_connector",{INFINITE_CAPACITY},{INFINITE_CAPACITY},'centroid connector zone ' || ?2,GeomFromWKB(?, 4326))
    """
    with conn:
        conn.executemany(existing_connectors_sql, existing_connectors[["modes", "link_id"]].to_records(index=False))
        conn.executemany(
            new_connectors_sql,
            connectors[["link_id", "a_node", "b_node", "modes", "geometry"]]
            .to_crs(4326)
            .to_wkb()
            .to_records(index=False),
        )


def k_nearest(
    k: int,
    centroids: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    distance_upper_bound: float,
    crs: Union[int, str],
):
    """
    Finds the k nearest nodes to each centroid using a KDTree spatial index.

    :Arguments:
        **k** (:obj:`int`): Number of nearest neighbours to find for each centroid.

        **centroids** (:obj:`gpd.GeoDataFrame`): GeoDataFrame containing centroid points
        with node_id and geometry columns.

        **nodes** (:obj:`gpd.GeoDataFrame`): GeoDataFrame containing network nodes with
        node_id and geometry columns to search within.

        **distance_upper_bound** (:obj:`float`): Maximum distance for neighbour search.

        **crs** (:obj:`int | str`): Coordinate reference system for distance calculations.

    :Returns:
        **pd.DataFrame**: DataFrame with columns a_node (centroid), b_node (nearest node),
        and distance, sorted by a_node and distance.
    """
    kdTree = KDTree(nodes.to_crs(crs).get_coordinates().to_numpy())

    distance, index = kdTree.query(
        centroids.to_crs(crs).get_coordinates().to_numpy(), k=k, distance_upper_bound=distance_upper_bound
    )

    # Add a new axis to make the slicing consistent
    if k == 1:
        distance = distance[:, None]
        index = index[:, None]

    res = []
    for i in range(k):
        # If a centroid doesn't have k neighbours then the "index" is returned as kdTree.n (or len(nodes))
        ith_nearest_nodes_mask = index[:, i] != kdTree.n
        df = pd.DataFrame(
            data={
                "a_node": centroids["node_id"].to_numpy(),
                "b_node": nodes["node_id"].iloc[index[ith_nearest_nodes_mask, i]].to_numpy(),
                "distance": distance[ith_nearest_nodes_mask, i],
            }
        )
        res.append(df)

    return pd.concat(res).sort_values(by=["a_node", "distance"])


def k_nearest_in_zone(
    k: int,
    zones: gpd.GeoDataFrame,
    centroids: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    distance_upper_bound: float,
    crs: Union[int, str],
):
    """
    Finds the k nearest nodes within each zone to the corresponding centroid.

    Uses spatial indexing to first determine which nodes fall within each zone, then
    calculates distances only between centroids and nodes in the same zone.

    :Arguments:
        **k** (:obj:`int`): Number of nearest neighbours to find for each centroid.

        **zones** (:obj:`gpd.GeoDataFrame`): GeoDataFrame containing zone polygons with
        zone_id and geometry columns.

        **centroids** (:obj:`gpd.GeoDataFrame`): GeoDataFrame containing centroid points
        with node_id and geometry columns, corresponding to zones.

        **nodes** (:obj:`gpd.GeoDataFrame`): GeoDataFrame containing network nodes with
        node_id and geometry columns to search within.

        **distance_upper_bound** (:obj:`float`): Maximum distance for neighbour search.

        **crs** (:obj:`int | str`): Coordinate reference system for distance calculations.

    :Returns:
        **pd.DataFrame**: DataFrame with columns a_node (centroid), b_node (nearest node),
        and distance, limited to k nearest nodes per centroid within the same zone.
    """
    res = zones.sindex.query(nodes.geometry, predicate="within")

    # If the mapping doesn't include all zones and all nodes then we should make it known. While possible to have nodes
    # lie outside of zones, because we have the "within zone" requirement means we should ignore but make it known. The
    # more concerning case is when a zone doesn't have any nodes within it.
    if len(zones_idx_with_nodes := np.unique(res[1])) != len(zones):
        logger.warning(f"There are {len(zones) - len(zones_idx_with_nodes)} zones without nodes!")

    if len(nodes_idx_with_zones := np.unique(res[0])) != len(nodes):
        logger.warning(f"There are {len(nodes) - len(nodes_idx_with_zones)} nodes not within the supplied zones!")

    # We now explode the dataframes to form that centroid -> set of nodes mapping (indexing into centroids should be the
    # same as zones)
    centroids_exploded = centroids.iloc[res[1]]
    nodes_exploded = nodes.iloc[res[0]]

    # We then compute the distance from a zone to all of the nodes it contains. This operation doesn't suffer quadratic
    # blow up that the naive approach does, because (typically) the mapping from nodes to zones is injective (the
    # distance is to a zone for a node is only calculated once per node). It's possible that multiple zones overlap and
    # thus can share nodes, but it should be fine for small numbers of overlapping zones.
    df = pd.DataFrame(
        data={
            "a_node": centroids_exploded.node_id.to_numpy(),
            "b_node": nodes_exploded.node_id.to_numpy(),
            "distance": centroids_exploded.geometry.to_crs(crs).distance(
                nodes_exploded.geometry.to_crs(crs), align=False
            ),
        },
    )

    # Before we do this filter, just make sure it's not the default value of inf.
    if distance_upper_bound < float("inf"):
        df = df[df["distance"] < distance_upper_bound]

    # This was considerable faster than using a groupby -> nsmallest operation for some reason in my testing.
    df = df.sort_values(by=["a_node", "distance"])
    return df.groupby(by="a_node").head(k)


def normalise_mode_strings(x):
    """
    Normalises a collection of mode strings by sorting unique characters.

    Takes a sequence of mode strings and returns a single string containing
    unique characters sorted alphabetically.

    :Arguments:
        **x** (:obj:`Iterable[str]`): Collection of mode strings to normalise.

    :Returns:
        **str**: Normalised string with unique characters sorted alphabetically.
    """
    return "".join(sorted(set(x)))
