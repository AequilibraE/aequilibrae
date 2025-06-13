from sqlite3 import Connection
from typing import Optional

from shapely.geometry import LineString, Polygon

from aequilibrae.utils.db_utils import commit_and_close

INFINITE_CAPACITY = 99999


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
