from math import pi, sqrt
import numpy as np
from scipy.cluster.vq import kmeans2, whiten
from scipy.spatial.distance import cdist
import shapely.wkb
from shapely.geometry import LineString

INFINITE_CAPACITY = 99999


def connector_creation(geo, zone_id: int, srid: int, mode_id: str, network, link_types="", connectors=1):
    if len(mode_id) > 1:
        raise Exception("We can only add centroid connectors for one mode at a time")

    conn = network.project.connect()
    curr = conn.cursor()
    logger = network.project.logger
    curr.execute("select count(*) from nodes where node_id=?", [zone_id])
    if curr.fetchone() is None:
        logger.warning("This centroid does not exist. Please create it first")
        return

    proj_nodes = network.nodes
    node = proj_nodes.get(zone_id)
    curr.execute("select count(*) from links where a_node=? and instr(modes,?) > 0", [zone_id, mode_id])
    if curr.fetchone()[0] > 0:
        logger.warning("Mode is already connected")
        return

    if len(link_types) > 0:
        lt = f"*[{link_types}]*"
    else:
        curr.execute("Select link_type_id from link_types")
        lt = "".join([x[0] for x in curr.fetchall()])
        lt = f"*[{lt}]*"

    sql = """select node_id, ST_asBinary(geometry), modes, link_types from nodes where ST_Within(geometry, GeomFromWKB(?, ?)) and
                    (nodes.rowid in (select rowid from SpatialIndex where f_table_name = 'nodes' and
                    search_frame = GeomFromWKB(?, ?)))
            and link_types glob ? and instr(modes, ?)>0"""

    # We expand the area by its average radius until it is 20 times
    # beginning with a strict search within the zone
    buffer = 0
    increase = sqrt(geo.area / pi)
    dt = []
    while dt == [] and buffer <= increase * 10:
        wkb = geo.buffer(buffer).wkb
        curr.execute(sql, [wkb, srid, wkb, srid, lt, mode_id])
        dt = curr.fetchall()
        buffer += increase

    if buffer > increase:
        msg = f"Could not find node inside zone {zone_id}. Search area was expanded until we found a suitable node"
        logger.warning(msg)
    if dt == []:
        logger.warning(
            f"FAILED! Could not find suitable nodes to connect within 5 times the diameter of zone {zone_id}."
        )
        return

    coords = []
    nodes = []
    for node_id, wkb, modes, link_types in dt:
        geo = shapely.wkb.loads(wkb)
        coords.append([geo.x, geo.y])
        nodes.append(node_id)

    num_connectors = connectors
    if len(nodes) == 0:
        raise Exception("We could not find any candidate nodes that satisfied your criteria")
    elif len(nodes) < connectors:
        logger.warning(
            f"We have fewer possible nodes than required connectors for zone {zone_id}. Will connect all of them."
        )
        num_connectors = len(nodes)

    if num_connectors == len(coords):
        all_nodes = nodes
    else:
        features = np.array(coords)
        whitened = whiten(features)
        centroids, allocation = kmeans2(whitened, num_connectors)

        all_nodes = set()
        for i in range(num_connectors):
            nds = [x for x, y in zip(nodes, list(allocation)) if y == i]
            centr = centroids[i]
            positions = [x for x, y in zip(whitened, allocation) if y == i]
            if positions:
                dist = cdist(np.array([centr]), np.array(positions)).flatten()
                node_to_connect = nds[dist.argmin()]
                all_nodes.add(node_to_connect)

    nds = list(all_nodes)
    data = [zone_id] + nds
    curr.execute(f'select b_node from links where a_node=? and b_node in ({",".join(["?"] * len(nds))})', data)

    data = [x[0] for x in curr.fetchall()]

    if data:
        qry = ",".join(["?"] * len(data))
        dt = [mode_id, zone_id] + data
        curr.execute(f"Update links set modes=modes || ? where a_node=? and b_node in ({qry})", dt)
        nds = [x for x in nds if x not in data]
        logger.warning(f"Mode {mode_id} added to {len(data)} existing centroid connectors for zone {zone_id}")
        conn.commit()

    curr.close()
    links = network.links

    for node_to_connect in nds:
        link = links.new()
        node_to = proj_nodes.get(node_to_connect)
        link.geometry = LineString([node.geometry, node_to.geometry])
        link.modes = mode_id
        link.direction = 0
        link.link_type = "centroid_connector"
        link.name = f"centroid connector zone {zone_id}"
        link.capacity_ab = INFINITE_CAPACITY
        link.capacity_ba = INFINITE_CAPACITY
        link.save()
    if nds:
        logger.warning(f"{len(nds)} new centroid connectors for mode {mode_id} added for centroid {zone_id}")

    conn.commit()
