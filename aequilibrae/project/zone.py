from sqlite3 import Connection
import numpy as np
from scipy.cluster.vq import kmeans2, whiten
from scipy.spatial.distance import cdist
import shapely.wkb
from shapely.geometry import Point, MultiPolygon, LineString
from .network.safe_class import SafeClass
from aequilibrae.project.database_connection import database_connection
from aequilibrae.project.network.links import Links
from aequilibrae.project.network.nodes import Nodes


class Zone(SafeClass):
    """Single zone object that can be queried and manipulated in memory"""

    def __init__(self, dataset: dict, zoning):
        self.geometry = MultiPolygon()
        self.zone_id = -1
        super().__init__(dataset)
        self.__zoning = zoning
        self.conn = zoning.conn  # type: Connection
        self.__new = dataset['geometry'] is None
        self.__network_links = zoning.network.links
        self.__network_nodes = zoning.network.nodes

    def delete(self):
        """Removes the zone from the database"""
        conn = database_connection()
        curr = conn.cursor()
        curr.execute(f'DELETE FROM zones where zone_id="{self.zone_id}"')
        conn.commit()
        self.__zoning._remove_zone(self.zone_id)
        del self

    def save(self):
        """Saves/Updates the zone data to the database"""

        if self.zone_id != self.__original__['zone_id']:
            raise ValueError('One cannot change the zone_id')

        conn = database_connection()
        curr = conn.cursor()

        curr.execute(f'select count(*) from zones where zone_id="{self.zone_id}"')
        if curr.fetchone()[0] == 0:
            data = [self.zone_id, self.geometry.wkb]
            curr.execute('Insert into zones (zone_id, geometry) values(?, ST_Multi(GeomFromWKB(?, 4326)))', data)

        for key, value in self.__dict__.items():
            if key != 'zone_id' and key in self.__original__:
                v_old = self.__original__.get(key, None)
                if value != v_old and value is not None:
                    self.__original__[key] = value
                    if key == 'geometry':
                        sql = "update 'zones' set geometry=ST_Multi(GeomFromWKB(?, 4326)) where zone_id=?"
                        curr.execute(sql, [value.wkb, self.zone_id])
                    else:
                        curr.execute(f"update 'zones' set '{key}'=? where zone_id=?", [value, self.zone_id])
        conn.commit()
        conn.close()

    def add_centroid(self, point: Point, mode_id: str, link_types='', connectors=1) -> None:
        """Adds a centroid and centroid connectors for the desired mode to the network file

           Centroid connectors are created by clustering all nodes inside the zone that
           satisfy the mode and link_types criteria in as many clusters as requested connectors

               Args:
                   *point* (:obj:`Point`): Shapely Point corresponding to the

                   *mode_id* (:obj:`str`): Mode ID we are trying to connect

                   *link_types* (:obj:`str`): String with all the link types that can be considered

                   *connectors* (:obj:`int`): Number of connectors to add
               """
        curr = self.conn.cursor()

        curr.execute('select count(*) from nodes where node_id=?', [self.zone_id])
        if curr.fetchone()[0] > 0:
            raise Exception('There is already a centroid for this zone')

        if not self.geometry.contains(point):
            Warning('Centroid is not inside the zone')

        if len(mode_id) > 1:
            raise Exception('We can only add centroid connectors for one mode at a time')

        if len(link_types) > 0:
            lt = link_types
        else:
            curr.execute('Select link_type_id from link_types')
            lt = ''.join([x[0] for x in curr.fetchall()])
            lt = f'*[{lt}]*'

        sql = '''select node_id, ST_asBinary(geometry), modes, link_types from nodes where ST_Within(geometry, GeomFromWKB(?, ?)) and
                        (nodes.rowid in (select rowid from SpatialIndex where f_table_name = 'nodes' and
                        search_frame = GeomFromWKB(?, ?)))
                and link_types glob ? and instr(modes, ?)>0'''

        wkb = self.geometry.wkb
        curr.execute(sql, [wkb, self._srid, wkb, self._srid, lt, mode_id])

        coords = []
        nodes = []
        for node_id, wkb, modes, link_types in curr.fetchall():
            geo = shapely.wkb.loads(wkb)
            coords.append([geo.x, geo.y])
            nodes.append(node_id)

        if len(nodes) == 0:
            raise Exception('We could not find any candidate nodes that satisfied your criteria')
        elif len(nodes) < connectors:
            Warning('We have fewer candidate nodes than required connectors. Will create as many as possible')
            num_connectors = len(nodes)
        else:
            num_connectors = connectors

        features = np.array(coords)
        whitened = whiten(features)
        centroids, allocation = kmeans2(whitened, num_connectors)

        for i in range(num_connectors):
            nds = [x for x, y in zip(nodes, list(allocation)) if y == i]
            centr = centroids[i]
            positions = [x for x, y in zip(whitened, allocation) if y == i]
            dist = cdist(np.array([centr]), np.array(positions)).flatten()
            node_to_connect = nds[dist.argmin()]

            link = self.__network_links.new()

            node = self.__network_nodes.get(node_to_connect)
            link.geometry = LineString([point, node.geometry])
            link.modes = mode_id
            link.direction = 0
            link.link_type = 'centroid_connector'
            link.name = f'centroid connector zone {self.zone_id} for mode {mode_id}'
            link.capacity_ab = 99999
            link.capacity_ba = 99999
            link.save()

        # We need to re-number the centroid just added in order to make it a centroid
        link = self.__network_links.get(link.link_id)
        node = self.__network_nodes.get(link.a_node)
        node.is_centroid = 1
        node.save()
        node.renumber(self.zone_id)
