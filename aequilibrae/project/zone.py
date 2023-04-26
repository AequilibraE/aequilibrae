import random
from sqlite3 import Connection
from shapely.geometry import Point, MultiPolygon
from .network.safe_class import SafeClass
from .network.connector_creation import connector_creation


class Zone(SafeClass):
    """Single zone object that can be queried and manipulated in memory"""

    def __init__(self, dataset: dict, zoning):
        self.geometry = MultiPolygon()
        self.zone_id = -1
        super().__init__(dataset, zoning.project)
        self.__zoning = zoning
        self.conn = zoning.conn  # type: Connection
        self.__new = dataset["geometry"] is None
        self.__network_links = zoning.network.links
        self.__network_nodes = zoning.network.nodes

    def delete(self):
        """Removes the zone from the database"""
        conn = self._project.connect()
        curr = conn.cursor()
        curr.execute(f'DELETE FROM zones where zone_id="{self.zone_id}"')
        conn.commit()
        conn.close()
        self.__zoning._remove_zone(self.zone_id)
        del self

    def save(self):
        """Saves/Updates the zone data to the database"""

        if self.zone_id != self.__original__["zone_id"]:
            raise ValueError("One cannot change the zone_id")

        conn = self._project.connect()
        curr = conn.cursor()

        curr.execute(f'select count(*) from zones where zone_id="{self.zone_id}"')
        if curr.fetchone()[0] == 0:
            data = [self.zone_id, self.geometry.wkb]
            curr.execute("Insert into zones (zone_id, geometry) values(?, ST_Multi(GeomFromWKB(?, 4326)))", data)

        for key, value in self.__dict__.items():
            if key != "zone_id" and key in self.__original__:
                v_old = self.__original__.get(key, None)
                if value != v_old and value is not None:
                    self.__original__[key] = value
                    if key == "geometry":
                        sql = "update 'zones' set geometry=ST_Multi(GeomFromWKB(?, 4326)) where zone_id=?"
                        curr.execute(sql, [value.wkb, self.zone_id])
                    else:
                        curr.execute(f"update 'zones' set '{key}'=? where zone_id=?", [value, self.zone_id])
        conn.commit()
        conn.close()

    def add_centroid(self, point: Point, robust=True) -> None:
        """Adds a centroid to the network file

        :Arguments:
            **point** (:obj:`Point`): Shapely Point corresponding to the desired centroid position.
            If None, uses the geometric center of the zone
            **robust** (:obj:`Bool`, Optional): Moves the centroid location around to avoid node conflict.
            Defaults to True.
        """

        # This is VERY small in real-world terms (between zero and 11cm)
        shift = 0.000001

        curr = self.conn.cursor()

        curr.execute("select count(*) from nodes where node_id=?", [self.zone_id])
        if curr.fetchone()[0] > 0:
            self._project.logger.warning("Centroid already exists. Failed to create it")
            return

        sql = "INSERT into nodes (node_id, is_centroid, geometry) VALUES(?,1,GeomFromWKB(?, ?));"

        if point is None:
            point = self.geometry.centroid

        if robust:
            check_sql = """SELECT count(*) FROM nodes
                             WHERE  nodes.geometry = GeomFromWKB(?, 4326) AND
                          nodes.ROWID IN (
                           SELECT ROWID FROM SpatialIndex WHERE f_table_name = 'nodes' AND
                           search_frame = GeomFromWKB(?, 4326))
                       """

            test_list = self.conn.execute(check_sql, [point.wkb, point.wkb]).fetchone()
            while sum(test_list):
                test_list = self.conn.execute(check_sql, [point.wkb, point.wkb]).fetchone()
                point = Point(point.x + random.random() * shift, point.y + random.random() * shift)

        data = [self.zone_id, point.wkb, self.__srid__]
        self.conn.execute(sql, data)
        self.conn.commit()

    def connect_mode(self, mode_id: str, link_types="", connectors=1) -> None:
        """Adds centroid connectors for the desired mode to the network file

        Centroid connectors are created by connecting the zone centroid to one or more nodes selected from
        all those that satisfy the mode and link_types criteria and are inside the zone.

        The selection of the nodes that will be connected is done simply by computing running the
        `KMeans2 <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans2.html>`_
        clustering algorithm from SciPy and selecting the nodes closest to each cluster centroid.

        When there are no node candidates inside the zone, the search area is progressively expanded until
        at least one candidate is found.

        If fewer candidates than required connectors are found, all candidates are connected.

        :Arguments:
            **mode_id** (:obj:`str`): Mode ID we are trying to connect

            **link_types** (:obj:`str`, `Optional`): String with all the link type IDs that can be considered.
            eg: yCdR. Defaults to ALL link types

            **connectors** (:obj:`int`, `Optional`): Number of connectors to add. Defaults to 1
        """
        connector_creation(
            self.geometry,
            zone_id=self.zone_id,
            srid=self.__srid__,
            mode_id=mode_id,
            link_types=link_types,
            connectors=connectors,
            network=self._project.network,
        )

    def disconnect_mode(self, mode_id: str) -> None:
        """Removes centroid connectors for the desired mode from the network file

        :Arguments:
            **mode_id** (:obj:`str`): Mode ID we are trying to disconnect from this zone
        """

        curr = self.conn.cursor()
        data = [self.zone_id, mode_id]
        curr.execute("Delete from links where a_node=? and modes=?", data)
        row_count = curr.rowcount

        data = [mode_id, self.zone_id, mode_id]
        curr.execute('Update links set modes = replace(modes, ?, "") where a_node=? and instr(modes,?) > 0', data)
        row_count += curr.rowcount

        if row_count:
            self._project.logger.warning(f"Deleted {row_count} connectors for mode {mode_id} for zone {self.zone_id}")
        else:
            self._project.warning("No centroid connectors for this mode")
        self.conn.commit()
