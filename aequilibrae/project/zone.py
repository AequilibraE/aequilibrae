import random
from sqlite3 import Connection
from typing import Optional

from shapely.geometry import Point, MultiPolygon

from .network.connector_creation import connector_creation
from .network.safe_class import SafeClass


class Zone(SafeClass):
    """Single zone object that can be queried and manipulated in memory"""

    def __init__(self, dataset: dict, zoning):
        self.geometry = MultiPolygon()
        self.zone_id = -1
        super().__init__(dataset, zoning.project)
        self.__zoning = zoning
        self.__new = dataset["geometry"] is None
        self.__network_links = zoning.network.links
        self.__network_nodes = zoning.network.nodes

    def delete(self):
        """Removes the zone from the database"""
        with self.project.db_connection as conn:
            conn.execute(f'DELETE FROM zones where zone_id="{self.zone_id}"')
        self.__zoning._remove_zone(self.zone_id)
        del self

    def save(self):
        """Saves/Updates the zone data to the database"""

        if self.zone_id != self.__original__["zone_id"]:
            raise ValueError("One cannot change the zone_id")

        with self.project.db_connection as conn:
            if conn.execute(f'select count(*) from zones where zone_id="{self.zone_id}"').fetchone()[0] == 0:
                data = [self.zone_id, self.geometry.wkb]
                conn.execute("Insert into zones (zone_id, geometry) values(?, ST_Multi(GeomFromWKB(?, 4326)))", data)

            for key, value in self.__dict__.items():
                if key != "zone_id" and key in self.__original__:
                    v_old = self.__original__.get(key, None)
                    if value != v_old and value is not None:
                        self.__original__[key] = value
                        if key == "geometry":
                            sql = "update 'zones' set geometry=ST_Multi(GeomFromWKB(?, 4326)) where zone_id=?"
                            conn.execute(sql, [value.wkb, self.zone_id])
                        else:
                            conn.execute(f"update 'zones' set '{key}'=? where zone_id=?", [value, self.zone_id])

    def add_centroid(self, point: Point, robust=True) -> None:
        """Adds a centroid to the network file

        :Arguments:
            **point** (:obj:`Point`): Shapely Point corresponding to the desired centroid position.
            If None, uses the geometric center of the zone

            **robust** (:obj:`Bool`, *Optional*): Moves the centroid location around to avoid node conflict.
            Defaults to ``True``.
        """

        # This is VERY small in real-world terms (between zero and 11cm)
        shift = 0.000001

        with self.project.db_connection as conn:
            if conn.execute("select count(*) from nodes where node_id=?", [self.zone_id]).fetchone()[0] > 0:
                self.project.logger.warning("Centroid already exists. Failed to create it")
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

                test_list = conn.execute(check_sql, [point.wkb, point.wkb]).fetchone()
                while sum(test_list):
                    test_list = conn.execute(check_sql, [point.wkb, point.wkb]).fetchone()
                    point = Point(point.x + random.random() * shift, point.y + random.random() * shift)

            data = [self.zone_id, point.wkb, self.__srid__]
            conn.execute(sql, data)

    def connect_mode(
        self, mode_id: str, link_types="", connectors=1, conn: Optional[Connection] = None, limit_to_zone=True
    ) -> None:
        """Adds centroid connectors for the desired mode to the network file

        Centroid connectors are created by connecting the zone centroid to one or more nodes selected from
        all those that satisfy the mode and link_types criteria and are inside the zone.

        The selection of the nodes that will be connected is done simply by searching for the node closest to the
        zone centroid, or the N closest nodes to the centroid.

        If fewer candidates than required connectors are found, all candidates are connected.

        :Arguments:
            **mode_id** (:obj:`str`): Mode ID we are trying to connect

            **link_types** (:obj:`str`, *Optional*): String with all the link type IDs that can be considered.
            eg: yCdR. Defaults to ALL link types

            **connectors** (:obj:`int`, *Optional*): Number of connectors to add. Defaults to 1

            **conn** (:obj:`sqlite3.Connection`, *Optional*): Connection to the database.

            **limit_to_zone** (:obj:`bool`): Limits the search for nodes inside the zone. Defaults to ``True``.
        """

        area = self.geometry if limit_to_zone else None
        connector_creation(
            zone_id=self.zone_id,
            mode_id=mode_id,
            link_types=link_types,
            connectors=connectors,
            proj_nodes=self.project.network.nodes.data,
            proj_links=self.project.network.links.data,
            network=self.project.network,
            conn=conn,
            delimiting_area=area,
        )

    def disconnect_mode(self, mode_id: str) -> None:
        """Removes centroid connectors for the desired mode from the network file

        :Arguments:
            **mode_id** (:obj:`str`): Mode ID we are trying to disconnect from this zone
        """

        with self.project.db_connection as conn:
            data = [self.zone_id, mode_id]
            row_count = conn.execute("Delete from links where a_node=? and modes=?", data).rowcount

            data = [mode_id, self.zone_id, mode_id]
            sql = 'Update links set modes = replace(modes, ?, "") where a_node=? and instr(modes,?) > 0'
            row_count += conn.execute(sql, data).rowcount

            if row_count:
                self.project.logger.warning(
                    f"Deleted {row_count} connectors for mode {mode_id} for zone {self.zone_id}"
                )
            else:
                self.project.warning("No centroid connectors for this mode")
