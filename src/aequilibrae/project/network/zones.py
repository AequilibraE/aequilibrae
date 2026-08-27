import logging
import random
import warnings
from pathlib import Path
from typing import Union

import pandas as pd
import shapely.wkb
from shapely import union_all
from shapely.geometry import LineString, MultiLineString, Point, Polygon

from aequilibrae.project.network.connector_creation import bulk_connector_creation, connector_creation
from aequilibrae.project.network.links import Links
from aequilibrae.project.network.nodes import Nodes
from aequilibrae.project.project_creation import run_queries_from_sql_file
from aequilibrae.project.project_table import SpatialProjectTable
from aequilibrae.utils.aeq_signal import SIGNAL, simple_progress
from aequilibrae.utils.db_utils import has_table
from aequilibrae.utils.geo_index import GeoIndex

logger = logging.getLogger(__name__)


class Zones(SpatialProjectTable):
    """
    Access to the API resources to manipulate the 'zones' table in the project

    .. code-block:: python

        >>> project = create_example(project_path)

        >>> zones = project.network.zones

        >>> zone_downtown = zones.get(1)
        >>> zones.update(1, population=637, employment=10039)

        # We can also add one more field to the table
        >>> fields = zones.fields
        >>> fields.add('parking_spots', 'Total licensed parking spots', 'INTEGER')

        >>> project.close()
    """

    name = "zones"
    key = "zone_id"
    record_name = "ZoneRecord"
    multi_part = True
    __geo_index: GeoIndex | None = None
    has_numeric_key = True

    def create_zones_table(self) -> None:
        """Creates the 'zones' table for project files that did not previously contain it"""

        if not self.has_zones:
            qry_file = Path(__file__).parents[1] / "database_specification" / "network" / "tables" / "zones.sql"
            with self._connection.transaction() as conn:
                run_queries_from_sql_file(conn, qry_file)
        else:
            logger.warning("zones table already exists. Nothing was done")

    @property
    def has_zones(self) -> bool:
        """Whether the project has a 'zones' table"""
        return has_table(self._connection._connection, self.name)

    def coverage(self) -> Polygon:
        """Returns a single polygon for the entire zoning coverage

        :Returns:
            **model coverage** (:obj:`Polygon`): Shapely (Multi)polygon of the zoning system.
        """
        dt = self._connection._connection.execute('SELECT ST_AsBinary("geometry") FROM zones').fetchall()
        polygons = [shapely.wkb.loads(x[0]) for x in dt]
        return union_all(polygons)

    def add_centroid(self, zone_id: int, point: Point | None = None, robust: bool = True) -> None:
        """Adds a centroid to the network file for the given zone

        :Arguments:
            **zone_id** (:obj:`int`): ID of the zone the centroid belongs to

            **point** (:obj:`Point`, *Optional*): Shapely Point corresponding to the desired centroid position.
            If None, uses the geometric center of the zone

            **robust** (:obj:`Bool`, *Optional*): Moves the centroid location around to avoid node conflict.
            Defaults to ``True``.
        """

        # This is VERY small in real-world terms (between zero and 11cm)
        shift = 0.000001

        with self._connection.transaction() as conn:
            if conn.execute("SELECT count(*) FROM nodes WHERE node_id=?", [zone_id]).fetchone()[0] > 0:
                logger.warning("Centroid already exists. Failed to create it")
                return

            if point is None:
                point = self.get(zone_id).geometry.centroid

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

            sql = "INSERT INTO nodes (node_id, is_centroid, geometry) VALUES(?,1,GeomFromWKB(?, ?));"
            conn.execute(sql, [zone_id, point.wkb, self.srid])

    def add_centroids(self, robust: bool = True) -> None:
        """Adds automatic centroids to the network file. It adds centroids to all zones that do not have one
        Centroid is added to the geographic centroid of the zone.

        :Arguments:
            **robust** (:obj:`bool`, *Optional*): Moves the centroid location around to avoid node conflict.
            Defaults to ``True``.
        """
        i = 0
        existing_centroids = pd.read_sql(
            "SELECT node_id FROM Nodes WHERE is_centroid = 1", self._connection._connection
        ).node_id.to_numpy()
        for zone in simple_progress(list(self), SIGNAL(object), "Adding centroids"):
            if zone.zone_id in existing_centroids:
                continue
            self.add_centroid(zone.zone_id, zone.geometry.centroid, robust)
            i += 1
        if i > 0:
            logger.info(f"{i} new centroids added to the network")
        else:
            logger.info("No new centroids added to the network")

    def connect_mode(
        self,
        mode_id: str,
        link_types: str = "",
        connectors: int = 1,
        limit_to_zone: bool = True,
        bulk: bool = False,
    ) -> None:
        """
        Adds centroid connectors for the desired mode to the network file

        Centroid connectors are created by connecting each zone centroid to one or more nodes selected from
        all those that satisfy the mode and link_types criteria and are inside the zone.

        The selection of the nodes that will be connected is done simply by searching for the node closest to each
        zone centroid, or the N closest nodes to the centroid.

        If fewer candidates than required connectors are found, all candidates are connected.

        CENTROIDS THAT ARE CURRENTLY CONNECTED ARE SKIPPED ALTOGETHER

        :Arguments:
            **mode_id** (:obj:`str`): Mode ID we are trying to connect

            **link_types** (:obj:`str`, *Optional*): String with all the link type IDs that can be considered.
                eg: yCdR. Defaults to ALL link types

            **connectors** (:obj:`int`, *Optional*): Number of connectors to add. Defaults to 1

            **limit_to_zone** (:obj:`bool`): Limits the search for nodes inside the zone. Defaults to ``True``.

            **bulk** (:obj:`bool`, *Optional*): Whether to use the bulk connector method or not. This is method is
                considerably faster for connecting a large amount of centroids but has a high runtime overhead.
        """

        nodes = Nodes(self._connection)
        links = Links(self._connection)
        proj_nodes = nodes.data
        link_data = links.data

        centroids = proj_nodes.query("is_centroid == 1", engine="python").node_id.to_numpy()
        centroid_conn = link_data.query("a_node in @centroids and modes.str.contains(@mode_id)", engine="python")
        connected_centroids = centroid_conn.a_node.to_numpy()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")
            conn = self._connection

            if not bulk:
                zones_todo = [zone for zone in self if zone.zone_id not in connected_centroids]
                for zone in simple_progress(zones_todo, SIGNAL(object), "Connecting zones"):
                    if zone.zone_id not in centroids:
                        logger.warning(f"Centroid for zone {zone.zone_id} does not exist. Please create it first.")
                        continue

                    connector_creation(
                        zone_id=zone.zone_id,
                        mode_id=mode_id,
                        link_types=link_types,
                        connectors=connectors,
                        proj_nodes=proj_nodes,
                        proj_links=link_data,
                        project_connection=self._connection,
                        links=links,
                        delimiting_area=zone.geometry if limit_to_zone else None,
                    )
            else:
                if len(link_types) > 0:
                    nodes = proj_nodes[proj_nodes.link_types.str.contains("|".join(list(link_types)))]
                else:
                    nodes = proj_nodes

                zones = self.data
                zones = zones[~zones.zone_id.isin(connected_centroids)]

                if zones.empty:
                    return

                bulk_connector_creation(
                    conn,
                    nodes,
                    link_data,
                    zones,
                    modes=[mode_id],
                    k_connectors=connectors,
                    projected_crs=None,
                    limit_to_zone=limit_to_zone,
                )
        self._invalidate()

    def disconnect_mode(self, mode_id: str, zone_id: int | None = None) -> None:
        """Removes centroid connectors for the desired mode from the network file

        :Arguments:
            **mode_id** (:obj:`str`): Mode ID we are trying to disconnect

            **zone_id** (:obj:`int`, *Optional*): Zone to disconnect. Disconnects all zones if not provided
        """

        with self._connection.transaction() as conn:
            if zone_id is None:
                zone_filter, data = "a_node IN (SELECT zone_id FROM zones)", []
            else:
                zone_filter, data = "a_node=?", [zone_id]

            row_count = conn.execute(f"DELETE FROM links WHERE {zone_filter} AND modes=?", [*data, mode_id]).rowcount

            sql = f'UPDATE links SET modes = replace(modes, ?, "") WHERE {zone_filter} AND instr(modes,?) > 0'
            row_count += conn.execute(sql, [mode_id, *data, mode_id]).rowcount

            if row_count:
                logger.warning(f"Deleted {row_count} connectors for mode {mode_id}")
            else:
                logger.warning("No centroid connectors for this mode")

    def get_closest_zone(self, geometry: Union[Point, LineString, MultiLineString]) -> int:
        """Returns the zone in which the given geometry is located.

        If the geometry is not fully enclosed by any zone, the zone closest to
        the geometry is returned

        :Arguments:
            **geometry** (:obj:`Point` or :obj:`LineString`): A Shapely geometry object

        :Returns:
            **zone_id** (:obj:`int`): ID of the zone applicable to the point provided
        """

        if self.__geo_index is None:
            self.__geometries = {zone.zone_id: zone.geometry for zone in self}
            self.__geo_index = GeoIndex()
            for zone_id, geo in self.__geometries.items():
                self.__geo_index.insert(feature_id=zone_id, geometry=geo)

        dists = {}
        for zone_id in self.__geo_index.nearest(geometry, 10):
            geo = self.__geometries[zone_id]
            if geo.contains(geometry):
                return zone_id
            dists[geo.distance(geometry)] = zone_id
        return dists[min(dists.keys())]

    def _invalidate(self) -> None:
        self.__geo_index = None
