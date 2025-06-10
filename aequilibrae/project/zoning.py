from copy import deepcopy
from os.path import join, realpath
from typing import Union, Dict
import warnings

import geopandas as gpd
import pandas as pd
import shapely.wkb
from shapely.geometry import Point, Polygon, LineString, MultiLineString
from shapely import union_all

from aequilibrae.project.basic_table import BasicTable
from aequilibrae.project.data_loader import DataLoader
from aequilibrae.project.network.connector_creation import connector_creation
from aequilibrae.project.project_creation import run_queries_from_sql_file
from aequilibrae.project.table_loader import TableLoader
from aequilibrae.project.zone import Zone
from aequilibrae.utils.aeq_signal import SIGNAL, simple_progress
from aequilibrae.utils.db_utils import commit_and_close
from aequilibrae.utils.geo_index import GeoIndex


class Zoning(BasicTable):
    """
    Access to the API resources to manipulate the 'zones' table in the project

    .. code-block:: python

        >>> project = create_example(project_path, "coquimbo")

        >>> zoning = project.zoning

        >>> zone_downtown = zoning.get(1)
        >>> zone_downtown.population = 637
        >>> zone_downtown.employment = 10039
        >>> zone_downtown.save()

        # We can also add one more field to the table
        >>> fields = zoning.fields
        >>> fields.add('parking_spots', 'Total licensed parking spots', 'INTEGER')
    """

    def __init__(self, network):
        super().__init__(network.project)
        self.__items: Dict[int, Zone] = {}
        self.network = network
        self.__table_type__ = "zones"
        self.__fields = []
        self.__geo_index = GeoIndex()
        if self.__has_zoning():
            self.__load()

    def new(self, zone_id: int) -> Zone:
        """Creates a new zone

        :Returns:
            **zone** (:obj:`Zone`): A new zone object populated only with ``zone_id`` (but not saved in the model yet)
        """

        if zone_id in self.__items:
            raise Exception(f"Zone ID {zone_id} already exists")

        data = {key: None for key in self.__fields}
        data["zone_id"] = zone_id

        self.project.logger.info(f"Zone with id {zone_id} was created")
        return self.__create_return_zone(data)

    def create_zoning_layer(self):
        """Creates the 'zones' table for project files that did not previously contain it"""

        if not self.__has_zoning():
            qry_file = join(realpath(__file__), "database_specification", "tables", "zones.sql")
            with self.network.project.db_connection as conn:
                run_queries_from_sql_file(conn, self.project.logger, qry_file)
            self.__load()
        else:
            self.project.warning("zones table already exists. Nothing was done", Warning)

    def coverage(self) -> Polygon:
        """Returns a single polygon for the entire zoning coverage

        :Returns:
            **model coverage** (:obj:`Polygon`): Shapely (Multi)polygon of the zoning system.
        """
        with self.network.project.db_connection as conn:
            dt = conn.execute('Select ST_asBinary("geometry") from zones;').fetchall()
        polygons = [shapely.wkb.loads(x[0]) for x in dt]
        return union_all(polygons)

    def get(self, zone_id: str) -> Zone:
        """Get a zone from the model by its ``zone_id``"""
        if zone_id not in self.__items:
            raise ValueError(f"Zone {zone_id} does not exist in the model")
        return self.__items[zone_id]

    def all_zones(self) -> dict:
        """Returns a dictionary with all Zone objects available in the model, using ``zone_id`` as key"""
        return self.__items

    def save(self):
        for item in self.__items.values():
            item.save()

    def add_centroids(self, robust=True):
        """Adds automatic centroids to the network file. It adds centroids to all zones that do not have one
        Centroid is added to the geographic centroid of the zone.

        :Arguments:
            **robust** (:obj:`bool`, *Optional*): Moves the centroid location around to avoid node conflict.
            Defaults to ``True``.
        """
        i = 0
        with commit_and_close(self.project.path_to_file, spatial=True) as conn:
            existing_centroids = pd.read_sql("SELECT node_id from Nodes where is_centroid = 1", conn).node_id.to_numpy()
            for zone_id in simple_progress(self.__items.keys(), SIGNAL(object), "Connecting zones"):
                if zone_id in existing_centroids:
                    continue
                zone = self.__items[zone_id]
                zone.add_centroid(zone.geometry.centroid, robust)
                i += 1
        if i > 0:
            self.project.logger.info(f"{i} new centroids added to the network")
        else:
            self.project.logger.info("No new centroids added to the network")

    def connect_mode(self, mode_id: str, link_types="", connectors=1, limit_to_zone=True):
        """Adds centroid connectors for the desired mode to the network file

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
        """

        proj_nodes = self.project.network.nodes.data
        links = self.project.network.links

        centroids = proj_nodes.reset_index().query("is_centroid == 1", engine="python").node_id.to_numpy()
        link_data = links.data
        centroid_conn = link_data.query("a_node in @centroids and modes.str.contains(@mode_id)", engine="python")
        connected_centroids = centroid_conn.a_node.to_numpy()

        with (
            commit_and_close(self.project.path_to_file, spatial=True) as conn,
            warnings.catch_warnings(),
        ):
            warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")
            zones_todo = [x for x in self.__items.keys() if x not in connected_centroids]
            for zone_id in simple_progress(zones_todo, SIGNAL(object), "Connecting zones"):
                if zone_id not in centroids:
                    self.project.logger.warning(f"Centroid for zone {zone_id} does not exist. Please create it first.")
                    continue

                zone = self.__items[zone_id]
                area = zone.geometry if limit_to_zone else None
                connector_creation(
                    zone_id=zone_id,
                    mode_id=mode_id,
                    link_types=link_types,
                    connectors=connectors,
                    proj_nodes=proj_nodes,
                    proj_links=link_data,
                    network=self.project.network,
                    conn=conn,
                    delimiting_area=area,
                )

    def get_closest_zone(self, geometry: Union[Point, LineString, MultiLineString]) -> int:
        """Returns the zone in which the given geometry is located.

        If the geometry is not fully enclosed by any zone, the zone closest to
        the geometry is returned

        :Arguments:
            **geometry** (:obj:`Point` or :obj:`LineString`): A Shapely geometry object

        :Returns:
            **zone_id** (:obj:`int`): ID of the zone applicable to the point provided
        """

        nearest = self.__geo_index.nearest(geometry, 10)
        dists = {}
        for zone_id in nearest:
            geo = self.__items[zone_id].geometry
            if geo.contains(geometry):
                return zone_id
            dists[geo.distance(geometry)] = zone_id
        return dists[min(dists.keys())]

    def refresh_geo_index(self):
        self.__geo_index.reset()
        for zone_id, zone in self.__items.items():
            self.__geo_index.insert(feature_id=zone_id, geometry=zone.geometry)

    def __has_zoning(self):
        with self.network.project.db_connection as conn:
            dt = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        return any("zone" in x[0].lower() for x in dt)

    def __load(self):
        tl = TableLoader()
        with self.network.project.db_connection as conn:
            zones_list = tl.load_table(conn, "zones")
        self.__fields = deepcopy(tl.fields)

        existing_list = [zn["zone_id"] for zn in zones_list]
        if zones_list:
            self.__properties = list(zones_list[0].keys())
        for zn in zones_list:
            if zn["zone_id"] not in self.__items:
                self.__items[zn["zone_id"]] = Zone(zn, self)

        to_del = [key for key in self.__items.keys() if key not in existing_list]
        for key in to_del:
            del self.__items[key]
        self.refresh_geo_index()

    def _remove_zone(self, zone_id: int):
        del self.__items[zone_id]

    def __create_return_zone(self, data):
        zone = Zone(data, self)
        self.__items[zone.zone_id] = zone
        return zone

    @property
    def data(self) -> gpd.GeoDataFrame:
        """Returns all zones data as a Pandas DataFrame

        :Returns:
            **table** (:obj:`GeoDataFrame`): GeoPandas GeoDataFrame with all the nodes
        """
        dl = DataLoader(self.project.path_to_file, "zones")
        return dl.load_table()
