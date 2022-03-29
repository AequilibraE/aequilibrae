from copy import deepcopy
from os.path import join, realpath
from warnings import warn

import shapely.wkb
from shapely.geometry import Polygon
from shapely.ops import unary_union

from aequilibrae import logger
from aequilibrae.project.project_creation import run_queries_from_sql_file
from aequilibrae.project.table_loader import TableLoader
from .basic_table import BasicTable
from .zone import Zone


class Zoning(BasicTable):
    """
    Access to the API resources to manipulate the zones table in the project

    ::

        from aequilibrae import Project

        p = Project()
        p.open('path/to/project/folder')

        zones = p.zoning

        # We edit the fields for a particular zone
        zone_downtown = zones.get(1)
        zone_downtown.population = 637
        zone_downtown.employment = 10039
        zone_downtown.save()

        fields = zones.fields

        # We can also add one more field to the table
        fields.add('parking_spots', 'Total licensed parking spots', 'INTEGER')

    """

    __items = {}

    def __init__(self, network):
        super().__init__()
        self.network = network
        self.__table_type__ = 'zones'
        self.__fields = []
        if self.__has_zoning():
            self.__load()

    def new(self, zone_id: int) -> Zone:
        """Creates a new zone

        Returns:
            *zone* (:obj:`Zone`): A new zone object populated only with zone_id (but not saved in the model yet)
            """

        if zone_id in self.__items:
            raise Exception(f"Zone ID {zone_id} already exists")

        data = {key: None for key in self.__fields}
        data["zone_id"] = zone_id

        logger.info(f"Zone with id {zone_id} was created")
        return self.__create_return_zone(data)

    def create_zoning_layer(self):
        """Creates the 'zones' table for project files that did not previously contain it"""

        if not self.__has_zoning():
            qry_file = join(realpath(__file__), "database_specification", "tables", "zones.sql")
            run_queries_from_sql_file(self.conn, qry_file)
            self.__load()
        else:
            warn("zones table already exists. Nothing was done", Warning)

    def coverage(self) -> Polygon:
        """ Returns a single polygon for the entire zoning coverage

        Returns:
            *model coverage* (:obj:`Polygon`): Shapely (Multi)polygon of the zoning system.
        """
        self._curr.execute('Select ST_asBinary("geometry") from zones;')
        polygons = [shapely.wkb.loads(x[0]) for x in self._curr.fetchall()]
        return unary_union(polygons)

    def get(self, zone_id: str) -> Zone:
        """Get a zone from the model by its **zone_id**"""
        if zone_id not in self.__items:
            raise ValueError(f"Zone {zone_id} does not exist in the model")
        return self.__items[zone_id]

    def all_zones(self) -> dict:
        """Returns a dictionary with all Zone objects available in the model. zone_id as key"""
        return self.__items

    def save(self):
        for item in self.__items.values():
            item.save()

    def __has_zoning(self):
        curr = self.conn.cursor()
        curr.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return any(["zone" in x[0].lower() for x in curr.fetchall()])

    def __load(self):
        tl = TableLoader()
        zones_list = tl.load_table(self._curr, "zones")
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

    def _remove_zone(self, zone_id: int):
        del self.__items[zone_id]

    def __create_return_zone(self, data):
        zone = Zone(data, self)
        self.__items[zone.zone_id] = zone
        return zone
