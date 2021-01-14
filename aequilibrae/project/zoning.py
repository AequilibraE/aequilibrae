from copy import deepcopy
from sqlite3 import Connection
from os.path import join, realpath
from warnings import warn
import shapely.wkb
from shapely.ops import unary_union
from shapely.geometry import Polygon
from aequilibrae.project.field_editor import FieldEditor
from aequilibrae.project.table_loader import TableLoader
from aequilibrae.project.project_creation import run_queries_from_sql_file
from .zone import Zone
from aequilibrae import logger
from aequilibrae.project.database_connection import database_connection


class Zoning:
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

        fields = zones.fields()

        # We can also add one more field to the table
        fields.add('parking_spots', 'Total licensed parking spots', 'INTEGER')

    """

    __items = {}

    def __init__(self, network):
        self.network = network
        self.__all_types = []
        self.conn = database_connection()
        self.__curr = self.conn.cursor()
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

    def extent(self) -> Polygon:
        """Queries the extent of the zoning system included in the model

        Returns:
            *model extent* (:obj:`Polygon`): Shapely polygon with the bounding box of the zoning system.
        """
        self.__curr.execute('Select ST_asBinary(GetLayerExtent("Links"))')
        poly = shapely.wkb.loads(self.__curr.fetchone()[0])
        return poly

    def coverage(self) -> Polygon:
        """ Returns a single polygon for the entire zoning coverage

        Returns:
            *model coverage* (:obj:`Polygon`): Shapely (Multi)polygon of the zoning system.
        """
        self.__curr.execute('Select ST_asBinary("geometry") from zones;')
        polygons = [shapely.wkb.loads(x[0]) for x in self.__curr.fetchall()]
        return unary_union(polygons)

    def get(self, zone_id: str) -> Zone:
        """Get a zone from the model by its **zone_id**"""
        if zone_id not in self.__items:
            raise ValueError(f"Zone {zone_id} does not exist in the model")
        return self.__items[zone_id]

    def fields(self) -> FieldEditor:
        """Returns a FieldEditor class instance to edit the zones table fields and their metadata"""
        return FieldEditor("zones")

    def all_zones(self) -> dict:
        """Returns a dictionary with all Zone objects available in the model. zone_id as key"""
        return self.__items

    def save(self):
        for zn in self.__items.values():  # type: Zone
            zn.save()

    def __copy__(self):
        raise Exception("Zones object cannot be copied")

    def __deepcopy__(self, memodict=None):
        raise Exception("Zones object cannot be copied")

    def __has_zoning(self):
        curr = self.conn.cursor()
        curr.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return any(["zone" in x[0].lower() for x in curr.fetchall()])

    def __load(self):
        tl = TableLoader()
        zones_list = tl.load_table(self.__curr, "zones")
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
