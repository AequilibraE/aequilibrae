from sqlite3 import IntegrityError, Connection
from os.path import join, dirname, realpath
from warnings import warn
from aequilibrae import logger
from aequilibrae.project.field_editor import FieldEditor
from aequilibrae.project.table_loader import TableLoader
from aequilibrae.project.project_creation import run_queries_from_sql_file
from .zone import Zone


class Zoning:
    """
    Access to the API resources to manipulate the zones table in the project

    ::

        from aequilibrae import Project

        p = Project()
        p.open('path/to/project/folder')

        zones = p.zones

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

    def __init__(self, project):
        self.__all_types = []
        self.__conn = project.conn  # type: Connection
        self.__curr = project.conn.cursor()
        if self.__has_zoning():
            self.__load()

    def create(self):
        """Creates the 'zones' table for project files that did not previously contain it"""

        if not self.__has_zoning():
            qry_file = join(realpath(__file__), 'database_specification', 'tables', 'zones.sql')
            run_queries_from_sql_file(self.__conn, qry_file)
            self.__load()
        else:
            warn('zones table already exists. Nothing was done', Warning)

    def get(self, zone_id: str) -> Zone:
        """Get a zone from the model by its **zone_id**"""
        if zone_id not in self.__items:
            raise ValueError(f'Zone {zone_id} does not exist in the model')
        return self.__items[zone_id]

    def fields(self) -> FieldEditor:
        """Returns a FieldEditor class instance to edit the zones table fields and their metadata"""
        return FieldEditor('zones')

    def all_zones(self) -> dict:
        """Returns a dictionary with all Zone objects available in the model. zone_id as key"""
        return self.__items

    def save(self):
        for zn in self.__items.values():  # type: Zone
            zn.save()

    def __copy__(self):
        raise Exception('Zones object cannot be copied')

    def __deepcopy__(self, memodict=None):
        raise Exception('Zones object cannot be copied')

    def __del__(self):
        self.__items.clear()

    def __has_zoning(self):
        curr = self.__conn.cursor()
        curr.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return any(['zone' in x[0].lower() for x in curr.fetchall()])

    def __load(self):
        tl = TableLoader()
        zones_list = tl.load_table(self.__curr, 'zones')

        existing_list = [zn['zone_id'] for zn in zones_list]
        if zones_list:
            self.__properties = list(zones_list[0].keys())
        for zn in zones_list:
            if zn['zone_id'] not in self.__items:
                self.__items[zn['zone_id']] = Zone(zn)

        to_del = [key for key in self.__items.keys() if key not in existing_list]
        for key in to_del:
            del self.__items[key]
