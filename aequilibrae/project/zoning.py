from sqlite3 import IntegrityError, Connection
from aequilibrae import logger
from .zone import Zone
from aequilibrae.project.field_editor import FieldEditor
from aequilibrae.project.table_loader import TableLoader


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
        self.conn = project.conn  # type: Connection
        self.curr = project.conn.cursor()

        tl = TableLoader()
        zones_list = tl.load_table(self.curr, 'zones')

        existing_list = [zn['zone_id'] for zn in zones_list]
        if zones_list:
            self.__properties = list(zones_list[0].keys())
        for zn in zones_list:
            if zn['zone_id'] not in self.__items:
                self.__items[zn['zone_id']] = Zone(zn)

        to_del = [key for key in self.__items.keys() if key not in existing_list]
        for key in to_del:
            del self.__items[key]

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
