import shapely.wkb
from shapely.geometry import Polygon

from aequilibrae.project.field_editor import FieldEditor


class BasicTable:
    """
    Basic resources used by all subclasses
    """

    def __init__(self, project):
        self.project = project
        self.__table_type__ = ""
        self.conn = project.connect()
        self._curr = self.conn.cursor()

    def extent(self) -> Polygon:
        """Queries the extent of thelayer  included in the model

        Returns:
            *model extent* (:obj:`Polygon`): Shapely polygon with the bounding box of the layer.
        """
        self.__curr.execute(f'Select ST_asBinary(GetLayerExtent("{self.__table_type__}"))')
        poly = shapely.wkb.loads(self.__curr.fetchone()[0])
        return poly

    @property
    def fields(self) -> FieldEditor:
        """Returns a FieldEditor class instance to edit the zones table fields and their metadata"""
        return FieldEditor(self.project, self.__table_type__)

    def refresh_connection(self):
        """Opens a new database connection to avoid thread conflict"""
        self.conn = self.project.connect()
        self.__curr = self.conn.cursor()

    def __copy__(self):
        raise Exception(f"{self.__table_type__} object cannot be copied")

    def __deepcopy__(self, memodict=None):
        raise Exception(f"{self.__table_type__} object cannot be copied")
