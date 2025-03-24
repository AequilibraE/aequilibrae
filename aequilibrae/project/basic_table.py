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

    def extent(self) -> Polygon:
        """Queries the extent of the layer included in the model

        :Returns:
            **model extent** (:obj:`Polygon`): Shapely polygon with the bounding box of the layer.
        """
        with self.project.db_connection as conn:
            data = conn.execute(f'Select ST_asBinary(GetLayerExtent("{self.__table_type__}"))').fetchone()[0]
        return shapely.wkb.loads(data)

    @property
    def fields(self) -> FieldEditor:
        """Returns a FieldEditor class instance to edit the zones table fields and their metadata"""
        return FieldEditor(self.project, self.__table_type__)

    def __copy__(self):
        raise Exception(f"{self.__table_type__} object cannot be copied")

    def __deepcopy__(self, memodict=None):
        raise Exception(f"{self.__table_type__} object cannot be copied")
