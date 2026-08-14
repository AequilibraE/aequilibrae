from __future__ import annotations
from typing import TYPE_CHECKING
import shapely.wkb

if TYPE_CHECKING:
    from aequilibrae import Project
    from logging import Logger


class SafeClass:
    _srid: int = 4326
    project: Project
    _logger: Logger
    __original__: dict
    _table: str
    __srid__: int

    def __init__(self, data_set: dict, project) -> None:
        self.__dict__["__original__"] = {}
        self.__dict__["project"] = project
        self.__dict__["_table"] = ""
        self.__dict__["__srid__"] = 4326
        for k, v in data_set.items():
            if k == "geometry" and v is not None:
                v = shapely.wkb.loads(v)
            self.__dict__[k] = v
            self.__original__[k] = v

    def __copy__(self):
        raise Exception("Object cannot be copied")

    def __deepcopy__(self, memodict=None):
        raise Exception("Object cannot be copied")

    def _save_new_with_geometry(self):
        data = []
        up_keys = []
        for key, val in self.__dict__.items():
            if key not in self.__original__ or key == "geometry":
                continue
            up_keys.append(f'"{key}"')
            data.append(val)
        markers = ",".join(["?"] * len(up_keys)) + ",GeomFromWKB(?, ?)"
        up_keys.append("geometry")
        data.extend([self.geometry.wkb, self.__srid__])
        sql = f"Insert into {self._table} ({','.join(up_keys)}) values({markers})"
        return data, sql
