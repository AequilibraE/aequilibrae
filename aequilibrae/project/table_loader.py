from sqlite3 import Cursor
from typing import List


class TableLoader:
    def __init__(self):
        self.fields = []
        self.sql = ""

    def load_table(self, curr: Cursor, table_name: str) -> List[dict]:
        self.__get_table_struct(curr, table_name)
        curr.execute(self.sql)
        return [dict(zip(self.fields, row)) for row in curr.fetchall()]

    def load_structure(self, curr: Cursor, table_name: str) -> None:
        self.__get_table_struct(curr, table_name)

    def __get_table_struct(self, curr: Cursor, table_name: str) -> None:
        curr.execute(f"pragma table_info({table_name})")
        self.fields = [x[1].lower() for x in curr.fetchall() if x[1].lower() != "ogc_fid"]
        keys = [f'"{fld}"' if fld != "geometry" else 'ST_AsBinary("geometry")' for fld in self.fields]
        self.sql = f'select {",".join(keys)} from "{table_name}"'
