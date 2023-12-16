from sqlite3 import Connection
from typing import List


class TableLoader:
    def __init__(self):
        self.fields = []
        self.sql = ""

    def load_table(self, conn: Connection, table_name: str) -> List[dict]:
        self.__get_table_struct(conn, table_name)
        return [dict(zip(self.fields, row)) for row in conn.execute(self.sql).fetchall()]

    def load_structure(self, conn: Connection, table_name: str) -> None:
        self.__get_table_struct(conn, table_name)

    def __get_table_struct(self, conn: Connection, table_name: str) -> None:
        dt = conn.execute(f"pragma table_info({table_name})").fetchall()
        self.fields = [x[1].lower() for x in dt if x[1].lower() != "ogc_fid"]
        keys = [f'"{fld}"' if fld != "geometry" else 'ST_AsBinary("geometry")' for fld in self.fields]
        self.sql = f'select {",".join(keys)} from "{table_name}"'
