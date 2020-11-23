from sqlite3 import Cursor


class TableLoader():
    def __init__(self):
        self.fields = []

    def load_table(self, curr: Cursor, table_name: str):
        self.__get_table_struct(curr, table_name)
        keys = []
        for fld in self.fields:
            if fld == 'geometry':
                keys.append('ST_AsBinary("geometry")')
            else:
                keys.append(f'"{fld}"')
        keys = ','.join(keys)
        curr.execute(f"select {keys} from '{table_name}'")
        return [dict(zip(self.fields, row)) for row in curr.fetchall()]

    def __get_table_struct(self, curr: Cursor, table_name: str) -> None:
        curr.execute(f'pragma table_info({table_name})')
        self.fields = [x[1].lower() for x in curr.fetchall()]
