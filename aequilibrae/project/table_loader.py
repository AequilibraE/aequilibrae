from sqlite3 import Cursor


class TableLoader():
    def load_table(self, curr: Cursor, table_name: str):
        fields = self.__get_table_struct(curr, table_name)
        keys = []
        for fld in fields:
            if fld == 'geometry':
                keys.append('ST_AsBinary("geometry")')
            else:
                keys.append(f'"{fld}"')
        keys = ','.join(keys)
        curr.execute(f"select {keys} from '{table_name}'")
        return [dict(zip(fields, row)) for row in curr.fetchall()]

    def __get_table_struct(self, curr: Cursor, table_name: str):
        curr.execute(f'pragma table_info({table_name})')
        return [x[1].lower() for x in curr.fetchall()]
