from sqlite3 import Cursor


class TableLoader():
    def load_table(self, curr: Cursor, table_name: str):
        fields = self.__get_table_struct(curr, table_name)
        curr.execute(f"select * from '{table_name}'")
        return [dict(zip(fields, row)) for row in curr.fetchall()]

    def __get_table_struct(self, curr: Cursor, table_name: str):
        curr.execute(f'pragma table_info({table_name})')
        return [x[1] for x in curr.fetchall()]
