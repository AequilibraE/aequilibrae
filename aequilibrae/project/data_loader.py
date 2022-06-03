from sqlite3 import Connection
import shapely.wkb
import pandas as pd


class DataLoader:
    def __init__(self, conn: Connection, table_name: str):
        self.conn = conn
        self.curr = conn.cursor()
        self.table_name = table_name

    def load_table(self) -> pd.DataFrame:
        fields, _, geo_field = self.__find_table_fields()
        fields = [f'"{x}"' for x in fields]
        if geo_field is not None:
            fields.append('ST_AsBinary("geometry") geometry')
        keys = ",".join(fields)
        df = pd.read_sql_query(f"select {keys} from '{self.table_name}'", self.conn)
        df.geometry = df.geometry.apply(shapely.wkb.loads)
        return df

    def __find_table_fields(self):
        geotypes = ["LINESTRING", "POINT", "POLYGON", "MULTIPOLYGON"]
        self.curr.execute(f"pragma table_info({self.table_name})")
        structure = self.curr.fetchall()
        fields = [x[1].lower() for x in structure]
        geotype = geo_field = None
        for x in structure:
            if x[2].upper() in geotypes:
                geotype = x[2]
                geo_field = x[1]
                break
        if geo_field is not None:
            fields = [x for x in fields if x != geo_field.lower()]

        return fields, geotype, geo_field
