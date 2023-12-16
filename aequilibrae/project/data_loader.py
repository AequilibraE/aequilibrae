from os import PathLike

import pandas as pd
import shapely.wkb

from aequilibrae.utils.db_utils import commit_and_close
from aequilibrae.utils.spatialite_utils import connect_spatialite


class DataLoader:
    def __init__(self, path_to_file: PathLike, table_name: str):
        self.__pth_file = path_to_file
        self.table_name = table_name

    def load_table(self) -> pd.DataFrame:
        with commit_and_close(connect_spatialite(self.__pth_file)) as conn:
            fields, _, geo_field = self.__find_table_fields()
            fields = [f'"{x}"' for x in fields]
            if geo_field is not None:
                fields.append('ST_AsBinary("geometry") geometry')
            keys = ",".join(fields)
            df = pd.read_sql_query(f"select {keys} from '{self.table_name}'", conn)
        if geo_field is not None:
            df.geometry = df.geometry.apply(shapely.wkb.loads)
        return df

    def __find_table_fields(self):
        with commit_and_close(connect_spatialite(self.__pth_file)) as conn:
            structure = conn.execute(f"pragma table_info({self.table_name})").fetchall()
        geotypes = ["LINESTRING", "POINT", "POLYGON", "MULTIPOLYGON"]
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
