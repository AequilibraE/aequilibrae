from os import PathLike
from typing import Union

import geopandas as gpd
import pandas as pd

from aequilibrae.utils.db_utils import commit_and_close
from aequilibrae.utils.spatialite_utils import connect_spatialite


class DataLoader:
    def __init__(self, path_to_file: PathLike, table_name: str):
        self.__pth_file = path_to_file
        self.table_name = table_name

    def load_table(self) -> Union[gpd.GeoDataFrame, pd.DataFrame]:
        with commit_and_close(connect_spatialite(self.__pth_file)) as conn:
            fields, _, geo_field = self.__find_table_fields()
            fields = [f'"{x}"' for x in fields]
            keys = ",".join(fields)
            if geo_field is not None:
                keys += ', Hex(ST_AsBinary("geometry")) as geometry'

            sql = f"select {keys} from '{self.table_name}'"
            if geo_field is None:
                return pd.read_sql_query(sql, conn)
            else:
                return gpd.GeoDataFrame.from_postgis(sql, conn, geom_col="geometry", crs="EPSG:4326")

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
