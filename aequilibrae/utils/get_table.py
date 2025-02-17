import geopandas as gpd
import pandas as pd

from aequilibrae.utils.find_table_fields import find_table_fields


def get_table(table_name, conn):
    """
    Selects table from database.

    :Arguments:
         **table_name** (:obj:`str`): desired table name
         **conn** (:obj:`sqlite3.Connection`): database connection
    """

    return pd.read_sql(f"SELECT * FROM {table_name};", con=conn)


def get_geo_table(table_name, conn):
    fields, _, geo_field = find_table_fields(table_name, conn=conn)
    fields = [f'"{x}"' for x in fields]
    keys = ",".join(fields)
    if geo_field is not None:
        keys += ', Hex(ST_AsBinary("geometry")) as geometry'

    sql = f"select {keys} from '{table_name}'"
    if geo_field is None:
        return pd.read_sql_query(sql, conn)
    else:
        return gpd.GeoDataFrame.from_postgis(sql, conn, geom_col="geometry", crs="EPSG:4326")
