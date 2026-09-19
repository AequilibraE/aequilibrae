import geopandas as gpd
import pandas as pd


def get_table(table_name, conn):
    """
    Selects table from database.

    :Arguments:
         **table_name** (:obj:`str`): desired table name
         **conn** (:obj:`sqlite3.Connection`): database connection
    """

    return pd.read_sql(f"SELECT * FROM {table_name};", con=conn)


def find_table_fields(table_name, conn):
    structure = conn.execute(f"pragma table_info({table_name})").fetchall()
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
