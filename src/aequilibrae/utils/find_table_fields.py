from aequilibrae.utils.db_utils import read_and_close


def find_table_fields(table_name, conn=None, db_path=None):
    with conn or read_and_close(db_path, spatial=True) as conn:
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
