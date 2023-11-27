# %%
import duckdb
db = duckdb.connect()
db.execute("INSTALL spatial")
db.execute("INSTALL httpfs")
db.execute("""
LOAD spatial;
LOAD httpfs;
SET s3_region='us-west-2';
""")
# %%
db.execute("""
COPY (
    SELECT
           type,
           subType,
           localityType,
           adminLevel,
           isoCountryCodeAlpha2,
           JSON(names) AS names,
           JSON(sources) AS sources,
           ST_GeomFromWkb(geometry) AS geometry
      FROM read_parquet('https://overturemapswestus2.blob.core.windows.net/release/2023-11-14-alpha.0/theme=admins/', filename=true, hive_partitioning=1)
     WHERE adminLevel = 2
       AND ST_GeometryType(ST_GeomFromWkb(geometry)) IN ('POLYGON','MULTIPOLYGON')
) TO '/tmp/countries.geojson'
WITH (FORMAT GDAL, DRIVER 'GeoJSON');
""")

# %%
query_buildings = """
COPY (
    SELECT
           theme,
           type,
           version,
           updateTime,
           JSON(sources) AS sources,
           JSON(names) AS names,
           height,
           numFloors,
           class,
           ST_GeomFromWkb(geometry) AS geometry
      FROM read_parquet('s3://overturemaps-us-west-2/release/2023-07-26-alpha.0/theme=buildings/type=*/*')
     WHERE ST_Within(ST_GeomFromWkb(geometry), ST_Envelope(ST_GeomFromText('POLYGON((-81.0575 -2.3667,-81.0555 -2.1611,-80.7804 -2.1437,-80.7785 -2.3854,-81.0575 -2.3667))')))
) TO 'buildings.parquet'
WITH (FORMAT PARQUET);
"""
# %%

import duckdb
import pandas as pd

# %%
con = duckdb.connect()
con.execute("INSTALL httpfs;")
con.execute("INSTALL spatial;")
con.execute("LOAD httpfs;")
con.execute("LOAD spatial;")
con.execute("SET s3_region='us-west-2';")



# %%
con.sql(query_buildings)



# %%
import duckdb
import geopandas as gpd
import os

# %%
conn = duckdb.connect("overture.duckdb")
c = conn.cursor()
# %%
c.execute(
    """INSTALL spatial;
    INSTALL httpfs;
    LOAD spatial;
    LOAD parquet;
"""
)
# %%
sql = """
SELECT
    count(*)
FROM
    read_parquet('s3://overturemaps-us-west-2/release/2023-11-14-alpha.0/theme=buildings/type=*/*')
"""

print(c.execute(sql).fetchall())
# %%
c.execute(
    """COPY (
    SELECT
           theme,
           type,
           version,
           updateTime,
           JSON(sources) AS sources,
           JSON(names) AS names,
           height,
           numFloors,
           class,
           ST_GeomFromWkb(geometry) AS geometry
      FROM read_parquet('s3://overturemaps-us-west-2/release/2023-11-14-alpha.0/theme=buildings/type=*/*')
     WHERE ST_Within(ST_GeomFromWkb(geometry), ST_Envelope(ST_GeomFromText('POLYGON((122.934570 20.425378, 122.934570 45.551483, 153.986672 45.551483, 153.986672 20.425378, 122.934570 20.425378))')))
) TO '../data/buildings.parquet'
WITH (FORMAT PARQUET);
"""
)
# %%
gdf = gpd.read_parquet("buildings.parquet")
gdf.head()