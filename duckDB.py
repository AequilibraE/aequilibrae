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
    SET s3_region='us-west-2';
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
       id,
       updatetime,
       version,
       CAST(names AS JSON) AS names,
       CAST(categories AS JSON) AS categories,
       confidence,
       CAST(websites AS JSON) AS websites,
       CAST(socials AS JSON) AS socials,
       CAST(emails AS JSON) AS emails,
       CAST(phones AS JSON) AS phones,
       CAST(brand AS JSON) AS brand,
       CAST(addresses AS JSON) AS addresses,
       CAST(sources AS JSON) AS sources,
       ST_GeomFromWKB(geometry)
    FROM
       read_parquet('s3://overturemaps-us-west-2/release/2023-11-14-alpha.0/theme=places/type=*/*', hive_partitioning=1)
    WHERE
        bbox.minx > -122.4447744
        AND bbox.maxx < -122.2477071
        AND bbox.miny > 47.5621587
        AND bbox.maxy < 47.7120663
    ) TO 'places_seattle.shp'
 WITH (FORMAT GDAL, DRIVER 'ESRI Shapefile');"""
 )

# %%
c.execute(
    """
    SELECT
       id,
       confidence,
       CAST(names AS JSON) AS names,
    FROM
       read_parquet('s3://overturemaps-us-west-2/release/2023-11-14-alpha.0/theme=places/type=*/*', filename=true, hive_partitioning=1)
    LIMIT 100
"""
 ).df()
# %%
# %%
import duckdb
import geopandas as gpd
import os

# %%
conn = duckdb.connect()
c = conn.cursor()
# %%
c.execute(
    """LOAD spatial;
    LOAD parquet;
    SET s3_region='us-west-2';
"""
)
# %%
transport_df = c.execute("""
    DESCRIBE SELECT
           type,
           JSON(bbox) AS bbox,
           connectors,
           road,
    FROM read_parquet('s3://overturemaps-us-west-2/release/2023-11-14-alpha.0/theme=transportation/type=*/*')
    WHERE bbox.minx > 148.7077
        AND bbox.maxx < 148.7324
        AND bbox.miny > -20.2780
        AND bbox.maxy < -20.2621
    LIMIT 10;
"""
).df()
# %%
c.execute("""
DESCRIBE SELECT * FROM read_parquet('s3://overturemaps-us-west-2/release/2023-11-14-alpha.0/theme=transportation/type=*/*', filename=true, hive_partitioning=1);
"""
).df()
# %%
transport_df = c.execute("""
    DESCRIBE SELECT
           type,
           JSON(bbox) AS bbox,
           CAST(connectors AS JSON) AS connectors,
           road,
    FROM read_parquet('s3://overturemaps-us-west-2/release/2023-11-14-alpha.0/theme=transportation/type=*/*')
    WHERE bbox.minx > 148.7077
        AND bbox.maxx < 148.7324
        AND bbox.miny > -20.2780
        AND bbox.maxy < -20.2621
    LIMIT 10;
                         """
).df()
# %%
transport_df = c.execute("""
    SELECT
           type,
           JSON(bbox) AS bbox,
    FROM read_parquet('s3://overturemaps-us-west-2/release/2023-11-14-alpha.0/theme=transportation/type=*/*')
    WHERE bbox.minx > 148.7077
        AND bbox.maxx < 148.7324
        AND bbox.miny > -20.2780
        AND bbox.maxy < -20.2621
    LIMIT 10;
                         """
).df()
# %%
transport_df = c.execute("""
    SELECT
           type,
           JSON(bbox) AS bbox,
         
           road,
    FROM read_parquet('s3://overturemaps-us-west-2/release/2023-11-14-alpha.0/theme=transportation/type=*/*')
    WHERE bbox.minx > 148.7077
        AND bbox.maxx < 148.7324
        AND bbox.miny > -20.2780
        AND bbox.maxy < -20.2621
    LIMIT 10;
                         """
).df()
# %%
transport_df
# %%
# file:///C:/Users/penny/AequilibraeLocalRepo/data/type=connector

transport_df = c.execute("""
    SELECT
           JSON(bbox) AS bbox,
           connectors,
           road,
    FROM read_parquet('C:/Users/penny/AequilibraeLocalRepo/data/type=connector/*')
     
                         """
).df()
# %%
from pandas_geojson import to_geojson
geo_json = to_geojson(df=transport_df, lat='lat', lon='long',
                 properties=['type','bbox','road'])
print(geo_json)
# %%
