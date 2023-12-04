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
# Downloading places theme to csv
places_df = c.execute(
    """
    SELECT
       id,
       CAST(names AS JSON) AS names,
       CAST(categories AS JSON) AS categories,
       CAST(brand AS JSON) AS brand,
       CAST(addresses AS JSON) AS addresses,
       ST_GeomFromWKB(geometry) AS geom
    FROM
       read_parquet('s3://overturemaps-us-west-2/release/2023-11-14-alpha.0/theme=places/type=*/*', filename=true, hive_partitioning=1)
    LIMIT 100
"""
 ).df()
places_df
# %%
places_df.to_csv("place_csv.csv")


# %%
# Downloading admin theme to csv
admins_df = c.execute("""
    SELECT
           localityType,
           isoCountryCodeAlpha2,
           JSON(names) AS names,
           ST_GeomFromWkb(geometry) AS geometry
    FROM read_parquet('s3://overturemaps-us-west-2/release/2023-07-26-alpha.0/theme=admins/type=*/*')
    WHERE adminLevel = 2
        AND bbox.minx > 148.7077
        AND bbox.maxx < 148.7324
        AND bbox.miny > -20.2780
        AND bbox.maxy < -20.2621 )
    LIMIT 100;
""").df()
admins_df
# %%
admins_df.to_csv("admins_csv.csv")


# %%
# Downloading buildings theme to csv
buildings_df = c.execute("""
    SELECT
           type,
           JSON(sources) AS sources,
           JSON(names) AS names,
           height,
           numFloors,
           class,
           ST_GeomFromWkb(geometry) AS geometry
    FROM read_parquet('s3://overturemaps-us-west-2/release/2023-07-26-alpha.0/theme=buildings/type=*/*')
    WHERE bbox.minx > 148.7077
        AND bbox.maxx < 148.7324
        AND bbox.miny > -20.2780
        AND bbox.maxy < -20.2621 )
    LIMIT 100;
"""
).df()
buildings_df
# %%
buildings_df.to_csv("buildings_csv.csv")


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
table = c.execute("""
SELECT * FROM read_parquet('E:/type=transportation/type=segment/*', filename=true, hive_partitioning=1) LIMIT 2;
"""
).df()

#%%
c.execute("""
DESCRIBE SELECT * FROM read_parquet('E:/type=transportation/type=segment/*', filename=true, hive_partitioning=1) LIMIT 2;
"""
).df()


# %%
# Best version of downloading the transportation segement theme
c.execute("""
    COPY (
        SELECT
          id,
          ST_GeomFromWkb(geometry) AS geometry,
          JSON(bbox) AS bbox,
          JSON(names) AS names,
          road,
          contextId,
          defaultLanguage,
          JSON(sources) AS sources,
          geopolDisplay,
          JSON(connectors) AS connectors,
          JSON(categories) AS categories,
          JSON(addresses) AS addresses,
        FROM read_parquet('E:/type=transportation/type=segment/*')
        WHERE bbox.minx > 148.7077
            AND bbox.maxx < 148.7324
            AND bbox.miny > -20.2780
            AND bbox.maxy < -20.2621 )
    )
    TO 'geopackage_seg.gpkg'
    WITH (FORMAT GDAL, DRIVER 'GPKG', LAYER_CREATION_OPTIONS 'WRITE_BBOX=YES');
""")


# %%
# Best version of downloading the transportation connection theme
c.execute("""
    COPY (SELECT
           type,
           JSON(bbox) AS bbox,
           connectors,
           road,
           ST_GeomFromWkb(geometry) AS geometry
    FROM read_parquet('E:/type=transportation/type=connector/*')
    WHERE bbox.minx > 148.7077
        AND bbox.maxx < 148.7324
        AND bbox.miny > -20.2780
        AND bbox.maxy < -20.2621 )
    TO 'geopackage_con.gpkg'
    WITH (FORMAT GDAL, DRIVER 'GPKG', LAYER_CREATION_OPTIONS 'WRITE_BBOX=YES');
                         """
)


# %%
# Best version of downloading the places theme
c.execute(
    """
    COPY(
    SELECT
       id,
       CAST(names AS JSON) AS names,
       CAST(categories AS JSON) AS categories,
       CAST(brand AS JSON) AS brand,
       CAST(addresses AS JSON) AS addresses,
       ST_GeomFromWKB(geometry) AS geom
    FROM
       read_parquet('s3://overturemaps-us-west-2/release/2023-11-14-alpha.0/theme=places/type=*/*', filename=true, hive_partitioning=1)
    WHERE bbox.minx > 148.7077
        AND bbox.maxx < 148.7324
        AND bbox.miny > -20.2780
        AND bbox.maxy < -20.2621 )
    )
    TO 'geopackage_pla.gpkg'
    WITH (FORMAT GDAL, DRIVER 'GPKG', LAYER_CREATION_OPTIONS 'WRITE_BBOX=YES');
"""
 )


# %%

import ast
classes = c.execute("""
SELECT 
    road,
FROM read_parquet('E:/type=transportation/type=segment/*', filename=true, hive_partitioning=1)
WHERE bbox.minx > 152.9214
        AND bbox.maxx < 153.1789
        AND bbox.miny > -27.5077
        AND bbox.maxy < -27.3885;
"""
).df()


# %%
# Extract 'class' values
class_values = classes['road'].apply(lambda x: ast.literal_eval(x)['class'])

# Get unique 'class' values
unique_classes = class_values.unique()

# Print the unique 'class' values
print(sorted(unique_classes))
# %%
