import json
import importlib.util as iutil
import sqlite3
import logging
from pathlib import Path
import string

from aequilibrae.context import get_active_project
from aequilibrae.parameters import Parameters
from aequilibrae.project.network.link_types import LinkTypes
from aequilibrae.context import get_logger
import importlib.util as iutil
from aequilibrae.utils.spatialite_utils import connect_spatialite
from aequilibrae.project.network.haversine import haversine
from aequilibrae.utils import WorkerThread

# from .haversine import haversine
# from ...utils import WorkerThread

import duckdb
import shapely
import geopandas as gpd
import pandas as pd
import numpy as np
from typing import Union
from shapely.geometry import LineString, Point

DEFAULT_OVM_S3_LOCATION = "s3://overturemaps-us-west-2/release/2023-11-14-alpha.0//theme=transportation"

spec = iutil.find_spec("PyQt5")
pyqt = spec is not None
if pyqt:
    from PyQt5.QtCore import pyqtSignal

spec = iutil.find_spec("qgis")
isqgis = spec is not None
if isqgis:
    import qgis 

class OVMDownloader(WorkerThread):
    if pyqt:
        downloading = pyqtSignal(object)

    def __emit_all(self, *args):
        if pyqt:
            self.downloading.emit(*args)

    def __init__(self, modes, project_path: Union[str, Path], logger: logging.Logger = None) -> None:
        WorkerThread.__init__(self, None)
        # self.project = project or get_active_project()
        self.logger = logger or get_logger()
        self.filter = self.get_ovm_filter(modes)
        # self.node_start = node_start
        # self.report = []
        # self.conn = None
        self.GeoDataFrame = []
        # self.nodes = {}
        # self.node_ids = {}  
        self.g_dataframes =[]
        # self.__link_types = None  # type: LinkTypes
        # self.__model_link_types = []
        # self.__model_link_type_ids = []
        # self.__link_type_quick_reference = {}
        self.__project_path = Path(project_path)
        self.pth = str(self.__project_path).replace("\\", "/")
        self.insert_qry = """INSERT INTO {} ({}, geometry) VALUES({}, GeomFromText(?, 4326))"""

    def initialise_duckdb_spatial(self):
        conn = duckdb.connect()
        c = conn.cursor()

        c.execute(
            """INSTALL spatial; 
                INSTALL httpfs;
                INSTALL parquet;
            """
        )
        c.execute(
            """LOAD spatial;
            LOAD parquet;
            SET s3_region='us-west-2';
            """
        )
        return c

    def downloadPlace(self, source, local_file_path=None):
        pth = str(self.__project_path / "new_geopackage_pla.parquet").replace("\\", "/")

        if source == "s3":
            data_source = "s3://overturemaps-us-west-2/release/2023-11-14-alpha.0/theme=places/type=*"
        elif source == "local":
            data_source = local_file_path.replace("\\", "/")
        else:
            raise ValueError("Invalid source. Use 's3' or provide a valid local file path.")

        sql = f"""
            COPY(
            SELECT
               id,
               CAST(names AS JSON) AS name,
               CAST(categories AS JSON) AS categories,
               CAST(brand AS JSON) AS brand,
               CAST(addresses AS JSON) AS addresses,
               ST_GeomFromWKB(geometry) AS geom
            FROM read_parquet('{data_source}/*', filename=true, hive_partitioning=1)
            WHERE bbox.minx > '{self.bbox[0]}'
                AND bbox.maxx < '{self.bbox[2]}'
                AND bbox.miny > '{self.bbox[1]}'
                AND bbox.maxy < '{self.bbox[3]}')
            TO '{pth}';
            """
        
        c = self.initialise_duckdb_spatial()
        c.execute(sql)


    def downloadTransportation(self, bbox, data_source, output_dir):
        data_source = Path(data_source) or DEFAULT_OVM_S3_LOCATION
        output_dir = Path(output_dir)
        # g_dataframes = []
        # for t in ['segment','connector']:
        # print(output_dir)
            
        output_file_link = output_dir / f'type=segment' / f'transportation_data_segment.parquet'
        output_file_node = output_dir / f'type=connector' / f'transportation_data_connector.parquet'
            # output_file = output_dir  / f'type={t}' / f'transportation_data_{t}.parquet'
        output_file_link.parent.mkdir(parents=True, exist_ok=True)
        output_file_node.parent.mkdir(parents=True, exist_ok=True)

        
        # sql = f"""
        #     DESCRIBE
        #     SELECT 
        #         road
        #     FROM read_parquet('{data_source}/type=segment/*', union_by_name=True)
        # """
        # c = self.initialise_duckdb_spatial()
        # g = c.execute(sql)
        # print(g.df())
        
        sql_link = f"""
            COPY (
            SELECT 
                id AS ovm_id,
                connectors,
                CAST(road AS JSON) ->>'lanes' AS direction,
                CAST(road AS JSON) ->>'class' AS link_type,
                CAST(road AS JSON) ->>'roadNames' ->>'common' AS name,
                CAST(road AS JSON) ->>'restrictions' ->> 'speedLimits' AS speed,
                road,
                geometry
            FROM read_parquet('{data_source}/type=segment/*', union_by_name=True)
            WHERE bbox.minx > '{bbox[0]}'
                AND bbox.maxx < '{bbox[2]}'
                AND bbox.miny > '{bbox[1]}'
                AND bbox.maxy < '{bbox[3]}')
            TO '{output_file_link}'
            (FORMAT 'parquet', COMPRESSION 'ZSTD');
        """
        c = self.initialise_duckdb_spatial()
        c.execute(sql_link)

        sql_node = f"""
            COPY (
            SELECT 
                id AS ovm_id,
                geometry
            FROM read_parquet('{data_source}/type=connector/*', union_by_name=True)
            WHERE bbox.minx > '{bbox[0]}'
                AND bbox.maxx < '{bbox[2]}'
                AND bbox.miny > '{bbox[1]}'
                AND bbox.maxy < '{bbox[3]}')
            TO '{output_file_node}'
            (FORMAT 'parquet', COMPRESSION 'ZSTD');
        """
        c.execute(sql_node)

        # Creating links GeoDataFrame
        df_link = pd.read_parquet(output_file_link)
        geo_link = gpd.GeoSeries.from_wkb(df_link.geometry, crs=4326)
        gdf_link = gpd.GeoDataFrame(df_link,geometry=geo_link)
        self.g_dataframes.append(gdf_link)

        # Creating nodes GeoDataFrame
        df_node = pd.read_parquet(output_file_node)
        geo_node = gpd.GeoSeries.from_wkb(df_node.geometry, crs=4326)
        gdf_node = gpd.GeoDataFrame(df_node,geometry=geo_node)
        self.g_dataframes.append(gdf_node)

    def download_test_data(self, l_data_source):
        '''This method only used to seed/bootstrap a local copy of a small test data set'''
        airlie_bbox = [148.7077, -20.2780, 148.7324, -20.2621 ]
        # brisbane_bbox = [153.1771, -27.6851, 153.2018, -27.6703]
        data_source = l_data_source.replace("\\", "/")


        for t in ['segment','connector']:
            (Path(__file__).parent.parent.parent.parent / "tests" / "data" / "overture" / "theme=transportation" / f'type={t}').mkdir(parents=True, exist_ok=True)
            pth1 = Path(__file__).parent.parent.parent.parent / "tests" / "data" / "overture" / "theme=transportation" / f"type={t}" / f'airlie_beach_transportation_{t}.parquet'
            sql = f"""
                COPY (
                SELECT 
                    *
                FROM read_parquet('{data_source}/type={t}/*', union_by_name=True)
                WHERE bbox.minx > '{airlie_bbox[0]}'
                    AND bbox.maxx < '{airlie_bbox[2]}'
                    AND bbox.miny > '{airlie_bbox[1]}'
                    AND bbox.maxy < '{airlie_bbox[3]}')
                TO '{pth1}'
                (FORMAT 'parquet', COMPRESSION 'ZSTD');
            """
            c = self.initialise_duckdb_spatial()
            c.execute(sql)

            df = pd.read_parquet(Path(pth1))
            geo = gpd.GeoSeries.from_wkb(df.geometry, crs=4326)
            gdf = gpd.GeoDataFrame(df,geometry=geo)
            gdf.to_parquet(Path(pth1))
        # return gdf    

    def get_ovm_filter(self, modes: list) -> str:
        """
        loosely adapted from http://www.github.com/gboeing/osmnx
        """

        p = Parameters().parameters["network"]["ovm"]
        all_tags = p["all_link_types"]

        p = p["modes"]
        all_modes = list(p.keys())

        tags_to_keep = []
        for m in modes:
            if m not in all_modes:
                raise ValueError(f"Mode {m} not listed in the parameters file")
            tags_to_keep += p[m]["link_types"]
        tags_to_keep = list(set(tags_to_keep))

        # Default to remove
        # service = '["service"!~"parking|parking_aisle|driveway|private|emergency_access"]'
        # access = '["access"!~"private"]'

        filtered = [x for x in all_tags if x not in tags_to_keep]
        filtered = "|".join(filtered)

        # filter = f'["area"!~"yes"]["highway"!~"{filtered}"]{service}{access}'

        return filter
    