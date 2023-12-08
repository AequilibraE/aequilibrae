import logging
import time
import re
from pathlib import Path

import requests
from aequilibrae.parameters import Parameters
from aequilibrae.context import get_logger
import gc
import importlib.util as iutil
from aequilibrae.utils import WorkerThread

import duckdb
import geopandas as gpd
import subprocess
import os
from typing import Union

DEFAULT_OVM_S3_LOCATION = "s3://overturemaps-us-west-2/release/2023-11-14-alpha.0"


def initialise_duckdb_spatial():
    conn = duckdb.connect()
    c = conn.cursor()

    c.execute(
        """INSTALL spatial; 
            INSTALL httpfs;"""
    )
    c.execute(
        """LOAD spatial;
        LOAD parquet;
        SET s3_region='us-west-2';
        """
    )
    return c


spec = iutil.find_spec("PyQt5")
pyqt = spec is not None
if pyqt:
    from PyQt5.QtCore import pyqtSignal


class OVMDownloader(WorkerThread):
    if pyqt:
        downloading = pyqtSignal(object)

    def __emit_all(self, *args):
        if pyqt:
            self.downloading.emit(*args)

    def __init__(self, bbox, modes, project_path: Union[str, Path], logger: logging.Logger = None):
        WorkerThread.__init__(self, None)
        self.logger = logger or get_logger()
        self.bbox = bbox
        self.filter = self.get_ovm_filter(modes)
        self.report = []
        self.gpkg = []
        self.__project_path = Path(project_path)

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
               CAST(names AS JSON) AS names,
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
        c.execute(sql)

    def downloadTransportation(self, bbox, data_source=None):
        pth = str(self.__project_path / "downloaded_ovm_data.parquet").replace("\\", "/")

        data_source = data_source or DEFAULT_OVM_S3_LOCATION

        sql = f"""
            COPY (
            SELECT JSON(bbox) AS bbox,
                ST_GeomFromWkb(geometry) AS geometry, 
                road, 
                connectors 
            FROM read_parquet('{data_source}/theme=transportation/type=*', filename=true, hive_partitioning=1)
            WHERE bbox.minx > '{bbox[0]}'
                AND bbox.maxx < '{bbox[2]}'
                AND bbox.miny > '{bbox[1]}'
                AND bbox.maxy < '{bbox[3]}')
            TO '{pth}';
        """
        c = initialise_duckdb_spatial()
        c.execute(sql)


    def download_test_data(data_source, test_data_location):
        '''This method only used to seed/bootstrap a local copy of a small test data set'''
        airlie_bbox = [....]
        sql = f"""
            COPY (
            SELECT JSON(bbox) AS bbox,
                ST_GeomFromWkb(geometry) AS geometry, 
                road, 
                connectors 
            FROM read_parquet('{data_source}/theme=transportation/type=*', filename=true, hive_partitioning=1)
            WHERE bbox.minx > '{airlie_bbox[0]}'
                AND bbox.maxx < '{airlie_bbox[2]}'
                AND bbox.miny > '{airlie_bbox[1]}'
                AND bbox.maxy < '{airlie_bbox[3]}')
            TO '{test_data_location}';
        """
        initialise_duckdb_spatial().execute(sql)

    def get_ovm_filter(self, modes: list) -> str:
        """
        loosely adapted from http://www.github.com/gboeing/osmnx
        """

        p = Parameters().parameters["network"]["osm"]
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
