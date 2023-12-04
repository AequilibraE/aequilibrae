import logging
import time
import re
from pathlib import Path

import requests
from aequilibrae.parameters import Parameters
from aequilibrae.context import get_logger
import gc
import importlib.util as iutil
from ...utils import WorkerThread

import duckdb
import geopandas as gpd
import subprocess
import os

conn = duckdb.connect()
c = conn.cursor()

c.execute("""INSTALL spatial; 
           INSTALL httpfs;""")
c.execute(
    """LOAD spatial;
    LOAD parquet;
    SET s3_region='us-west-2';
    """
)

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

    def __init__(self, polygons, modes, project_path: Path, logger: logging.Logger = None):
        WorkerThread.__init__(self, None)
        self.logger = logger or get_logger()
        self.polygons = polygons
        self.filter = self.get_ovm_filter(modes)
        self.report = []
        self.gpkg = []
        self.__project_path = project_path

    def downloadPlace(self):

        pth = str(self.__project_path / 'new_geopackage_pla.parquet')
        c.execute(
            f"""
            COPY(
            SELECT
               id,
               CAST(names AS JSON) AS names,
               CAST(categories AS JSON) AS categories,
               CAST(brand AS JSON) AS brand,
               CAST(addresses AS JSON) AS addresses,
               ST_GeomFromWKB(geometry) AS geom
            FROM read_parquet('s3://overturemaps-us-west-2/release/2023-11-14-alpha.0/theme=places/type=*/*', filename=true, hive_partitioning=1)
            WHERE bbox.minx > 148.7077
                AND bbox.maxx < 148.7324
                AND bbox.miny > -20.2780
                AND bbox.maxy < -20.2621 )
            TO "{pth}";
            """
        )

    def downloadTransportation(self):
        pth = str(self.__project_path / 'new_geopackage_tran.parquet').replace("\\", "/")
        c.execute(f"""
            COPY (
            SELECT
               type,
               JSON(bbox) AS bbox,
               connectors,
               road,
               ST_GeomFromWkb(geometry) AS geometry
            FROM read_parquet('s3://overturemaps-us-west-2/release/2023-11-14-alpha.0/theme=transportation/type=*/*', filename=true, hive_partitioning=1)
            WHERE bbox.minx > 148.7077
                AND bbox.maxx < 148.7324
                AND bbox.miny > -20.2780
                AND bbox.maxy < -20.2621 )
            TO '{pth}';
            """
                  )

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
        service = '["service"!~"parking|parking_aisle|driveway|private|emergency_access"]'
        access = '["access"!~"private"]'

        filtered = [x for x in all_tags if x not in tags_to_keep]
        filtered = "|".join(filtered)

        filter = f'["area"!~"yes"]["highway"!~"{filtered}"]{service}{access}'

        return filter
    #
    # def merge_geopackages(output_gpkg_path, *input_gpkg_paths):
    #     # Build the ogr2ogr command to merge GeoPackages
    #     ogr2ogr_command = [
    #         'ogr2ogr',
    #         '-f', 'GPKG',
    #         output_gpkg_path,
    #         *input_gpkg_paths
    #     ]
    #
    #     # Execute the ogr2ogr command
    #     subprocess.run(ogr2ogr_command)
