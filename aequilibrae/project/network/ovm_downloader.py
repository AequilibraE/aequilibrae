import logging
from pathlib import Path
from typing import Union, List

import duckdb
import geopandas as gpd
from shapely import Polygon

from aequilibrae.context import get_logger 
from aequilibrae.utils.aeq_signal import SIGNAL
from aequilibrae.utils.interface.worker_thread import WorkerThread

S3_TRANSPORTATION = "s3://overturemaps-us-west-2/release/2024-12-18.0/theme=transportation"
S3_PLACES = "s3://overturemaps-us-west-2/release/2024-12-18.0/theme=places"


class OVMDownloader(WorkerThread):
    signal = SIGNAL(object)

    def __init__(self, polygons: List[Polygon], modes, logger: logging.Logger = None):
        WorkerThread.__init__(self, None)
        self.logger = logger or get_logger()
        self.polygons = polygons
        self.filter = self.get_ovm_filter(modes)
        self.GeoDataFrame = []

    def initialise_duckdb_spatial(self):
        conn = duckdb.connect()
        self.signal.emit(["set_text", "Connecting to Duck DB"])
        c = conn.cursor()

        c.execute(
            """
            INSTALL spatial;
            INSTALL httpfs;
            INSTALL parquet;
            LOAD spatial;
            LOAD httpfs;
            SET s3_region='us-west-2';
            """
        )

        self.signal.emit(["set_text", "Database initialised"])
        return c

    def download_place(self, output_dir: Union[str, Path]):
        xmin, ymin, xmax, ymax = self.polygons.bounds

        ovm_data_path = Path(output_dir) / "ovm_data"
        ovm_data_path.mkdir(exist_ok=True)

        out_places = Path(output_dir) / "ovm_data" / "places.parquet"

        sql = f"""
            COPY(
            SELECT
               id,
               CAST(names AS JSON) AS name,
               CAST(categories AS JSON) AS categories,
               CAST(brand AS JSON) AS brand,
               CAST(addresses AS JSON) AS addresses,
               geometry
            FROM read_parquet('{S3_PLACES}/type=*', filename=true, hive_partitioning=1)
            WHERE 
                bbox.xmin > {xmin} AND 
                bbox.xmax < {xmax} AND 
                bbox.ymin > {ymin} AND 
                bbox.ymax < {ymax})
            TO '{out_places}'
            (FORMAT 'parquet', COMPRESSION 'zstd');
            """

        c = self.initialise_duckdb_spatial()
        c.execute(sql)

    def download_transportation(self, output_dir: Union[str, Path]):
        xmin, ymin, xmax, ymax = self.polygons.bounds

        # ovm_data_path = Path(self.project.project_base_path) / "ovm_data"
        ovm_data_path = Path(output_dir) / "ovm_data"
        ovm_data_path.mkdir(exist_ok=True)

        out_links = Path(output_dir) / "ovm_data" / "segments.parquet"
        out_nodes = Path(output_dir) / "ovm_data" / "connectors.parquet"

        c = self.initialise_duckdb_spatial()

        sql_link = f"""
            COPY (
                  SELECT
                      id AS ovm_id,
                      class AS link_type,
                      names.primary AS name,
                      speed_limits[1].max_speed.value AS speed,
                      access_restrictions[1].when.heading AS direction,
                      geometry
                  FROM read_parquet('{S3_TRANSPORTATION}/type=segment/*', union_by_name=True)
                  WHERE 
                      bbox.xmin > {xmin} AND 
                      bbox.xmax < {xmax} AND 
                      bbox.ymin > {ymin} AND 
                      bbox.ymax < {ymax})
            TO '{out_links}'
            (FORMAT 'parquet', COMPRESSION 'zstd');
        """
        c.execute(sql_link)

        sql_node = f"""
            COPY (
                  SELECT
                      id AS ovm_id,
                      geometry
                  FROM read_parquet('{S3_TRANSPORTATION}/type=connector/*', union_by_name=True)
                  WHERE 
                      bbox.xmin > {xmin} AND 
                      bbox.xmax < {xmax} AND 
                      bbox.ymin > {ymin} AND 
                      bbox.ymax < {ymax})
            TO '{out_nodes}'
            (FORMAT 'parquet', COMPRESSION 'zstd');
        """
        c.execute(sql_node)

        self.signal.emit(["set_text", "Downloaded connectors and segments"])

        links = gpd.read_parquet(out_links)
        nodes = gpd.read_parquet(out_nodes)
        return links, nodes


    # def get_ovm_filter(self, modes: list) -> str:
    #     """
    #     loosely adapted from http://www.github.com/gboeing/osmnx
    #     """

    #     p = Parameters().parameters["network"]["ovm"]
    #     all_tags = p["all_link_types"]

    #     p = p["modes"]
    #     all_modes = list(p.keys())

    #     tags_to_keep = []
    #     for m in modes:
    #         if m not in all_modes:
    #             raise ValueError(f"Mode {m} not listed in the parameters file")
    #         tags_to_keep += p[m]["link_types"]
    #     tags_to_keep = list(set(tags_to_keep))

    #     # Default to remove
    #     service = '["service"!~"parking|parking_aisle|driveway|private|emergency_access"]'
    #     access = '["access"!~"private"]'

    #     filtered = [x for x in all_tags if x not in tags_to_keep]
    #     filtered = "|".join(filtered)

    #     filter = f'["area"!~"yes"]["highway"!~"{filtered}"]{service}{access}'

    #     return filter

