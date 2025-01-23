import logging
from pathlib import Path
from typing import Union, List, Dict

import duckdb
import geopandas as gpd
from shapely import Polygon

from aequilibrae.context import get_active_project
from aequilibrae.context import get_logger
from aequilibrae.utils.aeq_signal import SIGNAL
from aequilibrae.utils.interface.worker_thread import WorkerThread

S3_OVERTURE = "s3://overturemaps-us-west-2/release/2025-01-22.0"


class OVMDownloader(WorkerThread):
    signal = SIGNAL(object)

    def __init__(self, polygons: List[Polygon], logger: logging.Logger = None, project: Union[str, Path] = None):
        WorkerThread.__init__(self, None)
        self.logger = logger or get_logger()
        self.polygons = polygons
        self.data: Dict[str, gpd.GeoDataFrame] = {"nodes": gpd.GeoDataFrame([]), "links": gpd.GeoDataFrame([])}
        self.project = project or get_active_project()

    def doWork(self):
        self.initialise_duckdb_spatial()

        self.signal.emit(["set_text", "Create ovm_data external folder"])
        ovm_data_path = Path(self.project.project_base_path) / "ovm_data"
        ovm_data_path.mkdir(exist_ok=True)

        self.signal.emit(["set_text", "Downloading data"])
        self.download_transportation(self.project.project_base_path)

    def initialise_duckdb_spatial(self):
        self.signal.emit(["set_text", "Connecting to Duck DB"])
        conn = duckdb.connect()
        self._c = conn.cursor()

        self._c.execute(
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

    def download_place(self, output_dir: Union[str, Path]):
        xmin, ymin, xmax, ymax = self.polygons.bounds

        out_places = Path(output_dir) / "ovm_data" / "places.parquet"

        sql = f"""
            COPY(
            SELECT
                id as ovm_id,
                sources[1].dataset as source,
                names.primary as name,
                categories.primary as primary_categories,
                categories.alternate as alternate_categories,
                confidence,
                addresses[1].freeform as addresses,
                ST_AsText(geometry) as geometry
            FROM read_parquet('{S3_OVERTURE}/theme=places/*/*')
            WHERE 
                bbox.xmin > {xmin} AND 
                bbox.xmax < {xmax} AND 
                bbox.ymin > {ymin} AND 
                bbox.ymax < {ymax})
            TO '{out_places}'
            (FORMAT 'parquet', COMPRESSION 'zstd');
            """

        self._c.execute(sql)

    def download_transportation(self, output_dir: Union[str, Path]):
        xmin, ymin, xmax, ymax = self.polygons.bounds

        out_links = Path(output_dir) / "ovm_data" / "segments.parquet"
        out_nodes = Path(output_dir) / "ovm_data" / "connectors.parquet"

        self.signal.emit(["set_text", "Downloading links"])
        sql_link = f"""
            COPY (
                  SELECT
                    id as ovm_id,
                    connectors,
                    names.primary as name,
                    class,
                    subclass,
                    access_restrictions,
                    speed_limits,
                    prohibited_transitions,
                    geometry
                  FROM read_parquet('{S3_OVERTURE}/theme=transportation/type=segment/*')
                  WHERE
                    (subclass NOT IN ('parking_aisle', 'driveway') OR subclass IS NULL) AND 
                    (bbox.xmin > {xmin} AND 
                    bbox.xmax < {xmax} AND 
                    bbox.ymin > {ymin} AND 
                    bbox.ymax < {ymax}))
            TO '{out_links}'
            (FORMAT 'parquet', COMPRESSION 'zstd');
        """
        self._c.execute(sql_link)

        self.signal.emit(["set_text", "Downloading nodes"])
        sql_node = f"""
            COPY (
                  SELECT
                    id AS ovm_id,
                    geometry
                  FROM read_parquet('{S3_OVERTURE}/theme=transportation/type=connector/*')
                  WHERE 
                    bbox.xmin > {xmin} AND 
                    bbox.xmax < {xmax} AND 
                    bbox.ymin > {ymin} AND 
                    bbox.ymax < {ymax})
            TO '{out_nodes}'
            (FORMAT 'parquet', COMPRESSION 'zstd');
        """
        self._c.execute(sql_node)

        self.signal.emit(["set_text", "Downloaded connectors and segments"])

        self.data["links"] = gpd.read_parquet(out_links)
        self.data["nodes"] = gpd.read_parquet(out_nodes)
