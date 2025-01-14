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

S3_TRANSPORTATION = "s3://overturemaps-us-west-2/release/2024-12-18.0/theme=transportation"
S3_PLACES = "s3://overturemaps-us-west-2/release/2024-12-18.0/theme=places"


class OVMDownloader(WorkerThread):
    signal = SIGNAL(object)

    def __init__(self, polygons: List[Polygon], modes, logger: logging.Logger = None, project: Union[str, Path] = None):
        WorkerThread.__init__(self, None)
        self.logger = logger or get_logger()
        self.polygons = polygons
        self.filter = self.get_ovm_filter(modes)
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

        self._c.execute(sql)

    def download_transportation(self, output_dir: Union[str, Path]):
        xmin, ymin, xmax, ymax = self.polygons.bounds

        out_links = Path(output_dir) / "ovm_data" / "segments.parquet"
        out_nodes = Path(output_dir) / "ovm_data" / "connectors.parquet"

        self.signal.emit(["set_text", "Downloading links"])
        sql_link = f"""
            COPY (
                  SELECT
                      id AS ovm_id,
                      class AS link_type,
                      names.primary AS name,
                      speed_limits[1].max_speed.value AS max_speed,
                      speed_limits[2].max_speed.unit AS speed_unit,
                      access_restrictions[1].when.heading AS restrict_direction,
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
        self._c.execute(sql_link)

        self.signal.emit(["set_text", "Downloading nodes"])
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
        self._c.execute(sql_node)

        self.signal.emit(["set_text", "Downloaded connectors and segments"])

        self.data["links"] = gpd.read_parquet(out_links)
        self.data["nodes"] = gpd.read_parquet(out_nodes)

    def get_ovm_filter(self, modes: list) -> str:
        """
        Analogous to get_osm_filter
        """

        # subclass != parking_aisle|driveway
        # access_restrictions[1].when.recognized[0] != as_private
        
        # I'm not sure about the modes and the parameters set (see project notes)
        
        pass