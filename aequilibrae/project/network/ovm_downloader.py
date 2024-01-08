import json
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
import shapely
import geopandas as gpd
import pandas as pd
import numpy as np
import subprocess
import os
from typing import Union
from shapely import count_coordinates, segmentize
from shapely.geometry import LineString

from aequilibrae.utils.spatialite_utils import connect_spatialite

DEFAULT_OVM_S3_LOCATION = "s3://overturemaps-us-west-2/release/2023-11-14-alpha.0//theme=transportation"





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

    def __init__(self, modes, project_path: Union[str, Path], logger: logging.Logger = None, node_start=10000):
        WorkerThread.__init__(self, None)
        self.logger = logger or get_logger()
        self.filter = self.get_ovm_filter(modes)
        self.node_start = node_start
        self.report = []
        self.conn = None
        self.GeoDataFrame = []
        self.nodes = {}
        self.node_ids = {}
        self.__project_path = Path(project_path)
        self.pth = str(self.__project_path / 'theme=transportation').replace("\\", "/")

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
        
        c = self.initialise_duckdb_spatial()
        c.execute(sql)

    '''
    from stack overflow
    https://stackoverflow.com/questions/62053253/how-to-split-a-linestring-to-segments
    '''
    def segments(self, curve):
        return list(map(LineString, zip(curve.coords[:-1], curve.coords[1:])))


    def downloadTransportation(self, bbox, data_source, output_dir):
        # self.conn = connect_spatialite(output_dir)
        # self.curr = self.conn.cursor()

        data_source = Path(data_source) or DEFAULT_OVM_S3_LOCATION
        output_dir = Path(output_dir)
        g_dataframes = []
        for t in ['segment','connector']:
            
            output_file = output_dir  / f'type={t}' / f'transportation_data_{t}.parquet'
            output_file.parent.mkdir(parents=True, exist_ok=True)

            output_file_link = output_dir  / f'type=segment' / f'transportation_data_segment.parquet'
            output_file_node = output_dir  / f'type=connector' / f'transportation_data_connector.parquet'
            sql_link = f"""
                COPY (
                SELECT 
                    id AS ovm_id,
                    connectors,
                    CAST(road AS JSON) ->>'class' AS link_type,
                    CAST(road AS JSON) ->>'restrictions' ->> 'speedLimits' ->> 'maxSpeed' AS speed,
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

            # Creating nodes GeoDataFrame
            df_node = pd.read_parquet(output_file_node)
            geo_node = gpd.GeoSeries.from_wkb(df_node.geometry, crs=4326)
            gdf_node = gpd.GeoDataFrame(df_node,geometry=geo_node)

            # Convert the 'speed' column values from JSON strings to Python objects, taking the first element if present
            gdf_link['speed'] = gdf_link['speed'].apply(lambda x: json.loads(x)[0] if x else None)
            
            gdf_node['node_id'] = self.create_node_ids(gdf_node)
            gdf_node['ogc_fid'] = pd.Series(list(range(1, len(gdf_node['node_id']) + 1)))
            gdf_node['is_centroid'] = 0


            # Function to process each row and create a new GeoDataFrame
            def process_row(gdf_link):
                # Extract necessary information from the row
                connectors = gdf_link['connectors']

                # Check if 'Connectors' has more than 2 elements
                if np.size(connectors) >= 2:
                    # Split the DataFrame into multiple rows
                    rows = []
                    for i in range(len(connectors) - 1):
                        new_row = {'a_node': self.nodes[connectors[i]]['node_id'], 'b_node': self.nodes[connectors[i + 1]]['node_id'], 'link_type': gdf_link['link_type'], 
                                   'speed': gdf_link['speed'], 'ovm_id': gdf_link['ovm_id'], 'geometry': gdf_link['geometry']}
                        rows.append(new_row)
                    processed_df = gpd.GeoDataFrame(rows)
                else:
                    raise ValueError("Invalid amount of connectors provided. Must be 2< to be considered a link.")
                return processed_df

            # Iterate over rows using iterrows()
            result_dfs = []
            for index, row in gdf_link.iterrows():
                # Process each row and append the resulting GeoDataFrame to the list
                processed_df = process_row(row)
                result_dfs.append(processed_df)

            # Concatenate the resulting DataFrames into a final GeoDataFrame
            final_result = pd.concat(result_dfs, ignore_index=True)

            # adding neccassary columns for aequilibrea data frame
            final_result['link_id'] = 1
            final_result['ogc_fid'] = pd.Series(list(range(1, len(final_result['link_id']) + 1)))
            final_result['direction'] = 0
            final_result['distance'] = 1
            final_result['name'] = 1
            final_result['travel_time'] = 1
            final_result['capacity'] = 1
            final_result['lanes'] = 1

            # [11:58] Jamie Cook
            def trim_geometry(node_lu, row, position):          
                lat_long_a = node_lu[self.node_ids[row["a_node"][position]]]['coord']
                lat_long_b = node_lu[self.node_ids[row["b_node"][position]]]['coord']
                
                start,end = -1, -1
                for j, coord in enumerate(row.geometry[position].coords):
                    if lat_long_a == coord:
                        start = j
                    if lat_long_b == coord:
                        end = j
                if start < 0 or end < 0:
                    raise RuntimeError("Couldn't find the start end coords in the given linestring")
                return shapely.LineString(row.geometry[position].coords[start:end+1])
 
            for i in range(1, len(final_result['link_id'])):
                final_result['geometry'][i] = trim_geometry(self.nodes, final_result[['a_node','b_node','geometry']], i)    

            mode_codes, not_found_tags = self.modes_per_link_type()
            final_result['modes'] = final_result['link_type'].apply(lambda x: mode_codes.get(x, not_found_tags))


            common_nodes = final_result['a_node'].isin(gdf_node['node_id'])
            # Check if any common nodes exist
            if common_nodes.any():                
                # If common node exist, retrieve the DataFrame of matched rows using boolean indexing
                matched_rows = final_result[common_nodes]

                # Create the 'link_types' and 'modes' columns for the 'gdf_node' DataFrame
                gdf_node['link_types'] = matched_rows['link_type']
                gdf_node['modes'] = matched_rows['modes']
            else:
                # No common nodes found
                raise ValueError("No common nodes.")


            link_order = ['ogc_fid', 'link_id', 'a_node', 'b_node', 'direction', 'distance', 'modes', 'link_type', 'name', 'speed', 'travel_time', 'capacity', 'ovm_id', 'lanes', 'geometry']
            final_result = final_result[link_order]

            final_result.to_parquet(output_file_link)
            g_dataframes.append(final_result)
            self.GeoDataFrame.append(g_dataframes)

            node_order = ['ogc_fid', 'node_id', 'is_centroid', 'modes', 'link_types', 'ovm_id', 'geometry']
            gdf_node = gdf_node[node_order]

            gdf_node.to_parquet(output_file_node)
            g_dataframes.append(gdf_node)
            self.GeoDataFrame.append(g_dataframes)
        return g_dataframes    

    def create_node_ids(self, data_frame):
        '''
        Creates node_ids as well as the self.nodes and self.node_ids dictories
        '''
        node_ids = []
        data_frame['node_id'] = 1
        for i in range(len(data_frame['ovm_id'])):
            node_count = i + self.node_start
            node_ids.append(node_count)
            self.node_ids[node_count] = data_frame['ovm_id'][i]
            self.nodes[data_frame['ovm_id'][i]] = {'node_id': node_count, 'lat': data_frame['geometry'][i].y, 'lon': data_frame['geometry'][i].x, 'coord': (data_frame['geometry'][i].x, data_frame['geometry'][i].y)}
        data_frame['node_id'] = pd.Series(node_ids)
        return data_frame['node_id']

    def modes_per_link_type(self):
        p = Parameters()
        modes = p.parameters["network"]["ovm"]["modes"]
        result = [(key, key[0]) for key in modes.keys()]
        mode_codes = {p[0]: p[1] for p in result}
        type_list = {}
        notfound = ""
        for mode, val in modes.items():
            all_types = val["link_types"]
            md = mode_codes[mode]
            for tp in all_types:
                type_list[tp] = "{}{}".format(type_list.get(tp, ""), md)
            if val["unknown_tags"]:
                notfound += md

        type_list = {k: "".join(set(v)) for k, v in type_list.items()}
        return type_list, "{}".format(notfound)

    def download_test_data(self, l_data_source):
        '''This method only used to seed/bootstrap a local copy of a small test data set'''
        airlie_bbox = [148.7077, -20.2780, 148.7324, -20.2621 ]
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
