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

    def __init__(self, modes, project_path: Union[str, Path], logger: logging.Logger = None):
        WorkerThread.__init__(self, None)
        self.logger = logger or get_logger()
        self.filter = self.get_ovm_filter(modes)
        self.report = []
        self.json = []
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

            # creating links geodataframe
            df_link = pd.read_parquet(output_file_link)
            geo_link = gpd.GeoSeries.from_wkb(df_link.geometry, crs=4326)
            gdf_link = gpd.GeoDataFrame(df_link,geometry=geo_link)

            # creating nodes geodataframe
            df_node = pd.read_parquet(output_file_node)
            geo_node = gpd.GeoSeries.from_wkb(df_node.geometry, crs=4326)
            gdf_node = gpd.GeoDataFrame(df_node,geometry=geo_node)

            
            # print(f"node geo: {gdf_node['geometry']}")
            # print(f"node geo: {gdf_link['geometry']}")

            # line = LineString (148.7165748 -20.2730668, 148.7165148 -20.273062, 148.7164585 -20.2730418, 148.7164104 -20.2730078)
            # line_segments = self.segments(line)
            # # resultt = split(gdf_node['geometry']}")
            gdf_link['a_node'] = None
            gdf_link['b_node'] = None
            gdf_link['num_node'] = None
            split_lines = []
            n = 0
            # for connector in gdf_link['connectors']:
            #     # for goems in gdf_link['geometry']:
            #     #     segmentize(ids,2)
            #         # print(count_coordinates(ids))
            #     print(np.size(connector))
            #     print(connector)

                
            #     # for i in range(count_coordinates(ids) - 1):
            #     # if np.size(connector) >2:
                
            #     for i in range(np.size(connector) - 1):
            #         link_segments = self.segments(gdf_link['geometry'][n])
            #         print(f'link_segment: {link_segments}')
            #         for segment in link_segments:
            #             gdf_link['a_node'][n] = connector[i]
            #             gdf_link.loc[n, 'b_node'] = connector[i + 1]
            #             gdf_link.loc[n, 'geometry'] = segment
            #             print(f'segment: {segment} - {n}')
            #             gdf_link.loc[n, 'num_node'] = np.size(connector) 
            #             n += 1
            #     print(f'n: {n}')
            #     print()
            # print(f'n: {n}')
                # n += 1
                    
            # Convert the 'speed' column values from JSON strings to Python objects, taking the first element if present
            gdf_link['speed'] = gdf_link['speed'].apply(lambda x: json.loads(x)[0] if x else None)

            # Function to process each row and create a new GeoDataFrame
            def process_row(gdf_link):
                # Extract necessary information from the row
                ovm_id = gdf_link['ovm_id']
                connectors = gdf_link['connectors']
                link_type = gdf_link['link_type']
                speed = gdf_link['speed']
                geometry = gdf_link['geometry']

                # Check if 'Connectors' has more than 2 elements
                if np.size(connectors) > 2:
                    # Split the DataFrame into multiple rows
                    rows = []
                    for i in range(len(connectors) - 1):
                        new_row = {'a_node': connectors[i], 'b_node': connectors[i + 1], 'link_type': link_type, 'speed': speed, 'ovm_id': ovm_id, 'geometry': geometry}
                        rows.append(new_row)
                    processed_df = gpd.GeoDataFrame(rows)
                elif np.size(connectors) == 2:
                    # For cases where 'Connectors' has 2 elements
                    processed_df = gpd.GeoDataFrame({'a_node': connectors[0], 'b_node': connectors[-1], 'link_type': link_type, 'speed': speed, 'ovm_id': ovm_id, 'geometry': geometry}, index=[0])
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
            final_result['ogc_fid'] = 1
            final_result['link_id'] = 1

          
            final_result['direction'] = 0
            final_result['distance'] = 1


            mode_codes, not_found_tags = self.modes_per_link_type()
            final_result['modes'] = final_result['link_type'].apply(lambda x: mode_codes.get(x, not_found_tags))


            final_result['name'] = 1
            final_result['travel_time'] = 1
            final_result['capacity'] = 1
            final_result['lanes'] = 1


            


            gdf_node['ogc_fid'] = 1
            gdf_node['node_id'] = 1
            gdf_node['is_centroid'] = 0
            gdf_node['modes'] = 1
            gdf_node['link_types'] = 1

            common_nodes = final_result['a_node'].isin(gdf_node['ovm_id'])
            # Check if any common nodes exist
            if common_nodes.any():
                # At least one common node is found
                print("Common nodes exist.")
                
                # You can access the DataFrame of matched rows using boolean indexing
                matched_rows = final_result[common_nodes]
                gdf_node['link_types'] = matched_rows['link_type']
                gdf_node['modes'] = matched_rows['modes']
            else:
                # No common nodes found
                print("No common nodes.")
                # gdf_node['link_types'] = gdf_link['link_type']
                # gdf_node['modes'] = gdf_link['modes']


            link_order = ['ogc_fid', 'link_id', 'a_node', 'b_node', 'direction', 'distance', 'modes', 'link_type', 'name', 'speed', 'travel_time', 'capacity', 'ovm_id', 'lanes', 'geometry']
            final_result = final_result[link_order]

            final_result.to_parquet(output_file_link)
            g_dataframes.append(final_result)
            self.json.extend(g_dataframes)

            node_order = ['ogc_fid', 'node_id', 'is_centroid', 'modes', 'link_types', 'ovm_id', 'geometry']
            gdf_node = gdf_node[node_order]

            gdf_node.to_parquet(output_file_node)
            g_dataframes.append(gdf_node)
            self.json.extend(g_dataframes)
        return g_dataframes    

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
