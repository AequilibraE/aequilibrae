import json
import logging
from pathlib import Path

from aequilibrae.parameters import Parameters
from aequilibrae.context import get_logger
import importlib.util as iutil
from aequilibrae.project.network.haversine import haversine
from aequilibrae.utils import WorkerThread

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


    def downloadTransportation(self, bbox, data_source, output_dir):
        data_source = Path(data_source) or DEFAULT_OVM_S3_LOCATION
        output_dir = Path(output_dir)
        g_dataframes = []
        # for t in ['segment','connector']:
            
        output_file_link = output_dir  / f'type=segment' / f'transportation_data_segment.parquet'
        output_file_node = output_dir  / f'type=connector' / f'transportation_data_connector.parquet'
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

        # Creating nodes GeoDataFrame
        df_node = pd.read_parquet(output_file_node)
        geo_node = gpd.GeoSeries.from_wkb(df_node.geometry, crs=4326)
        gdf_node = gpd.GeoDataFrame(df_node,geometry=geo_node)

        # Convert the 'speed' column values from JSON strings to Python objects, taking the first element if present
        # gdf_link['speed'] = gdf_link['speed'].apply(lambda x: json.loads(x)[0] if x else None)
        gdf_link['name'] = gdf_link['name'].apply(lambda x: json.loads(x)[0]['value'] if x else None)
        
        gdf_node['node_id'] = self.create_node_ids(gdf_node)
        gdf_node['ogc_fid'] = pd.Series(list(range(1, len(gdf_node) + 1)))
        gdf_node['is_centroid'] = 0

        # Iterate over rows using iterrows()
        result_dfs = []
        # print('table')
        # print(self.get_speed(gdf_link)['speed'])
        # print()
        for index, row in gdf_link.iterrows():
            # Process each row and append the resulting GeoDataFrame to the list
            processed_df = self.split_connectors(row)

            # processed_df = split_speeds(row)
            result_dfs.append(processed_df)

        # Concatenate the resulting DataFrames into a final GeoDataFrame
        final_result = pd.concat(result_dfs, ignore_index=True)

        # adding neccassary columns for aequilibrea data frame
        final_result['link_id'] = pd.Series(list(range(1, len(final_result) + 1)))
        final_result['ogc_fid'] = pd.Series(list(range(1, len(final_result) + 1)))
        final_result['geometry'] = [self.trim_geometry(self.node_ids, row) for e, row in final_result[['a_node','b_node','geometry']].iterrows()]
        
        final_result['travel_time'] = 1
        final_result['capacity'] = 1

        distance_list = []
        for i in range(0, len(final_result)):
            distance = sum(
                [
                haversine(x[0], x[1], y[0], y[1])
                for x, y in zip(list(final_result['geometry'][i].coords)[1:], list(final_result['geometry'][i].coords)[:-1])  
                ]  
            )
            distance_list.append(distance)
        final_result["distance"] = distance_list
            
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

        link_order = ['ogc_fid', 'link_id', 'connectors', 'a_node', 'b_node', 'direction', 'distance', 'modes', 'link_type', 'road', 'name', 'restrictions', 'speed', 'travel_time', 'capacity', 'ovm_id', 'lanes_ab', 'lanes_ba', 'geometry']
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
        for i in range(len(data_frame)):
            node_count = i + self.node_start
            node_ids.append(node_count)
            self.node_ids[node_count] = {'ovm_id': data_frame['ovm_id'][i], 'lat': data_frame['geometry'][i].y, 'lon': data_frame['geometry'][i].x, 'coord': (data_frame['geometry'][i].x, data_frame['geometry'][i].y)}
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
                
    def trim_geometry(self, node_lu, row):
        lat_long_a = node_lu[row["a_node"]]['coord']
        lat_long_b = node_lu[row["b_node"]]['coord']
        start,end = -1, -1
        for j, coord in enumerate(row.geometry.coords):
            if lat_long_a == coord:
                start = j
            if lat_long_b == coord:
                end = j
        if start < 0 or end < 0:
            raise RuntimeError("Couldn't find the start end coords in the given linestring")
        return shapely.LineString(row.geometry.coords[start:end+1])

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
    
    # Function to process each row and create a new GeoDataFrame
    def split_connectors(self, row):
        # Extract necessary information from the row     
        connectors = row['connectors']
        direction_dictionary = self.get_direction(row['direction'])
        # Check if 'Connectors' has more than 2 elements
        if np.size(connectors) >= 2:
            # Split the DataFrame into multiple rows
            rows = []
            for i in range(len(connectors) - 1):
                # print(self.get_direction(row['direction']))
                new_row = {'connectors': [self.nodes[connectors[ii]]['node_id'] for ii in range(len(connectors))],
                           'a_node': self.nodes[connectors[i]]['node_id'], 
                           'b_node': self.nodes[connectors[i + 1]]['node_id'], 
                           'direction': direction_dictionary['direction'], 
                           'link_type': row['link_type'], 
                           'road': row['road'], 
                           'name': row['name'], 'speed': self.get_speed(row['speed']), 
                           'ovm_id': row['ovm_id'], 
                           'geometry': row['geometry'], 
                           'lanes_ab': direction_dictionary['lanes_ab'], 
                           'lanes_ba': direction_dictionary['lanes_ba'], 
                           'restrictions': row['speed']}
                rows.append(new_row)
            processed_df = gpd.GeoDataFrame(rows)
        else:
            raise ValueError("Invalid amount of connectors provided. Must be 2< to be considered a link.")
        return processed_df

    def get_speed(self, speed_row):
        """
        This function returns the speed of a road, if they have multiple speeds listed it will total the speeds listed by the proportions of the road they makeup.
        """
        if speed_row == None:
            adjusted_speed = speed_row
        else:
            speed = json.loads(speed_row)
            if type(speed) == dict:
                adjusted_speed = speed['maxSpeed'][0]
            elif type(speed) == list and len(speed) >= 1:
                # Extract the 'at' list from each dictionary
                # eg [[0.0, 0.064320774], [0.064320774, 1.0]]
                at_values_list = [entry['at'] for entry in speed]

                # Calculate differences between consecutive numbers in each 'at' list. This list iterates through each 'at' 
                # list in at_values_list and calculates the difference between consecutive elements using (at[i + 1] - at[i]).
                # The result is a flat list of differences for all 'at' lists.
                # eg [0.064320774, 0.935679226]
                differences = [diff for at in at_values_list for diff in (at[i + 1] - at[i] for i in range(len(at) - 1))]
                
                new_list = []
                for element in differences:
                    # Find the index of the value in the differences list
                    index_d = differences.index(element)

                    # Access the corresponding entry in the original 'data' list to access the 'maxSpeed' value
                    speed_segment = speed[index_d]['maxSpeed'][0] * element
                    new_list.append(speed_segment)
                
                adjusted_speed = round(sum(new_list),2)
        return adjusted_speed
    
    @staticmethod
    def get_direction(directions_list):
        new_diction = {}
        new_list = []
        at_dictionary = {}

        direction_dict = {'forward': 1, 'backward': -1, 'bothWays': 0,
                            'alternating': 'Travel is one-way and changes between forward and backward constantly', 
                            'reversible': 'Travel is one-way and changes between forward and backward infrequently'}
        check_numbers = lambda lst: 1 if all(x == 1 for x in lst) else -1 if all(x == -1 for x in lst) else 0       
        new_diction = lambda new_list: {'direction': check_numbers(new_list),
                                'lanes_ab': new_list.count(1) if 1 in new_list else None,
                                'lanes_ba': new_list.count(-1) if -1 in new_list else None}

        if directions_list is None:
            new_list = [-1, 1]
        elif directions_list != None:
           
            for direct in directions_list:
                # print(type(direct))
                if type(direct) == dict:
                    direction = direction_dict[direct['direction']]
                    new_list.append(direction)
                elif type(direct) == list:
                    print(direct)
                    new_list = []
                    for lists in direct[1]['value']:
                        direction = direction_dict[lists['direction']]
                        new_list.append(direction)
                        print(new_list)
                        print()
                        for i in range(len(direct)-1):
                            print(new_list)
                            at_dictionary[str(direct[i]['at'])] = new_diction(new_list=new_list)
                    for i in at_dictionary.keys():
                        return at_dictionary[i]

        # new_diction = {'direction': check_numbers(new_list), 
        #                 'lanes_ab': new_list.count(1) if 1 in new_list else None, 
        #                 'lanes_ba': new_list.count(-1) if -1 in new_list else None}
        
        print(f'at: {at_dictionary}')
        print(at_dictionary.keys())
        if at_dictionary == {}:
            print('empty')
        else:
            for i in at_dictionary.keys():
                print(at_dictionary[i])
        return new_diction(new_list=new_list)