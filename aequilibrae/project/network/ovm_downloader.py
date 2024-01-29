import json
import importlib.util as iutil
import sqlite3
import logging
from pathlib import Path

from aequilibrae.context import get_active_project
from aequilibrae.parameters import Parameters
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

    def __init__(self, modes, project_path: Union[str, Path], logger: logging.Logger = None, node_start=10000, project=None) -> None:
        WorkerThread.__init__(self, None)
        self.project = project or get_active_project()
        self.logger = logger or get_logger()
        self.filter = self.get_ovm_filter(modes)
        self.node_start = node_start
        self.report = []
        self.conn = None
        self.GeoDataFrame = []
        self.nodes = {}
        self.node_ids = {}
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
            
        output_file_link = output_dir / f'theme=transportation' / f'type=segment' / f'transportation_data_segment.parquet'
        output_file_node = output_dir / f'theme=transportation' / f'type=connector' / f'transportation_data_connector.parquet'
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
        # all_nodes =[]
        # for node in gdf_node['ovm_id']:
        #     all_nodes.append(node)
        # print(all_nodes)
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
        
        final_result['travel_time_ab'] = None
        final_result['capacity_ab'] = None

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
        fields = self.get_link_fields()
        link_order = fields.copy() + ['geometry']

        for element in link_order:
            if element not in final_result:
                final_result[element] = None
        

        # link_order = ['ogc_fid', 'link_id', 'a_node', 'b_node', 'direction', 'distance', 'modes', 'link_type', 'name', 'speed_ab', 'lanes_ab', 'lanes_ba', 'travel_time_ab', 'capacity_ab', 'ovm_id', 'geometry']
        final_result = final_result[link_order]
        

        final_result.to_parquet(output_file_link)
        g_dataframes.append(final_result)
        self.GeoDataFrame.append(g_dataframes)

        final_result['geometry'] = final_result['geometry'].astype(str)

        node_order = ['ogc_fid', 'node_id', 'is_centroid', 'modes', 'link_types', 'ovm_id', 'geometry']
        gdf_node = gdf_node[node_order]

        gdf_node.to_parquet(output_file_node)
        g_dataframes.append(gdf_node)
        self.GeoDataFrame.append(g_dataframes)
        # print(self.pth)

        self.conn = connect_spatialite(self.pth)
        self.curr = self.conn.cursor()

        table = "links"
        # fields = self.get_link_fields()
        # fields.pop(fields.index('link_id'))
        
        self.__update_table_structure()
        field_names = ",".join(fields)        

        self.logger.info("Adding network nodes")
        self.__emit_all(["text", "Adding network nodes"])

        sql = "insert into nodes(node_id, is_centroid, ovm_id, geometry) Values(?, 0, ?, MakePoint(?,?, 4326))"
        node_df = []
        for node_attributes in gdf_node.iterrows():

            node_df.append([node_attributes[1].iloc[1],
                              node_attributes[1].iloc[5],
                              node_attributes[1].iloc[6].coords[0][0],
                              node_attributes[1].iloc[6].coords[0][1]])
        node_df = (
            pd.DataFrame(node_df, columns=["A", "B", "C", "D"])
            .drop_duplicates(subset=["C", "D"])
            .to_records(index=False)
        )
        self.conn.executemany(sql, node_df)
        self.conn.commit()

        all_attrs = final_result.head().values.tolist()

        sql = self.insert_qry.format(table, field_names, ",".join(["?"] * (len(link_order) - 1)))
        self.logger.info("Adding network links")
        self.__emit_all(["text", "Adding network links"])
        try:
            self.curr.executemany(sql, all_attrs)
        except Exception as e:
            self.logger.error("error when inserting link {}. Error {}".format(all_attrs[0], e.args))
            self.logger.error(sql)
            raise e

        self.conn.commit()
        self.curr.close()

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
                new_row = {'a_node': self.nodes[connectors[i]]['node_id'], 
                           'b_node': self.nodes[connectors[i + 1]]['node_id'], 
                           'direction': direction_dictionary['direction'], 
                           'link_type': row['link_type'],
                           'name': row['name'], 'speed_ab': self.get_speed(row['speed']), 
                           'ovm_id': row['ovm_id'], 
                           'geometry': row['geometry'],
                           'lanes_ab': direction_dictionary['lanes_ab'], 
                           'lanes_ba': direction_dictionary['lanes_ba']
                           }
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
    
    def __update_table_structure(self):
        curr = self.conn.cursor()
        curr.execute("pragma table_info(Links)")
        structure = curr.fetchall()
        has_fields = [x[1].lower() for x in structure]
        fields = [field.lower() for field in self.get_link_fields()] + ["ovm_id"]
        for field in [f for f in fields if f not in has_fields]:
            print(field)
            ltype = self.get_link_field_type(field).upper()
            print(ltype)
            print()
            curr.execute(f"Alter table Links add column {field} {ltype}")
        self.conn.commit()

    @staticmethod
    def get_link_fields():
        p = Parameters()
        fields = p.parameters["network"]["links"]["fields"]
        owf = [list(x.keys())[0] for x in fields["one-way"]]

        twf1 = ["{}_ab".format(list(x.keys())[0]) for x in fields["two-way"]]
        twf2 = ["{}_ba".format(list(x.keys())[0]) for x in fields["two-way"]]

        return owf + twf1 + twf2 + ["ovm_id"]
    
    @staticmethod
    def get_link_field_type(field_name):
        p = Parameters()
        fields = p.parameters["network"]["links"]["fields"]

        if field_name[-3:].lower() in ["_ab", "_ba"]:
            field_name = field_name[:-3]
            for tp in fields["two-way"]:
                if field_name in tp:
                    return tp[field_name]["type"]
        else:
            for tp in fields["one-way"]:
                if field_name in tp:
                    return tp[field_name]["type"]
    
    @staticmethod
    def get_direction(directions_list):
        new_list = []
        at_dictionary = {}

        # Dictionary mapping direction strings to numeric values or descriptions
        direction_dict = {'forward': 1, 'backward': -1, 'bothWays': 0,
                            'alternating': 'Travel is one-way and changes between forward and backward constantly', 
                            'reversible': 'Travel is one-way and changes between forward and backward infrequently'}
        
        # Lambda function to check numbers and create a new dictionary
        check_numbers = lambda lst: {
                'direction': 1 if all(x == 1 for x in lst) else -1 if all(x == -1 for x in lst) else 0,
                'lanes_ab': lst.count(1) if 1 in lst else None,
                'lanes_ba': lst.count(-1) if -1 in lst else None
            }

        if directions_list is None:
            new_list = [-1, 1]
        elif directions_list != None:       
            for direct in directions_list:
                if type(direct) == dict:
                    
                    # Extract direction from the dictionary and append to new_list
                    direction = direction_dict[direct['direction']]
                    new_list.append(direction)
                elif type(direct) == list: 
                    a_list =[]
                    at_dictionary[str(direct[0]['at'])] = direct[0]['at'][1] - direct[0]['at'][0]
                    max_key = max(at_dictionary, key=at_dictionary.get)
                    a_list.append(max_key) 

                    # Check if the current list is the one with maximum 'at' range
                    if str(direct[0]['at']) == a_list[-1]:
                        new_list.clear()            
                        for lists in direct[0]['value']:
                            direction = direction_dict[lists['direction']]
                            new_list.append(direction)

        return check_numbers(lst=new_list)