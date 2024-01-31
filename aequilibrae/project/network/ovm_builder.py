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

spec = iutil.find_spec("PyQt5")
pyqt = spec is not None
if pyqt:
    from PyQt5.QtCore import pyqtSignal

spec = iutil.find_spec("qgis")
isqgis = spec is not None
if isqgis:
    import qgis 

class OVMBuilder(WorkerThread):
    if pyqt:
        building = pyqtSignal(object)

    def __init__(self, ovm_download: list, project_path: Union[str, Path], logger: logging.Logger = None, node_start=10000, project=None) -> None:
        WorkerThread.__init__(self, None)
        self.project = project or get_active_project()
        self.logger = logger or get_logger()
        self.node_start = node_start
        self.report = []
        self.conn = None
        self.GeoDataFrame = []
        self.nodes = {}
        self.node_ids = {}  
        self.links_gdf = ovm_download[0]
        self.nodes_gdf = ovm_download[1]
        self.__link_types = None  # type: LinkTypes
        self.__model_link_types = []
        self.__model_link_type_ids = []
        self.__link_type_quick_reference = {}
        self.__project_path = Path(project_path)
        self.pth = str(self.__project_path).replace("\\", "/")
        self.insert_qry = """INSERT INTO {} ({}, geometry) VALUES({}, GeomFromText(?, 4326))"""

    def __emit_all(self, *args):
        if pyqt:
            self.building.emit(*args)

    def doWork(self,output_dir: Path):
        self.conn = connect_spatialite(self.pth)
        self.curr = self.conn.cursor()
        self.__worksetup()
        self.formatting(self.links_gdf, self.nodes_gdf, output_dir)
        print(self.pth)
        self.__emit_all(["finished_threaded_procedure", 0])

    def formatting(self, links_gdf: gpd.GeoDataFrame, nodes_gdf: gpd.GeoDataFrame, output_dir: Path):
        g_dataframes = []
        output_dir = Path(output_dir)
        output_file_link = output_dir / f'type=segment' / f'transportation_data_segment.parquet'
        output_file_node = output_dir / f'type=connector' / f'transportation_data_connector.parquet'
      
        links_gdf['name'] = links_gdf['name'].apply(lambda x: json.loads(x)[0]['value'] if x else None)

        nodes_gdf['node_id'] = self.create_node_ids(nodes_gdf)        
        nodes_gdf['ogc_fid'] = pd.Series(list(range(1, len(nodes_gdf) + 1)))
        nodes_gdf['is_centroid'] = 0

        # Iterate over rows using iterrows()
        result_dfs = []
        for index, row in links_gdf.iterrows():
            # Process each row and append the resulting GeoDataFrame to the list
            processed_df = self.split_connectors(row)

            # processed_df = split_speeds(row)
            result_dfs.append(processed_df)

        # Concatenate the resulting DataFrames into a final GeoDataFrame
        final_result = pd.concat((df.dropna(axis=1, how='all') for df in result_dfs), ignore_index=True)

        # adding neccassary columns for aequilibrea data frame
        final_result['link_id'] = pd.Series(list(range(1, len(final_result) + 1)))
        final_result['ogc_fid'] = pd.Series(list(range(1, len(final_result) + 1)))
        final_result['geometry'] = [self.trim_geometry(self.node_ids, row) for e, row in final_result[['a_node','b_node','geometry']].iterrows()]

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

        common_nodes = final_result['a_node'].isin(nodes_gdf['node_id'])
        # Check if any common nodes exist
        if common_nodes.any():                
            # If common node exist, retrieve the DataFrame of matched rows using boolean indexing
            matched_rows = final_result[common_nodes]

            # Create the 'link_types' and 'modes' columns for the 'nodes_gdf' DataFrame
            nodes_gdf['link_types'] = matched_rows['link_type']
            nodes_gdf['modes'] = matched_rows['modes']
            nodes_gdf['ovm_id'] = nodes_gdf['ovm_id']
            nodes_gdf['geometry'] = nodes_gdf['geometry']
        else:
            # No common nodes found
            raise ValueError("No common nodes.")
        fields = self.get_link_fields()
        link_order = fields.copy() + ['geometry']

        for element in link_order:
            if element not in final_result:
                final_result[element] = None
        
        final_result = final_result[link_order]
        final_result.to_parquet(output_file_link)
        g_dataframes.append(final_result)
        self.GeoDataFrame.append(g_dataframes)

        # For goemetry to work in the sql
        final_result = pd.DataFrame(final_result)
        final_result['geometry'] = final_result['geometry'].astype(str)

        node_order = ['ogc_fid', 'node_id', 'is_centroid', 'modes', 'link_types', 'ovm_id', 'geometry']
        nodes_gdf = nodes_gdf[node_order]
        
        nodes_gdf.to_parquet(output_file_node)
        g_dataframes.append(nodes_gdf)
        self.GeoDataFrame.append(g_dataframes)

        table = "links"
        # fields = self.get_link_fields()
        # fields.pop(fields.index('link_id'))
        
        self.__update_table_structure()
        field_names = ",".join(fields)        

        self.logger.info("Adding network nodes")
        self.__emit_all(["text", "Adding network nodes"])

        sql = "insert into nodes(node_id, is_centroid, ovm_id, geometry) Values(?, 0, ?, MakePoint(?,?, 4326))"
        node_df = []
        for node_attributes in nodes_gdf.iterrows():

            node_df.append([node_attributes[1].iloc[1],
                              node_attributes[1].iloc[5],
                              node_attributes[1].iloc[6].coords[0][0],
                              node_attributes[1].iloc[6].coords[0][1]])
        node_df = (
            pd.DataFrame(node_df, columns=["A", "B", "C", "D"])
            .drop_duplicates(subset=["C", "D"])
            .to_records(index=False)
        )
        # print(type(node_df))
        # print(len(node_df))
        # print(node_df)
        # print(node_df[0])
        # print(len(node_df))
        self.conn.executemany(sql, node_df)
        self.conn.commit()
        del nodes_gdf

        all_attrs = final_result.values.tolist()

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
        del final_result
        self.curr.close()

    def __worksetup(self):
        self.__link_types = self.project.network.link_types
        lts = self.__link_types.all_types()
        for lt_id, lt in lts.items():
            self.__model_link_types.append(lt.link_type)
            self.__model_link_type_ids.append(lt_id)

    def __repair_link_type(self, link_type: str) -> str:
        original_link_type = link_type
        link_type = "".join([x for x in link_type if x in string.ascii_letters + "_"]).lower()
        split = link_type.split("_")
        for i, piece in enumerate(split[1:]):
            if piece in ["link", "segment", "stretch"]:
                link_type = "_".join(split[0 : i + 1])

        if len(link_type) == 0:
            link_type = "empty"

        if len(self.__model_link_type_ids) >= 51 and link_type not in self.__model_link_types:
            link_type = "aggregate_link_type"

        if link_type in self.__model_link_types:
            lt = self.__link_types.get_by_name(link_type)
            if original_link_type not in lt.description:
                lt.description += f", {original_link_type}"
                lt.save()
            self.__link_type_quick_reference[original_link_type.lower()] = link_type
            return link_type

        letter = link_type[0]
        if letter in self.__model_link_type_ids:
            letter = letter.upper()
            if letter in self.__model_link_type_ids:
                for letter in string.ascii_letters:
                    if letter not in self.__model_link_type_ids:
                        break
        letter
        lt = self.__link_types.new(letter)
        lt.link_type = link_type
        lt.description = f"Link types from Overture Maps: {original_link_type}"
        lt.save()
        self.__model_link_types.append(link_type)
        self.__model_link_type_ids.append(letter)
        self.__link_type_quick_reference[original_link_type.lower()] = link_type
        return link_type

    def create_node_ids(self, data_frame: gpd.GeoDataFrame) -> pd.Series:
        '''
        Creates node_ids as well as the self.nodes and self.node_ids dictories
        '''
        node_ids = []
        data_frame['node_id'] = 1
        for i in range(len(data_frame)):
            node_count = i + self.node_start
            node_ids.append(node_count)
            self.node_ids[node_count] = {'ovm_id': data_frame['ovm_id'][i], 'lat': data_frame['geometry'][i].y, 'lon': data_frame['geometry'][i].x, 'coord': (data_frame['geometry'][i].x, data_frame['geometry'][i].y)}
            self.nodes[data_frame['ovm_id'][i]] = {'lat': data_frame['geometry'][i].y, 'lon': data_frame['geometry'][i].x, 'coord': (data_frame['geometry'][i].x, data_frame['geometry'][i].y), 'node_id': node_count}
        data_frame['node_id'] = pd.Series(node_ids)
        return data_frame['node_id']

    def modes_per_link_type(self):
        p = Parameters(self.project)
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
                
    def trim_geometry(self, node_lu: dict, row: dict) -> shapely.LineString:
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
    
    # Function to process each row and create a new GeoDataFrame
    def split_connectors(self, row: dict) -> gpd.GeoDataFrame:
        # Extract necessary information from the row     
        connectors = row['connectors']
        
        direction_dictionary = self.get_direction(row['direction'])
        # Check if 'Connectors' has more than 2 elements
        if np.size(connectors) >= 2:
            # Split the DataFrame into multiple rows
            rows = []

            for i in range(len(connectors) - 1):
                new_row = {'a_node': self.nodes[connectors[i]]['node_id'], 
                           'b_node': self.nodes[connectors[i + 1]]['node_id'], 
                           'direction': direction_dictionary['direction'], 
                           'link_type': self.__link_type_quick_reference.get(row["link_type"].lower(), self.__repair_link_type(row["link_type"])),
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

    def get_speed(self, speed_row) -> float:
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
            ltype = self.get_link_field_type(field).upper()
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
    def get_link_field_type(field_name: list):
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
    def get_direction(directions_list: list):
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