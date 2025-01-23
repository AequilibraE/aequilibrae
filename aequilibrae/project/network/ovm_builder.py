import gc
import json
import string
from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import split
from pyproj import Geod
from shapely.geometry import Polygon, Point

from aequilibrae.context import get_active_project
from aequilibrae.parameters import Parameters
from aequilibrae.project.network.haversine import haversine
from aequilibrae.project.network.link_types import LinkTypes
from aequilibrae.utils.aeq_signal import SIGNAL
from aequilibrae.utils.interface.worker_thread import WorkerThread
from aequilibrae.utils.spatialite_utils import connect_spatialite
from aequilibrae.utils.db_utils import commit_and_close

MIN_SEGMENT_LENGTH = 0.01
POINT_PRECISION = 7


class OVMBuilder(WorkerThread):
    signal = SIGNAL(object)

    def __init__(self, data, project, model_area: Polygon, clean: bool) -> None:
        WorkerThread.__init__(self, None)

        project.logger.info("Preparing Overture Maps builder")
        self.signal.emit(["set_text", "Preparing Overture Maps builder"])

        self.project = project or get_active_project()
        self.logger = self.project.logger
        self.path = self.project.project_base_path
        self.node_start = 10_000
        self.model_area = model_area
        self.report = []
        self.clean = clean

        self.nodes_df = data["nodes"]
        self.node_df.loc[:, "node_id"] = np.arange(self.node_start, self.node_start + self.node_df.shape[0])
        gc.collect()
        self.links_df = data["links"]

        self.__geod = Geod(ellps="WGS84")

    def doWork(self):
        self.formatting(self.links_gdf, self.nodes_gdf)

        with commit_and_close(connect_spatialite(self.path)) as conn:
            self.__update_table_structure(conn)
            self.__filter_data()

            self.__do_clean(conn)

        self.signal.emit(["finished", 0])

    
    def __filter_data(self):
        # subclass NOT IN ("parking_aisle", "driveway")
        # Other OSM classes are not available in Overture Maps
        self.links_df = self.links_df[~self.links_df["subclass"].isin(["parking_aisle", "driveway"])]

        # access = '["access"!~"private"]'
        rest = self.links_df.explode("access_restrictions")
        rest = rest[~rest["access_restrictions"].isna()].reset_index(names="idx")
        rest = pd.json_normalize(rest["access_restrictions"].tolist()).set_index(rest.idx)

        private_segments = rest[rest["when.recognized"].fillna("").str.join("|").str.contains("as_private")]
        private_segments = private_segments.index.tolist()

        self.links_df.drop(index=private_segments, inplace=True)

    def __get_all_break_points(self, segment_id):
        segment_points = set(self.sub_segments.loc[self.sub_segments['idx'] == segment_id, 'ref'])
        restriction_points = set(self.restrictions.loc[self.restrictions['idx'] == segment_id, ['ref_from', 'ref_to']].to_numpy().flatten())
        speed_points = set(self.speed.loc[self.speed['idx'] == segment_id, ['ref_from', 'ref_to']].to_numpy().flatten())

        segment_break_points = sorted(segment_points)
        all_break_points = sorted(segment_points | restriction_points | speed_points)

        return segment_break_points, all_break_points

    def __get_length(self, line_geometry):
        return self.__geod.geometry_length(line_geometry)
    

    def __round_point(self, point: Point, precision: int) -> Point:
        """Borrowed from https://github.com/OvertureMaps/transportation-splitter/blob/main/transportation_splitter.py"""
        return Point(round(point.x, precision), round(point.y, precision))


    def merge_dataframes(self, df_a, df_b, df_c):
       
        result = pd.DataFrame()
        final_result = pd.DataFrame()
        
        for _, row_a in df_a.iterrows():
            for _, row_b in df_b.iterrows():
                start = max(row_a['ref_from'], row_b['ref_from'])
                end = min(row_a['ref_to'], row_b['ref_to'])
                
                if start < end:
                    new_row = {**row_a.to_dict(), **row_b.to_dict()}
                    if start not in self.__linear_references:
                        start = min(self.__linear_references, key=lambda x: abs(x - start))
                    if end not in self.__linear_references:
                        end = min(self.__linear_references, key=lambda x: abs(x - end))
                    new_row.update({'ref_from': start, 'ref_to': end})
                    result = pd.concat([result, pd.DataFrame([new_row])], ignore_index=True)
        
        for _, row_result in result.iterrows():
            idx = row_result['idx']
            mask_c = (df_c['idx'] == idx)
            df_c_filtered = df_c[mask_c]
            
            for _, row_c in df_c_filtered.iterrows():
                start = max(row_result['ref_from'], row_c['ref_from'])
                end = min(row_result['ref_to'], row_c['ref_to'])
                
                if start < end:
                    new_row = {**row_result.to_dict(), **row_c.to_dict()}
                    new_row.update({'ref_from': start, 'ref_to': end})
                    final_result = pd.concat([final_result, pd.DataFrame([new_row])], ignore_index=True)

        return final_result.sort_values(['idx', 'ref_from']).reset_index(drop=True)

    ######## TABLE STRUCTURE UPDATING ########
    def __update_table_structure(self, conn):
        structure = conn.execute("pragma table_info(Links)").fetchall()
        has_fields = [x[1].lower() for x in structure]
        fields = [field.lower() for field in self.get_link_fields()] + ["ovm_id"]
        for field in [f for f in fields if f not in has_fields]:
            ltype = self.get_link_field_type(field).upper()
            conn.execute(f"Alter table Links add column {field} {ltype}")
        conn.commit()

    
    def __do_clean(self, conn):
        if not self.clean:
            conn.execute("VACUUM;")
            return
        self.logger.info("Cleaning up the network down to the selected area")
        links = gpd.GeoDataFrame.from_postgis("SELECT link_id, asBinary(geometry) AS geom FROM links", conn, crs=4326)
        existing_link_ids = gpd.sjoin(links, self.model_area, how="left").dropna().link_id.to_numpy()
        to_delete = [[x] for x in links[~links.link_id.isin(existing_link_ids)].link_id]
        conn.executemany("DELETE FROM links WHERE link_id = ?", to_delete)
        conn.commit()
        conn.execute("VACUUM;")
