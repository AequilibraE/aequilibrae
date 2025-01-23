import gc

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod
from shapely.geometry import Polygon, Point
from shapely.ops import split

from aequilibrae.context import get_active_project
from aequilibrae.utils.aeq_signal import SIGNAL
from aequilibrae.utils.interface.worker_thread import WorkerThread
from aequilibrae.utils.spatialite_utils import connect_spatialite
from aequilibrae.utils.db_utils import commit_and_close

MIN_SEGMENT_LENGTH = 0.01
POINT_PRECISION = 7


# TODO: update link_types and nodes, dump data into project_database.
# Export restrictions and similar
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

        self.node_df = data["nodes"]
        self.node_df.loc[:, "node_id"] = np.arange(self.node_start, self.node_start + self.node_df.shape[0])
        gc.collect()

        self.links_df = data["links"]
        self.links_df["idx"] = self.links_df.index + 1

        self.__geod = Geod(ellps="WGS84")
        self.segment_data = []
        self.split_points = []

    def doWork(self):

        with commit_and_close(connect_spatialite(self.path)) as conn:
            self.__update_table_structure(conn)
            self.get_segment_data()
            self.__do_clean(conn)

        self.signal.emit(["finished", 0])

    def __get_all_break_points(self, segment_id):
        segment_points = set(self.sub_segments.loc[self.sub_segments["idx"] == segment_id, "ref"])
        restriction_points = set(
            self.restrictions.loc[self.restrictions["idx"] == segment_id, ["ref_from", "ref_to"]].to_numpy().flatten()
        )
        speed_points = set(self.speed.loc[self.speed["idx"] == segment_id, ["ref_from", "ref_to"]].to_numpy().flatten())

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
                start = max(row_a["ref_from"], row_b["ref_from"])
                end = min(row_a["ref_to"], row_b["ref_to"])

                if start < end:
                    new_row = {**row_a.to_dict(), **row_b.to_dict()}
                    if start not in self.__linear_references:
                        start = min(self.__linear_references, key=lambda x: abs(x - start))
                    if end not in self.__linear_references:
                        end = min(self.__linear_references, key=lambda x: abs(x - end))
                    new_row.update({"ref_from": start, "ref_to": end})
                    result = pd.concat([result, pd.DataFrame([new_row])], ignore_index=True)

        for _, row_result in result.iterrows():
            idx = row_result["idx"]
            mask_c = df_c["idx"] == idx
            df_c_filtered = df_c[mask_c]

            for _, row_c in df_c_filtered.iterrows():
                start = max(row_result["ref_from"], row_c["ref_from"])
                end = min(row_result["ref_to"], row_c["ref_to"])

                if start < end:
                    new_row = {**row_result.to_dict(), **row_c.to_dict()}
                    new_row.update({"ref_from": start, "ref_to": end})
                    final_result = pd.concat([final_result, pd.DataFrame([new_row])], ignore_index=True)

        return final_result.sort_values(["idx", "ref_from"]).reset_index(drop=True)

    def __get_access_restrictions(self):
        # Filter by restrictions
        restrictions = self.links_df[self.links_df["access_restrictions"].notna()].explode("access_restrictions")
        restrictions = pd.json_normalize(restrictions["access_restrictions"].tolist()).set_index(restrictions.idx)
        restrictions.reset_index(inplace=True)

        # Get vehicle information if exists
        vehicles = restrictions[restrictions["when.vehicle"].notna()].explode("when.vehicle")
        vehicles = pd.json_normalize(vehicles["when.vehicle"].tolist()).set_index(vehicles.index)

        # Rename columns
        c = ["idx", "access_type", "between", "during", "heading", "using", "recognized", "mode"]
        restrictions = restrictions.join(vehicles).fillna(np.nan).reset_index(drop=True)
        restrictions.drop(["when", "when.vehicle"], axis=1, inplace=True)
        if vehicles.shape[1] > 0:
            c.extend(["vehicle_dimension", "vehicle_comparison", "vehicle_value", "vehicle_value_unit"])
        restrictions.columns = c

        # Get references from/to restrictions
        restrictions[["ref_from", "ref_to"]] = restrictions["between"].apply(pd.Series)
        restrictions["ref_from"] = restrictions["ref_from"].fillna(0.0)
        restrictions["ref_to"] = restrictions["ref_to"].fillna(1.0)
        restrictions.drop("between", axis=1, inplace=True)

        
        private_segments = restrictions[restrictions["recognized"].fillna("").str.join("|").str.contains("as_private")]
        private_segments = private_segments.index.tolist()

        self.links_df = self.links_df[~self.links_df["idx"].isin(private_segments)]
        
        print(f"Removing {len(private_segments)} private links")

        return restrictions[~restrictions["idx"].isin(private_segments)]

    def __get_speed_limits(self):
        # Filter by segments with speed limits
        speed = self.links_df[self.links_df["speed_limits"].notna()].explode("speed_limits")
        speed = pd.json_normalize(speed["speed_limits"].tolist()).set_index([speed.idx])
        speed = speed.reset_index().drop("when", axis=1)

        # Rename columns
        c = ["idx", "min_speed", "variable_max_speed", "between", "max_speed_value", "max_speed_unit"]
        if speed.shape[1] > 6:
            c.extend(
                [
                    "speed_during",
                    "speed_when_heading",
                    "speed_when_using",
                    "speed_when_recognized",
                    "speed_when_mode",
                    "speed_when_vehicle",
                ]
            )
        speed.columns = c

        # Get references from/to speed limits
        speed[["ref_from", "ref_to"]] = speed["between"].apply(pd.Series)
        speed["ref_from"] = speed["ref_from"].fillna(0.0)
        speed["ref_to"] = speed["ref_to"].fillna(1.0)
        speed.drop("between", axis=1, inplace=True)
        return speed

    def __get_sub_segments(self):
        sub_segments = self.links_df[["idx", "connectors"]].explode("connectors")
        sub_segments = pd.json_normalize(sub_segments["connectors"].tolist()).set_index([sub_segments.idx])
        sub_segments = sub_segments.reset_index().rename(columns={"connector_id": "connector", "at": "ref"})

        sub_segments = sub_segments.merge(self.node_df, on="connector", how="right")
        return gpd.GeoDataFrame(sub_segments, geometry=sub_segments.geometry)

    def __find_reference_geometry(self):
        """Contains code pieces borrowed
        from https://github.com/OvertureMaps/transportation-splitter/blob/main/transportation_splitter.py"""

        self.restrictions = self.__get_access_restrictions()
        self.speed = self.__get_speed_limits()

        self.links_df.set_index("idx", inplace=True)

        counter = 1

        l = [np.nan for i in range(self.restrictions.shape[1])]
        l[-2:] = [0.0, 1.0]

        s = [np.nan for i in range(self.speed.shape[1])]
        s[-2:] = [0.0, 1.0]

        for seg in self.links_df.index:
            if seg % 1000 == 0:
                print(f"Breaking links ---> {seg} / {self.links_df.shape[0]}")
    
            result_data = []

            geometry = self.links_df.loc[seg].geometry

            # Parte 1 - Retornar todos os break points
            connector_reference, self.__linear_references = self.__get_all_break_points(seg)
            segment_length = self.__get_length(geometry)

            # Vamos ajustar as referências
            remove = []
            for idx, lr in enumerate(self.__linear_references[:-1]):
                if (self.__linear_references[idx + 1] - lr) < MIN_SEGMENT_LENGTH:
                    if lr in connector_reference:
                        remove.append(self.__linear_references[idx + 1])
                    else:
                        remove.append(lr)

            # Remove duplicated connectors
            for e in list(set(remove)):
                self.__linear_references.remove(e)

            coords = np.array(geometry.coords)

            # Vamos procurar o ponto ao qual correspondem as frações de segmento que são diferentes de 0 ou 1.
            # Essas obviamente correspondem aos conectores já existentes.
            for idx, lr in enumerate(self.__linear_references[:-1]):

                target_length = lr * segment_length
                coord_idx = 0
                for (lon1, lat1), (lon2, lat2) in zip(coords[:-1], coords[1:]):
                    forward_az, _, subsegment_length = self.__geod.inv(
                        lon1, lat1, lon2, lat2, return_back_azimuth=False
                    )
                    if round(target_length - subsegment_length, 6) <= 0:
                        break
                    target_length -= subsegment_length
                    coord_idx += 1

                split_lon, split_lat, _ = self.__geod.fwd(
                    lon1, lat1, forward_az, target_length, return_back_azimuth=False
                )
                point_geometry = self.__round_point(Point(split_lon, split_lat), POINT_PRECISION)

                if lr not in connector_reference:
                    self.split_points.append(
                        {"idx": seg, "ref": lr, "connector": f"connector_{counter}", "geometry": point_geometry}
                    )
                    counter += 1

                result_data.append({"idx": seg, "ref_from": lr, "ref_to": self.__linear_references[idx + 1]})

            result_data = pd.DataFrame(result_data)
            rest = self.restrictions[self.restrictions["idx"] == seg].copy()
            if rest.shape[0] == 0:
                l[0] = seg
                rest.loc[0] = l

            sl = self.speed[self.speed["idx"] == seg].copy()
            if sl.shape[0] == 0:
                s[0] = seg
                sl.loc[0] = s

            df = self.merge_dataframes(rest, sl, result_data)
            self.segment_data.append(df)

    def get_segment_data(self):
        self.sub_segments = self.__get_sub_segments()

        self.__find_reference_geometry()

        split_points = pd.DataFrame(self.split_points)

        sub_segments = pd.concat([self.sub_segments, split_points], ignore_index=True).sort_values(["idx", "ref"])
        sub_segments.reset_index(drop=True, inplace=True)

        segment_data = pd.concat(self.segment_data)
        segment_data = segment_data.merge(
            sub_segments[["idx", "ref", "connector"]],
            left_on=["idx", "ref_from"],
            right_on=["idx", "ref"],
            how="left",
        )
        segment_data = segment_data.merge(
            sub_segments[["idx", "ref", "connector"]],
            left_on=["idx", "ref_to"],
            right_on=["idx", "ref"],
            how="left",
        )
        segment_data.drop(columns=["ref_x", "ref_y"], inplace=True)
        segment_data.rename(columns={"connector_x": "connector_from", "connector_y": "connector_to"}, inplace=True)
        segment_data.insert(segment_data.shape[1], "split_geom", None)

        sub_segments = sub_segments.drop_duplicates("connector").set_index("connector")

        for idx, row in segment_data.iterrows():
            if idx % 1000 == 0:
                print(f"Building segments ---> {idx} / {segment_data.shape[0]}")

            res = split(self.links_df.loc[row["idx"]].geometry, sub_segments.loc[row["connector_to"]].geometry)
            res = split([geom for geom in res.geoms][0], sub_segments.loc[row["connector_from"]].geometry)

            geo = [geom for geom in res.geoms]
            if len(geo) > 1:
                segment_data.at[idx, "split_geom"] = geo[1]
            else:
                segment_data.at[idx, "split_geom"] = geo[0]

        self.data = segment_data

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
