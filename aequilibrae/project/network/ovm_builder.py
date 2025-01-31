import gc
import string
from typing import Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod
from shapely.geometry import Polygon, Point
from shapely.ops import split

from aequilibrae.context import get_active_project
from aequilibrae.parameters import Parameters
from aequilibrae.project.project_creation import remove_triggers, add_triggers
from aequilibrae.utils.aeq_signal import SIGNAL
from aequilibrae.utils.db_utils import commit_and_close
from aequilibrae.utils.interface.worker_thread import WorkerThread
from aequilibrae.utils.spatialite_utils import connect_spatialite
from aequilibrae.project.network.osm.model_area_gridding import geometry_grid

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
        self.path = self.project.path_to_file
        self.node_start = 10_000
        self.model_area = geometry_grid(model_area, 4326)
        self.report = []
        self.clean = clean

        self.node_df = data["nodes"]
        self.node_df.loc[:, "node_id"] = np.arange(self.node_start, self.node_start + self.node_df.shape[0])
        self.node_df = self.node_df.rename(columns={"ovm_id": "connector"})
        gc.collect()

        self.links_df = data["links"]
        self.links_df["idx"] = self.links_df.index + 1
        self.links_df["subclass"] = self.links_df["subclass"].fillna(self.links_df["class"])

        self.__geod = Geod(ellps="WGS84")
        self.segment_data = []
        self.split_points = []
        self.__all_ltp = {}

    def doWork(self):

        self._restrictions = self.access_restrictions()
        self._speed = self.speed_limits()

        self.get_nodes_data()
        self.find_break_point_coord()
        self.split_segments()

        with commit_and_close(connect_spatialite(self.path)) as conn:
            self.__update_table_structure(conn)

            self.importing_network(conn)

            self.logger.info("Cleaning things up")
            conn.execute(
                "DELETE FROM nodes WHERE node_id NOT IN (SELECT a_node FROM links union all SELECT b_node FROM links)"
            )
            conn.commit()
            self.__do_clean(conn)

        self.signal.emit(["finished", 0])

    def access_restrictions(self):
        restrictions = self.links_df[self.links_df["access_restrictions"].notna()].explode("access_restrictions")
        restrictions = pd.json_normalize(restrictions["access_restrictions"].tolist()).set_index(restrictions.idx)
        restrictions.reset_index(inplace=True)

        cols = [x.split(".") for x in restrictions.columns]
        cols = [x[0] if len(x) == 1 else x[1] for x in cols]

        restrictions.columns = cols

        # Remove private links
        filt = restrictions[restrictions["recognized"].fillna("").str.join("|").str.contains("as_private")]
        filt = filt["idx"].tolist()

        restrictions = restrictions[~restrictions["idx"].isin(filt)]
        self.links_df = self.links_df[~self.links_df["idx"].isin(filt)]

        restrictions["mode"] = (
                restrictions["mode"]
                .str.join("|")
                .str.replace("foot", "w")
                .str.replace("bicycle", "b")
                .str.replace("bus", "t")
                .str.replace("car", "c")
                .str.replace("motor_vehicle", "ct")
            )
        
        restrictions[["ref_from", "ref_to"]] = restrictions["between"].apply(pd.Series)
        restrictions["ref_from"] = restrictions["ref_from"].fillna(0.0)
        restrictions["ref_to"] = restrictions["ref_to"].fillna(1.0)

        head = {"backward": 1, "forward": -1}

        restricts = {}
        for _, row in restrictions.iterrows():
            key = (row["idx"], row["ref_from"], row["ref_to"])
            if key not in restricts:
                restricts[key] = {"direction": 0, "modes": ""}

            if row["heading"] in ["backward", "forward"]:
                if not restricts[key]["direction"]:
                    restricts[key]["direction"] = head[row["heading"]]

            m = "bctw"
            if isinstance(row["mode"], str):
                for mode in row["mode"].split("|"):
                    if mode in ["motorcycle", "hov", "hgv", "emergency"]:
                        restricts[key]["modes"] += "h"
                        continue
                    if row["access_type"] in ["allowed", "designated"]:
                        restricts[key]["modes"] += mode
                    elif row["access_type"] in ["denied"]:
                        if len(restricts[key]["modes"]) > 0:
                            restricts[key]["modes"] = restricts[key]["modes"].replace(mode, "")
                        else:
                            restricts[key]["modes"] = m.replace(mode, "")

        rests = pd.DataFrame(restricts).transpose().reset_index(names=["idx", "ref_from", "ref_to"])
        rests = rests[(rests["direction"].notna()) | (rests["modes"].notna())]
        rests.loc[rests["modes"] == "", "modes"] = "bctw"
        rests["modes"] = rests["modes"].str.replace("h", "")
        rests.loc[rests["modes"] == "", "modes"] = None
        
        return rests[rests["modes"].notna()]

    def speed_limits(self):
        speed = self.links_df[self.links_df["speed_limits"].notna()].explode("speed_limits")
        speed = pd.json_normalize(speed["speed_limits"].tolist()).set_index([speed.idx])
        speed.reset_index(inplace=True)

        cols = [x.split(".") for x in speed.columns]
        cols = [x[0] if len(x) == 1 else x[1] for x in cols]

        speed.columns = cols

        speed[["ref_from", "ref_to"]] = speed["between"].apply(pd.Series)
        speed["ref_from"] = speed["ref_from"].fillna(0.0)
        speed["ref_to"] = speed["ref_to"].fillna(1.0)

        cols = ["idx", "value", "unit", "ref_from", "ref_to"]
        return speed[cols]

    def get_nodes_data(self):

        sub_segments = self.links_df[["idx", "connectors"]].explode("connectors")
        sub_segments = pd.json_normalize(sub_segments["connectors"].tolist()).set_index([sub_segments.idx])
        sub_segments = sub_segments.reset_index().rename(columns={"connector_id": "connector", "at": "ref"})
        self.__sub_segments = self.node_df.join(sub_segments.set_index("connector"), on="connector")

    def __get_all_break_points(self, segment_id):
        # Obtém os break points e os pontos que são parte do segmento original
        segment_pts = self.__sub_segments[self.__sub_segments["idx"] == segment_id]["ref"].values.flatten()
        restriction_pts = self._restrictions[self._restrictions["idx"] == segment_id][
            ["ref_from", "ref_to"]
        ].values.flatten()
        speed_pts = self._speed[self._speed["idx"] == segment_id][["ref_from", "ref_to"]].values.flatten()

        return sorted(set(segment_pts)), sorted(set(segment_pts) | set(restriction_pts) | set(speed_pts))

    def __get_length(self, line_geometry):
        """Returns segment length"""

        return self.__geod.geometry_length(line_geometry)

    def __round_point(self, point: Point, precision: int) -> Point:
        """Borrowed from https://github.com/OvertureMaps/transportation-splitter"""
        return Point(round(point.x, precision), round(point.y, precision))

    def __merge_dataframes(self, linear_references, df_a, df_b, df_c):

        result = pd.DataFrame()
        final_result = pd.DataFrame()

        for _, row_a in df_a.iterrows():
            for _, row_b in df_b.iterrows():
                start = max(row_a["ref_from"], row_b["ref_from"])
                end = min(row_a["ref_to"], row_b["ref_to"])

                if start < end:
                    new_row = {**row_a.to_dict(), **row_b.to_dict()}
                    if start not in linear_references:
                        start = min(linear_references, key=lambda x: abs(x - start))
                    if end not in linear_references:
                        end = min(linear_references, key=lambda x: abs(x - end))
                    new_row.update({"ref_from": start, "ref_to": end})
                    result = pd.concat([result, pd.DataFrame([new_row])], ignore_index=True)

        for _, row_result in result.iterrows():
            for _, row_c in df_c.iterrows():
                start = max(row_result["ref_from"], row_c["ref_from"])
                end = min(row_result["ref_to"], row_c["ref_to"])

                if start < end:
                    new_row = {**row_result.to_dict(), **row_c.to_dict()}
                    new_row.update({"ref_from": start, "ref_to": end})
                    final_result = pd.concat([final_result, pd.DataFrame([new_row])], ignore_index=True)

        return final_result.sort_values(["idx", "ref_from"])

    def find_break_point_coord(self):

        counter = 1
        max_link = self.links_df["idx"].values.max()

        for e, seg in enumerate(self.links_df.idx):
            if e % 1000 == 0:
                print(f"Finding break points --> {e} / {max_link}")

            result_data = []

            geometry = self.links_df[self.links_df["idx"] == seg].geometry.values[0]

            connector_reference, linear_references = self.__get_all_break_points(seg)
            segment_length = self.__get_length(geometry)

            # Vamos ajustar as referências
            remove = []
            for idx, lr in enumerate(linear_references[:-1]):
                if (linear_references[idx + 1] - lr) < MIN_SEGMENT_LENGTH:
                    if lr in connector_reference:
                        remove.append(linear_references[idx + 1])
                    else:
                        remove.append(lr)

            # Remove um dos novos conectores que tecnicamente é igual aos existentes.
            # É melhor usar o que já está de acordo com a base de dados.
            for e in list(set(remove)):
                linear_references.remove(e)

            coords = np.array(geometry.coords)

            # Vamos procurar o ponto ao qual correspondem as frações de segmento que são diferentes de 0 ou 1.
            # Essas obviamente correspondem aos conectores já existentes.
            for idx, lr in enumerate(linear_references[:-1]):

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

                result_data.append({"idx": seg, "ref_from": lr, "ref_to": linear_references[idx + 1]})

            result_data = pd.DataFrame(result_data)
            rest = self._restrictions[self._restrictions["idx"] == seg].copy()
            if rest.shape[0] == 0:
                rest.loc[0] = [seg, 0.0, 1.0, 0, "bctw"]

            sl = self._speed[self._speed["idx"] == seg].copy()
            if sl.shape[0] == 0:
                sl.loc[0] = [seg, None, None, 0.0, 1.0]

            df = self.__merge_dataframes(linear_references, rest, sl, result_data)
            self.segment_data.append(df)

    def split_segments(self):
        split_points = pd.DataFrame(self.split_points)
        split_points = gpd.GeoDataFrame(split_points, geometry=split_points["geometry"], crs=self.links_df.crs)

        max_node = self.node_df["node_id"].max() + 1
        split_points["node_id"] = np.arange(max_node, max_node + split_points.shape[0])

        self.__sub_segments = pd.concat([self.__sub_segments, split_points], ignore_index=True)
        self.__sub_segments.sort_values(["idx", "ref"], inplace=True)

        self._data = pd.concat(self.segment_data)

        use_cols = ["idx", "ref", "node_id", "connector"]
        self._data = self._data.merge(
            self.__sub_segments[use_cols], left_on=["idx", "ref_from"], right_on=["idx", "ref"], how="left"
        )
        self._data = self._data.merge(self.__sub_segments[use_cols], left_on=["idx", "ref_to"], right_on=["idx", "ref"], how="left")
        self._data = self._data.drop(columns=["ref_x", "ref_y"]).rename(
            columns={
                "node_id_x": "a_node",
                "node_id_y": "b_node",
                "connector_x": "connector_from",
                "connector_y": "connector_to",
            }
        )
        self._data.insert(self._data.shape[1], "split_geom", None)

        self.__sub_segments = self.__sub_segments.drop_duplicates("connector").set_index("connector")

        for idx, row in self._data.iterrows():
            if idx % 1000 == 0:
                print(f"Splitting geometries --> {idx} / {self._data.shape[0]}")

            geometry = self.links_df.loc[self.links_df["idx"] == row["idx"]].geometry.values[0]
            res = split(geometry, self.__sub_segments.loc[row["connector_to"]].geometry)
            res = split(list(res.geoms)[0], self.__sub_segments.loc[row["connector_from"]].geometry)

            geo = list(res.geoms)
            if len(geo) > 1:
                self._data.at[idx, "split_geom"] = geo[1]
            else:
                self._data.at[idx, "split_geom"] = geo[0]

        self._data.loc[(self._data.direction >= 0), "speed_ab"] = self._data["value"]
        self._data.loc[(self._data.direction <= 0), "speed_ba"] = self._data["value"]

        self._data = self._data.merge(self.links_df[["ovm_id", "subclass", "access_restrictions", "speed_limits", "idx"]], on="idx")

        cols = ["ref_from", "ref_to", "value", "unit", "connector_from", "connector_to", "idx"]
        self._data = self._data.drop(cols, axis=1).rename({"split_geom": "geometry", "subclass": "link_type"}, axis=1)

        self._data["link_id"] = self._data.index + 1
        self._data.loc[self._data["access_restrictions"].notna(), "access_restrictions"] = self._data["access_restrictions"].astype(str)
        self._data.loc[self._data["speed_limits"].notna(), "speed_limits"] = self._data["speed_limits"].astype(str)
        self._data["geometry"] = self._data["geometry"].astype(str)
        self._data["link_type"] = self._data["link_type"].fillna("")
        self._data["direction"] = self._data["direction"].fillna(0)

        def normalize_string(s):
            s = ''.join(sorted(set(s)))
            if set(s).issubset({'b', 'c', 't', 'w'}):
                return s
            else:
                return None
        
        self._data.loc[self._data["modes"].notna(), "modes"] = self._data['modes'].apply(normalize_string)
        self._data.loc[self._data["modes"].isna(), "modes"] = "bctw"

    def __define_link_type(self, link_type: str) -> Tuple[str, str]:
        proj_link_types = self.project.network.link_types
        original_link_type = link_type
        link_type = "".join([x for x in link_type if x in string.ascii_letters + "_"]).lower()

        split = link_type.split("_")
        for i, piece in enumerate(split[1:]):
            if piece in ["link", "segment", "stretch"]:
                link_type = "_".join(split[0 : i + 1])

        if len(self.__all_ltp) >= 51:
            link_type = "aggregate_link_type"

        if len(link_type) == 0:
            link_type = "empty"

        if link_type in self.__all_ltp:
            lt = proj_link_types.get_by_name(link_type)
            if original_link_type not in lt.description:
                lt.description += f", {original_link_type}"
                lt.save()
            self.__all_ltp[link_type] = lt.link_type_id
            return [lt.link_type_id, link_type]

        letter = link_type[0]
        if letter in self.__all_ltp.values():
            letter = letter.upper()
            if letter in self.__all_ltp.values():
                for letter in string.ascii_letters:
                    if letter not in self.__all_ltp.values():
                        break
        lt = proj_link_types.new(letter)
        lt.link_type = link_type
        lt.description = f"Link types from Overture Maps: {original_link_type}"
        lt.save()
        self.__all_ltp[link_type] = letter
        return [letter, link_type]

    def importing_network(self, conn):
        # Add nodes
        self.__sub_segments["is_centroid"] = 0
        self.__sub_segments["modes"] = None
        self.__sub_segments["link_types"] = None
        self.__sub_segments["lon"] = self.__sub_segments.geometry.x
        self.__sub_segments["lat"] = self.__sub_segments.geometry.y

        cols = ["node_id", "ovm_id", "is_centroid", "modes", "link_types", "lon", "lat"]
        insert_qry = f"INSERT INTO nodes ({','.join(cols[:-2])}, geometry) VALUES(?,?,?,?,?, MakePoint(?,?, 4326))"

        self.__sub_segments.reset_index(names="ovm_id", inplace=True)

        remove_triggers(conn, self.logger, "network")

        conn.executemany(insert_qry, self.__sub_segments[cols].to_records(index=False))

        del self.node_df
        gc.collect()

        add_triggers(conn, self.logger, "network")

        # Add missing link_types
        all_ltp = pd.read_sql("SELECT link_type, link_type_id FROM link_types", conn)
        self.__all_ltp = dict(all_ltp.to_records(index=False))

        for lt in self._data["link_type"].unique():
            self.__define_link_type(lt)

        # Add links
        self._data.loc[self._data["link_type"] == "", "link_type"] = "empty"

        insert_qry = "INSERT INTO links ({}, geometry) VALUES({}, GeomFromText(?, 4326))"
        cols_no_geo = self._data.columns.tolist()
        cols_no_geo.remove("geometry")
        insert_qry = insert_qry.format(", ".join(cols_no_geo), ", ".join(["?"] * len(cols_no_geo)))

        cols = cols_no_geo + ["geometry"]
        links_df = self._data[cols].to_records(index=False)

        del self.links_df
        gc.collect()

        conn.executemany(insert_qry, links_df)

    ######## TABLE STRUCTURE UPDATING ########
    def __update_table_structure(self, conn):
        links_structure = conn.execute("pragma table_info(Links)").fetchall()
        has_links_fields = [x[1].lower() for x in links_structure]
        fields = ["ovm_id", "access_restrictions", "speed_limits"]
        for field in [f for f in fields if f not in has_links_fields]:
            ltype = self.get_link_field_type(field).upper()
            conn.execute(f"Alter table Links add column {field} {ltype}")

        nodes_structure = conn.execute("pragma table_info(Nodes)").fetchall()
        has_nodes_fields = [x[1].lower() for x in nodes_structure]
        if "ovm_id" not in has_nodes_fields:
            ltype = self.get_link_field_type("ovm_id").upper()
            conn.execute(f"Alter table Nodes add column ovm_id {ltype}")

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

    @staticmethod
    def get_link_field_type(field_name):
        p = Parameters()
        ovm = p.parameters["network"]["ovm"]["fields"]

        return ovm[field_name]["type"]
