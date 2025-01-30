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
        self.node_df = self.node_df.rename(columns={"ovm_id": "connector"})
        gc.collect()

        self.links_df = data["links"]
        self.links_df["idx"] = self.links_df.index + 1
        self.links_df["subclass"] = self.links_df["subclass"].fillna(self.links_df["class"])

        self.__geod = Geod(ellps="WGS84")
        self.segment_data = []
        self.split_points = []

    def doWork(self):

        self.__restrictions = self.access_restrictions()
        self.__speed = self.speed_limits()

        with commit_and_close(connect_spatialite(self.path)) as conn:
            self.__update_table_structure(conn)
            self.__do_clean(conn)

        self.signal.emit(["finished", 0])

    def access_restrictions(self):
        restrictions = self.links_df[self.links_df["access_restrictions"].notna()].explode("access_restrictions")
        restrictions = pd.json_normalize(restrictions["access_restrictions"].tolist()).set_index(restrictions.link_id)
        restrictions.reset_index(inplace=True)

        cols = [x.split(".") for x in restrictions.columns]
        cols = [x[0] if len(x) == 1 else x[1] for x in cols]

        restrictions.columns = cols

        # Remove private links
        filt = restrictions[restrictions["recognized"].fillna("").str.join("|").str.contains("as_private")]
        filt = filt["idx"].tolist()

        # Remove private segments
        restrictions = restrictions[~restrictions["idx"].isin(filt)]
        self.links_df = self.links_df[~self.links_df["idx"].isin(filt)]

        # Ignore cases when 'during' and 'vehicle' are not none
        restrictions = restrictions[(restrictions["during"].isna()) & (restrictions["vehicle"].isna())]

        # We'll ignore designated road sections
        designated = restrictions[restrictions["access_type"] == "designated"].copy()
        restrictions.drop(designated.index, inplace=True)

        # We'll focus only on deniability.
        allowed = restrictions[restrictions["access_type"] == "allowed"].copy()
        restrictions.drop(allowed.index, inplace=True)

        dupes = restrictions[
            (restrictions["access_type"].notna()) & (restrictions["heading"].isna()) & (restrictions["mode"].isna())
        ].copy()
        restrictions.drop(dupes.index, inplace=True)

        # Keep selected modes, only
        restrictions["mode"] = (
            restrictions["mode"]
            .str.join("|")
            .str.replace("foot", "w")
            .str.replace("bicycle", "b")
            .str.replace("bus", "t")
            .str.replace("car", "c")
            .str.replace("motor_vehicle", "ct")
            .str.replace("|", "")
        )

        pat = "|".join(["motorcycle", "hgv", "hov", "emergency"])

        other_modes = restrictions[restrictions["mode"].str.contains(pat, case=False, na=False)].copy()
        restrictions.drop(other_modes.index, inplace=True)

        # We remove mode specific restrictions
        mode_specific = restrictions[(restrictions["heading"].notna()) & (restrictions["mode"].notna())].copy()
        restrictions.drop(mode_specific.index, inplace=True)

        restrictions[["ref_from", "ref_to"]] = restrictions["between"].apply(pd.Series)
        restrictions["ref_from"] = restrictions["ref_from"].fillna(0.0)
        restrictions["ref_to"] = restrictions["ref_to"].fillna(1.0)

        link_ids = restrictions.link_id.unique()
        assemble = pd.DataFrame([("bctw", 0) for i in link_ids], columns=["modes", "direction"], index=link_ids)

        for idx, row in restrictions.iterrows():
            if row["heading"] == "backward":
                assemble.at[row["idx"], "direction"] = 1
            elif row["heading"] == "forward":
                assemble.at[row["idx"], "direction"] = -1

            if row["mode"] is not None:
                for m in row["mode"]:
                    assemble.at[row["idx"], "modes"] = assemble.loc[row["idx"], "modes"].replace(m, "")

        mode_direction = restrictions.join(assemble, on="idx")

        cols = ["idx", "ref_from", "ref_to", "modes", "direction"]
        mode_direction = mode_direction[cols]

        # Remove duplicated direction and mode references for the same link
        mode_direction.drop_duplicates(["idx", "ref_from", "ref_to", "modes", "direction"], inplace=True)

        return mode_direction

    def speed_limits(self):

        speed = self.links_df[self.links_df["speed_limits"].notna()].explode("speed_limits")
        speed = pd.json_normalize(speed["speed_limits"].tolist()).set_index([speed.link_id])
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
        sub_segments = pd.json_normalize(sub_segments["connectors"].tolist()).set_index([sub_segments.link_id])
        sub_segments = sub_segments.reset_index().rename(columns={"connector_id": "connector", "at": "ref"})
        self.__sub_segments = self.node_df.join(sub_segments.set_index("connector"), on="connector")

    def get_all_break_points(self, segment_id):
        # Obtém os break points e os pontos que são parte do segmento original
        segment_pts = self.__sub_segments[self.__sub_segments["idx"] == segment_id]["ref"].values.flatten()
        restriction_pts = self.__restrictions[self.__restrictions["idx"] == segment_id][
            ["ref_from", "ref_to"]
        ].values.flatten()
        speed_pts = self.__speed[self.__speed["idx"] == segment_id][["ref_from", "ref_to"]].values.flatten()

        return sorted(set(segment_pts)), sorted(set(segment_pts) | set(restriction_pts) | set(speed_pts))

    def get_length(self, line_geometry):
        """Returns segment length"""

        return self.__geod.geometry_length(line_geometry)

    def round_point(self, point: Point, precision: int) -> Point:
        """Borrowed from https://github.com/OvertureMaps/transportation-splitter"""
        return Point(round(point.x, precision), round(point.y, precision))

    def merge_dataframes(linear_references, df_a, df_b, df_c):

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

        return final_result.sort_values(["link_id", "ref_from"])

    def find_break_point_coord(self):

        counter = 1
        split_points = []
        all_data = []
        max_link = self.links_df["idx"].values.max()

        for e, seg in enumerate(self.links_df.idx):
            if e % 1000 == 0:
                print(f"Finding break points --> {e} / {max_link}")

            result_data = []

            geometry = self.links_df[self.links_df["idx"] == seg].geometry.values[0]

            connector_reference, linear_references = self.get_all_break_points(seg)
            segment_length = self.get_length(geometry)

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
                point_geometry = self.round_point(Point(split_lon, split_lat), POINT_PRECISION)

                if lr not in connector_reference:
                    split_points.append(
                        {"idx": seg, "ref": lr, "connector": f"connector_{counter}", "geometry": point_geometry}
                    )
                    counter += 1

                result_data.append({"idx": seg, "ref_from": lr, "ref_to": linear_references[idx + 1]})

            result_data = pd.DataFrame(result_data)
            rest = self.__restrictions[self.__restrictions["idx"] == seg].copy()
            if rest.shape[0] == 0:
                rest.loc[0] = [seg, 0.0, 1.0, "bctw", 0]

            sl = self.__speed[self.__speed["idx"] == seg].copy()
            if sl.shape[0] == 0:
                sl.loc[0] = [seg, None, None, 0.0, 1.0]

            df = self.merge_dataframes(linear_references, rest, sl, result_data)
            all_data.append(df)

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
