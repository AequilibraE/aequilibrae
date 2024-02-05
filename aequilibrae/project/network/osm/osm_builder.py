import gc
import importlib.util as iutil
import string
from math import floor
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
from pandas import json_normalize
from shapely import MultiLineString
from shapely.geometry import Polygon, LineString

from aequilibrae.context import get_active_project
from aequilibrae.parameters import Parameters
from aequilibrae.project.project_creation import remove_triggers, add_triggers
from aequilibrae.utils import WorkerThread
from aequilibrae.utils.db_utils import commit_and_close, read_and_close, list_columns
from aequilibrae.utils.spatialite_utils import connect_spatialite
from .model_area_gridding import geometry_grid

pyqt = iutil.find_spec("PyQt5") is not None
if pyqt:
    from PyQt5.QtCore import pyqtSignal

if iutil.find_spec("qgis") is not None:
    pass


class OSMBuilder(WorkerThread):
    if pyqt:
        building = pyqtSignal(object)

    def __init__(self, data, project, model_area: Polygon) -> None:
        WorkerThread.__init__(self, None)

        project.logger.info("Preparing OSM builder")
        self.__emit_all(["text", "Preparing OSM builder"])

        self.project = project or get_active_project()
        self.logger = self.project.logger
        self.model_area = geometry_grid(model_area, 4326)
        self.path = self.project.path_to_file
        self.node_start = 10000
        self.report = []
        self.__all_ltp = pd.DataFrame([])
        self.__link_id = 1
        self.__valid_links = []

        nids = np.arange(data["nodes"].shape[0]) + self.node_start
        nodes = data["nodes"].assign(is_centroid=0, modes="", link_types="", node_id=nids).reset_index(drop=True)
        self.node_df = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.lon, nodes.lat), crs=4326)
        del nodes
        del data["nodes"]
        gc.collect()
        self.links_df = data["links"]

    def __emit_all(self, *args):
        if pyqt:
            self.building.emit(*args)

    def doWork(self):
        with commit_and_close(connect_spatialite(self.path)) as conn:
            self.__update_table_structure(conn)
            self.importing_network(conn)
            conn.execute(
                "DELETE FROM nodes WHERE node_id NOT IN (SELECT a_node FROM links union all SELECT b_node FROM links)"
            )
        self.__emit_all(["finished_threaded_procedure", 0])

    def importing_network(self, conn):
        node_count = pd.DataFrame(self.links_df["nodes"].explode("nodes")).assign(counter=1).groupby("nodes").count()

        self.node_df.osm_id = self.node_df.osm_id.astype(np.int64)
        self.node_df.set_index(["osm_id"], inplace=True)

        self.logger.info("Creating necessary link types")
        self.__emit_all(["text", "Creating necessary link types"])
        self.__build_link_types()
        shape_ = self.links_df.shape[0]
        message_step = floor(shape_ / 100)
        self.__emit_all(["maxValue", shape_])

        self.establish_modes_for_all_links(conn)
        self.process_link_attributes()

        self.logger.info("Geo-procesing links")
        self.__emit_all(["text", "Adding network links"])
        geometries = []
        for counter, (idx, link) in enumerate(self.links_df.iterrows()):
            self.__emit_all(["Value", counter])
            if counter % message_step == 0:
                self.logger.info(f"Creating segments from {counter:,} out of {shape_ :,} OSM link objects")

            # How can I link have less than two points?
            if not isinstance(link["nodes"], list):
                geometries.append(LineString())
                self.logger.error(f"OSM link {idx} does not have a list of nodes.")
                continue

            if len(link["nodes"]) < 2:
                self.logger.error(f"Link {idx} has less than two nodes. {link.nodes}")
                geometries.append(LineString())
                continue

            # The link is a straight line between two points
            # Or all midpoints are only part of a single link
            node_indices = node_count.loc[link["nodes"], "counter"]
            if len(link["nodes"]) == 2 or node_indices[1:-1].max() == 1:
                # The link has no intersections
                geo = self.__build_geometry(link.nodes)
            else:
                # The link has intersections
                intersecs = np.where(node_indices > 1)[0]
                geos = []
                for i, j in zip(intersecs[:-1], intersecs[1:]):
                    geos.append(self.__build_geometry(link.nodes[i: j + 1]))
                geo = MultiLineString(geos)

            geometries.append(geo)

        # Builds the link Geo dataframe
        self.links_df.drop(columns=["nodes"], inplace=True)
        self.links_df = gpd.GeoDataFrame(self.links_df, geometry=geometries, crs=4326)
        self.links_df = self.links_df.clip(self.model_area).explode(index_parts=False)
        self.links_df = self.links_df[self.links_df.geometry.length > 0]

        self.links_df.loc[:, "link_id"] = np.arange(self.links_df.shape[0]) + 1

        self.node_df.reset_index(inplace=True)
        cols = ["node_id", "osm_id", "is_centroid", "modes", "link_types"]
        self.node_df = gpd.GeoDataFrame(self.node_df[cols], geometry=self.node_df.geometry, crs=self.node_df.crs)

        # Saves the data to disk in case of issues loading it to the database
        osm_data_path = Path(self.project.project_base_path) / "osm_data"
        osm_data_path.mkdir(exist_ok=True)
        self.links_df.to_parquet(osm_data_path / "links.parquet")
        self.node_df.to_parquet(osm_data_path / "nodes.parquet")

        self.logger.info("Adding nodes to file")
        self.__emit_all(["text", "Adding nodes to file"])

        # Removing the triggers before adding all nodes makes things a LOT faster
        remove_triggers(conn, self.logger, "network")

        self.node_df.to_file(self.project.path_to_file, driver="SQLite", spatialite=True, layer="nodes", mode="a")
        del self.node_df
        gc.collect()

        # But we need to add them back to add the links
        add_triggers(conn, self.logger, "network")

        # self.links_df.to_file(self.project.path_to_file, driver="SQLite", spatialite=True, layer="links", mode="a")

        # I could not get the above line to work, so I used the following code instead
        insert_qry = "INSERT INTO links ({},a_node, b_node, distance, geometry) VALUES({},0,0,0, GeomFromWKB(?, 4326))"
        cols_no_geo = self.links_df.columns.tolist()
        cols_no_geo.remove("geometry")
        insert_qry = insert_qry.format(", ".join(cols_no_geo), ", ".join(["?"] * len(cols_no_geo)))

        geos = self.links_df.geometry.to_wkb()
        cols = cols_no_geo + ["geometry"]
        links_df = pd.DataFrame(self.links_df[cols_no_geo]).assign(geometry=geos)[cols].to_records(index=False)

        del self.links_df
        gc.collect()
        self.logger.info("Adding links to file")
        self.__emit_all(["text", "Adding links to file"])
        conn.executemany(insert_qry, links_df)

    def __build_geometry(self, nodes: List[int]) -> LineString:
        return LineString(self.node_df.loc[nodes, "geometry"])



    def __process_link_chunk(self):

        if "tags" in df.columns:
            df = pd.concat([df, json_normalize(df["tags"])], axis=1).drop(columns=["tags"])
            df.columns = [x.replace(":", "_") for x in df.columns]

    def __build_link_types(self):
        data = []
        with read_and_close(self.project.path_to_file) as conn:
            self.__all_ltp = pd.read_sql('SELECT link_type_id, link_type, "" as highway from link_types', conn)

        self.links_df.highway.fillna("missing", inplace=True)
        self.links_df.highway = self.links_df.highway.str.lower()
        for i, lt in enumerate(self.links_df.highway.unique()):
            if str(lt) in self.__all_ltp.highway.values:
                continue
            data.append([*self.__define_link_type(str(lt)), str(lt)])
            self.__all_ltp = pd.concat(
                [self.__all_ltp, pd.DataFrame(data, columns=["link_type_id", "link_type", "highway"])]
            )
            self.__all_ltp.drop_duplicates(inplace=True)
        self.links_df = self.links_df.merge(self.__all_ltp[["link_type", "highway"]], on="highway", how="left")
        self.links_df.drop(columns=["highway"], inplace=True)

    def __define_link_type(self, link_type: str) -> str:
        proj_link_types = self.project.network.link_types
        original_link_type = link_type
        link_type = "".join([x for x in link_type if x in string.ascii_letters + "_"]).lower()

        split = link_type.split("_")
        for i, piece in enumerate(split[1:]):
            if piece in ["link", "segment", "stretch"]:
                link_type = "_".join(split[0: i + 1])

        if self.__all_ltp.shape[0] >= 51:
            link_type = "aggregate_link_type"

        if len(link_type) == 0:
            link_type = "empty"

        if link_type in self.__all_ltp.link_type.values:
            lt = proj_link_types.get_by_name(link_type)
            if original_link_type not in lt.description:
                lt.description += f", {original_link_type}"
                lt.save()
            return [lt.link_type_id, link_type]

        letter = link_type[0]
        if letter in self.__all_ltp.link_type_id.values:
            letter = letter.upper()
            if letter in self.__all_ltp.link_type_id.values:
                for letter in string.ascii_letters:
                    if letter not in self.__all_ltp.link_type_id.values:
                        break
        lt = proj_link_types.new(letter)
        lt.link_type = link_type
        lt.description = f"Link types from Open Street Maps: {original_link_type}"
        lt.save()
        return [letter, link_type]


    def process_link_attributes(self):
        self.links_df = self.links_df.assign(direction=0, link_id=0)
        self.links_df.loc[self.links_df.oneway == "yes", "direction"] = 1
        self.links_df.loc[self.links_df.oneway == "backward", "direction"] = -1
        p = Parameters()
        fields = p.parameters["network"]["links"]["fields"]

        for x in fields["one-way"]:
            keys_ = list(x.values())[0]
            field = list(x.keys())[0]
            osm_name = keys_.get("osm_source", field).replace(":", "_")
            self.links_df.rename(columns={osm_name: field}, inplace=True, errors="ignore")

        for x in fields["two-way"]:
            keys_ = list(x.values())[0]
            field = list(x.keys())[0]
            if "osm_source" not in keys_:
                continue
            osm_name = keys_.get("osm_source", field).replace(":", "_")
            self.links_df[f"{field}_ba"] = self.links_df[osm_name].copy()
            self.links_df.rename(columns={osm_name: f"{field}_ab"}, inplace=True, errors="ignore")
            if "osm_behaviour" in keys_ and keys_["osm_behaviour"] == "divide":
                self.links_df[f"{field}_ab"] = pd.to_numeric(self.links_df[f"{field}_ab"], errors="coerce") / 2
                self.links_df[f"{field}_ba"] = pd.to_numeric(self.links_df[f"{field}_ba"], errors="coerce") / 2

                if f"{field}_forward" in self.links_df:
                    fld = pd.to_numeric(self.links_df[f"{field}_forward"], errors="coerce")
                    self.links_df.loc[fld > 0, f"{field}_ab"] = fld[fld > 0]
                if f"{field}_backward" in self.links_df:
                    fld = pd.to_numeric(self.links_df[f"{field}_backward"], errors="coerce")
                    self.links_df.loc[fld > 0, f"{field}_ba"] = fld[fld > 0]
        cols = list_columns(self.project.conn, "links") + ["nodes"]
        self.links_df = self.links_df[[x for x in cols if x in self.links_df.columns]]
        gc.collect()

    def establish_modes_for_all_links(self, conn):
        p = Parameters()
        modes = p.parameters["network"]["osm"]["modes"]

        mode_codes = conn.execute("SELECT mode_name, mode_id from modes").fetchall()
        mode_codes = {p[0]: p[1] for p in mode_codes}

        type_list = {}
        notfound = ""
        for mode, val in modes.items():
            all_types = val["link_types"]
            md = mode_codes[mode]
            for tp in all_types:
                type_list[tp] = "".join(sorted("{}{}".format(type_list.get(tp, ""), md)))
            if val["unknown_tags"]:
                notfound += md

        type_list = {k: "".join(set(v)) for k, v in type_list.items()}

        df = pd.DataFrame([[k, v] for k, v in type_list.items()], columns=["link_type", "modes"])
        self.links_df = self.links_df.merge(df, on="link_type", how="left")
        self.links_df.modes.fillna("".join(sorted(notfound)), inplace=True)

    ######## TABLE STRUCTURE UPDATING ########
    def __update_table_structure(self, conn):
        structure = conn.execute("pragma table_info(Links)").fetchall()
        has_fields = [x[1].lower() for x in structure]
        fields = [field.lower() for field in self.get_link_fields()] + ["osm_id"]
        for field in [f for f in fields if f not in has_fields]:
            ltype = self.get_link_field_type(field).upper()
            conn.execute(f"Alter table Links add column {field} {ltype}")
        conn.commit()

    @staticmethod
    def get_link_fields():
        p = Parameters()
        fields = p.parameters["network"]["links"]["fields"]
        owf = [list(x.keys())[0] for x in fields["one-way"]]

        twf1 = ["{}_ab".format(list(x.keys())[0]) for x in fields["two-way"]]
        twf2 = ["{}_ba".format(list(x.keys())[0]) for x in fields["two-way"]]

        return owf + twf1 + twf2 + ["osm_id"]

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