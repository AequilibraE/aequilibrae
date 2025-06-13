import gc
import string
from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from pandas import json_normalize
from shapely.geometry import Polygon

from aequilibrae.context import get_active_project
from aequilibrae.parameters import Parameters
from aequilibrae.project.project_creation import remove_triggers, add_triggers
from aequilibrae.utils.db_utils import commit_and_close, list_columns
from aequilibrae.utils.aeq_signal import SIGNAL, simple_progress
from aequilibrae.utils.interface.worker_thread import WorkerThread
from .model_area_gridding import geometry_grid


class OSMBuilder(WorkerThread):
    signal = SIGNAL(object)

    def __init__(self, data, project, model_area: Polygon, clean: bool) -> None:
        WorkerThread.__init__(self, None)

        project.logger.info("Preparing OSM builder")
        self.signal.emit(["set_text", "Preparing OSM builder"])

        self.project = project or get_active_project()
        self.logger = self.project.logger
        self.model_area = geometry_grid(model_area, 4326)
        self.path = self.project.path_to_file
        self.node_start = 10000
        self.clean = clean
        self.report = []
        self.__all_ltp = pd.DataFrame([])
        self.__link_id = 1
        self.__valid_links = []

        # Building shapely geometries makes the code surprisingly slower.
        self.node_df = data["nodes"]
        self.node_df.loc[:, "node_id"] = np.arange(self.node_start, self.node_start + self.node_df.shape[0])
        gc.collect()
        self.links_df = data["links"]

    def doWork(self):
        with commit_and_close(self.path, spatial=True) as conn:
            self.__update_table_structure(conn)
            self.importing_network(conn)

            self.logger.info("Cleaning things up")
            conn.execute(
                "DELETE FROM nodes WHERE node_id NOT IN (SELECT a_node FROM links union all SELECT b_node FROM links)"
            )
            conn.commit()
            self.__do_clean(conn)

        self.signal.emit(["finished"])

    def importing_network(self, conn):
        self.logger.info("Importing the network")
        node_count = pd.DataFrame(self.links_df["nodes"].explode("nodes")).assign(counter=1).groupby("nodes").count()

        self.node_df.osm_id = self.node_df.osm_id.astype(np.int64)
        self.node_df.set_index(["osm_id"], inplace=True)

        self.__process_link_chunk()

        self.logger.info("Geo-procesing links")
        geometries = []
        self.links_df.set_index(["osm_id"], inplace=True)

        for idx, link in simple_progress(self.links_df.iterrows(), self.signal, "Adding network links"):
            # How can I link have less than two points?
            if not isinstance(link["nodes"], list):
                self.logger.debug(f"OSM link/feature {idx} does not have a list of nodes.")
                continue

            if len(link["nodes"]) < 2:
                self.logger.debug(f"Link {idx} has less than two nodes. {link.nodes}")
                continue

            # The link is a straight line between two points
            # Or all midpoints are only part of a single link
            node_indices = node_count.loc[link["nodes"], "counter"].to_numpy()
            if len(link["nodes"]) == 2 or node_indices[1:-1].max() == 1:
                # The link has no intersections
                geometries.append([idx, self._build_geometry(link.nodes)])
            else:
                # Make sure we get the first and last nodes, as they are certainly the extremities of the sublinks
                node_indices[0] = 2
                node_indices[-1] = 2
                # The link has intersections
                # We build repeated records for links when they have intersections
                # This is because it is faster to do this way and then have all the data repeated
                # when doing the join with the link fields below
                intersecs = np.where(node_indices > 1)[0]
                for i, j in zip(intersecs[:-1], intersecs[1:]):
                    geometries.append([idx, self._build_geometry(link.nodes[i : j + 1])])

        # Builds the link Geo dataframe
        self.links_df.drop(columns=["nodes"], inplace=True)
        # We build a dataframe with the geometries created above
        # and join with the database
        geo_df = pd.DataFrame(geometries, columns=["link_id", "geometry"]).set_index("link_id")
        self.links_df = self.links_df.join(geo_df, how="inner")

        self.links_df.loc[:, "link_id"] = np.arange(self.links_df.shape[0]) + 1

        self.node_df = self.node_df.reset_index()

        # Saves the data to disk in case of issues loading it to the database
        osm_data_path = Path(self.project.project_base_path) / "osm_data"
        osm_data_path.mkdir(exist_ok=True)
        self.links_df.to_parquet(osm_data_path / "links.parquet")
        self.node_df.to_parquet(osm_data_path / "nodes.parquet")

        self.logger.info("Adding nodes to file")
        self.signal.emit(["set_text", "Adding nodes to file"])

        # Removing the triggers before adding all nodes makes things a LOT faster
        remove_triggers(conn, self.logger, "network")

        cols = ["node_id", "osm_id", "is_centroid", "modes", "link_types", "lon", "lat"]
        insert_qry = f"INSERT INTO nodes ({','.join(cols[:-2])}, geometry) VALUES(?,?,?,?,?, MakePoint(?,?, 4326))"
        conn.executemany(insert_qry, self.node_df[cols].to_records(index=False))

        del self.node_df
        gc.collect()

        # But we need to add them back to add the links
        add_triggers(conn, self.logger, "network")

        # self.links_df.to_file(self.project.path_to_file, driver="SQLite", spatialite=True, layer="links", mode="a")

        # I could not get the above line to work, so I used the following code instead
        self.links_df.index.name = "osm_id"
        self.links_df.reset_index(inplace=True)
        insert_qry = "INSERT INTO links ({},a_node, b_node, distance, geometry) VALUES({},0,0,0, GeomFromText(?, 4326))"
        cols_no_geo = self.links_df.columns.tolist()
        cols_no_geo.remove("geometry")
        insert_qry = insert_qry.format(", ".join(cols_no_geo), ", ".join(["?"] * len(cols_no_geo)))

        cols = cols_no_geo + ["geometry"]
        links_df = self.links_df[cols].to_records(index=False)

        del self.links_df
        gc.collect()
        self.logger.info("Adding links to file")
        self.signal.emit(["set_text", "Adding links to file"])
        conn.executemany(insert_qry, links_df)

    def _build_geometry(self, nodes: List[int]) -> str:
        slice = self.node_df.loc[nodes, :]
        txt = ",".join((slice.lon.astype(str) + " " + slice.lat.astype(str)).tolist())
        return f"LINESTRING({txt})"

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

    def __process_link_chunk(self):
        self.logger.info("Processing link modes, types and fields")
        self.signal.emit(["set_text", "Processing link modes, types and fields"])

        # It is hard to define an optimal chunk_size, so let's assume that 1GB is a good size per chunk
        # And let's also assume that each row will be 200 fields at 8 bytes each
        # This makes 2Gb roughly equal to 2.6 million rows, so 2 million would so.
        chunk_size = 1_000_000
        list_dfs = [self.links_df.iloc[i : i + chunk_size] for i in range(0, self.links_df.shape[0], chunk_size)]
        self.links_df = []
        # Initialize link types
        with self.project.db_connection as conn:
            self.__all_ltp = pd.read_sql('SELECT link_type_id, link_type, "" as highway from link_types', conn)
            for df in simple_progress(list_dfs, self.signal, "Processing chunks"):
                if "tags" in df.columns:
                    # It is critical to reset the index for the concat below to work
                    df.reset_index(drop=True, inplace=True)
                    df = pd.concat([df, json_normalize(df["tags"])], axis=1).drop(columns=["tags"])
                    df.columns = [x.replace(":", "_") for x in df.columns]
                    df = self.__build_link_types(df)
                    df = self.__establish_modes_for_all_links(conn, df)
                    df = self.__process_link_attributes(df)
                else:
                    self.logger.error("OSM link data does not have tags. Skipping an entire data chunk")
                    df = pd.DataFrame([])
                self.links_df.append(df)
        self.links_df = pd.concat(self.links_df, ignore_index=True)

    def __build_link_types(self, df):
        data = []
        df = df.fillna(value={"highway": "missing"})
        df.highway = df.highway.str.lower()
        for lt in df.highway.unique():
            if str(lt) in self.__all_ltp.highway.values:
                continue
            data.append([*self.__define_link_type(str(lt)), str(lt)])
            self.__all_ltp = pd.concat(
                [self.__all_ltp, pd.DataFrame(data, columns=["link_type_id", "link_type", "highway"])]
            )
            self.__all_ltp.drop_duplicates(inplace=True)
        df = df.merge(self.__all_ltp[["link_type", "highway"]], on="highway", how="left")
        return df.drop(columns=["highway"])

    def __define_link_type(self, link_type: str) -> Tuple[str, str]:
        proj_link_types = self.project.network.link_types
        original_link_type = link_type
        link_type = "".join([x for x in link_type if x in string.ascii_letters + "_"]).lower()

        split = link_type.split("_")
        for i, piece in enumerate(split[1:]):
            if piece in ["link", "segment", "stretch"]:
                link_type = "_".join(split[0 : i + 1])

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

    def __establish_modes_for_all_links(self, conn, df: pd.DataFrame) -> pd.DataFrame:
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

        df_aux = pd.DataFrame([[k, v] for k, v in type_list.items()], columns=["link_type", "modes"])
        df = df.merge(df_aux, on="link_type", how="left").fillna(value={"modes": "".join(sorted(notfound))})
        return df

    def __process_link_attributes(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(direction=0, link_id=0)
        if "oneway" in df.columns:
            df.loc[df.oneway == "yes", "direction"] = 1
            df.loc[df.oneway == "backward", "direction"] = -1
        p = Parameters()
        fields = p.parameters["network"]["links"]["fields"]

        for x in fields["one-way"]:
            if "link_type" in x.keys():
                continue
            keys_ = list(x.values())[0]
            field = list(x.keys())[0]
            osm_name = keys_.get("osm_source", field).replace(":", "_")
            df.rename(columns={osm_name: field}, inplace=True, errors="ignore")

        for x in fields["two-way"]:
            keys_ = list(x.values())[0]
            field = list(x.keys())[0]
            if "osm_source" not in keys_:
                continue
            osm_name = keys_.get("osm_source", field).replace(":", "_")
            if osm_name not in df.columns:
                continue
            df[f"{field}_ba"] = df[osm_name].copy()
            df.rename(columns={osm_name: f"{field}_ab"}, inplace=True, errors="ignore")
            if "osm_behaviour" in keys_ and keys_["osm_behaviour"] == "divide":
                df[f"{field}_ab"] = pd.to_numeric(df[f"{field}_ab"], errors="coerce")
                df[f"{field}_ba"] = pd.to_numeric(df[f"{field}_ba"], errors="coerce")

                # Divides the values by 2 or zero them depending on the link direction
                df.loc[df.direction == 0, f"{field}_ab"] /= 2
                df.loc[df.direction == -1, f"{field}_ab"] = 0

                df.loc[df.direction == 0, f"{field}_ba"] /= 2
                df.loc[df.direction == 1, f"{field}_ba"] = 0

                if f"{field}_forward" in df:
                    fld = pd.to_numeric(df[f"{field}_forward"], errors="coerce")
                    df.loc[fld > 0, f"{field}_ab"] = fld[fld > 0]
                if f"{field}_backward" in df:
                    fld = pd.to_numeric(df[f"{field}_backward"], errors="coerce")
                    df.loc[fld > 0, f"{field}_ba"] = fld[fld > 0]
        with self.project.db_connection as conn:
            cols = list_columns(conn, "links") + ["nodes"]
        return df[[x for x in cols if x in df.columns]]

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
