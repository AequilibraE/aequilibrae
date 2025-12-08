import csv
import math
import re
import string
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import shapely.wkb
import shapely.wkt
from pyproj import Transformer
from shapely.geometry import LineString, Point

from aequilibrae import logger
from aequilibrae.parameters import Parameters
from aequilibrae.utils.db_utils import commit_and_close


def __dfs(graph, start):
    """A quick and dirty DFS implementation to return the leaves of a graph."""
    visited = set()
    real = []
    stack = [start]
    while len(stack) > 0:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            if vertex in graph:
                stack.extend(graph[vertex])
            else:
                real.append(vertex)
        else:
            raise ValueError(f"Recursive use_group ('{vertex}') found")
    return real


def resolve_recusive_dict(base_dict):
    """Resolve each entry in the graph."""
    resolved = defaultdict(list)

    for key, values in base_dict.items():
        for value in values:
            resolved[key].extend(__dfs(base_dict, value))

    return dict(resolved)


class GMNSBuilder:
    def __init__(
        self, net, link_path: str, node_path: str, uses_path: str = None, geom_path: str = None, srid: int = 4326
    ) -> None:
        self.p = Parameters()
        self.links = net.links
        self.nodes = net.nodes
        self.link_types = net.link_types
        self.modes = net.modes
        self.__path_file = net.project.path_to_file

        self.link_df = pd.read_csv(link_path).fillna("")
        self.node_df = pd.read_csv(node_path).fillna("")
        self.uses_df = pd.read_csv(uses_path) if uses_path is not None else uses_path
        self.geom_df = pd.read_csv(geom_path) if geom_path is not None else geom_path
        self.srid = srid

        self.l_equiv = self.p.parameters["network"]["gmns"]["link"]["equivalency"]
        self.n_equiv = self.p.parameters["network"]["gmns"]["node"]["equivalency"]

    def doWork(self):
        p = self.p
        gmns_n_fields = p.parameters["network"]["gmns"]["node"]["fields"]
        gmns_l_fields = p.parameters["network"]["gmns"]["link"]["fields"]

        # Checking if all required fields are in GMNS links and nodes files
        for field in [x for x in gmns_n_fields if gmns_n_fields[x]["required"]]:
            if field not in self.node_df.columns.to_list():
                raise ValueError(f"In GMNS nodes file: field '{field}' required, but not found.")

        for field in [x for x in gmns_l_fields if gmns_l_fields[x]["required"]]:
            if field not in self.link_df.columns.to_list():
                raise ValueError(f"In GMNS links file: field '{field}' required, but not found.")

        gmns_geom = self.l_equiv["geometry"]
        if gmns_geom not in self.link_df.columns.to_list():
            if self.geom_df is None:
                raise ValueError(
                    "To create an aequilibrae links table, geometries information must be provided either in the GMNS link table or in a separate file ('geometry_path' attribute)."
                )
            else:
                self.link_df = self.link_df.merge(self.geom_df, on="geometry_id", how="left")

        self.maybe_transform_srid(self.srid)

        # Creating direction list based on list of two-way links
        direction = self.get_aeq_direction()

        # Creating speeds, capacities and lanes lists based on direction list
        speed_ab, speed_ba, capacity_ab, capacity_ba, lanes_ab, lanes_ba, toll_ab, toll_ba = self.get_ab_lists(
            direction
        )

        # Adding new fields to AequilibraE links table / Preparing it to receive information from GMNS table.
        l_fields = self.links.fields
        l_fields.add("notes", description="More information about the link", data_type="TEXT")

        if "toll" in self.link_df.columns.to_list():
            l_fields.add("toll_ab", description="Toll", data_type="NUMERIC")
            l_fields.add("toll_ba", description="Toll", data_type="NUMERIC")

        if self.l_equiv["lanes"] in self.link_df.columns.to_list():
            l_fields.add("lanes_ab", description="Lanes", data_type="NUMERIC")
            l_fields.add("lanes_ba", description="Lanes", data_type="NUMERIC")

        other_ldict = {}
        other_lfields = [x for x in gmns_l_fields if not gmns_l_fields[x]["required"]]
        for fld in other_lfields:
            if fld in self.link_df.columns.to_list() and fld not in l_fields.all_fields():
                l_fields.add(
                    f"{fld}",
                    description=f"{gmns_l_fields[fld]['description']}",
                    data_type=f"{gmns_l_fields[fld]['type']}",
                )
                if fld == "toll":
                    other_ldict.update({"toll_ab": toll_ab})
                    other_ldict.update({"toll_ba": toll_ba})

                other_ldict.update({f"{fld}": self.link_df[fld]})

        l_fields.save()

        all_fields = list(gmns_l_fields)
        missing_f = [c for c in list(self.link_df.columns) if c not in all_fields]
        if missing_f != []:
            print(
                f"Fields not imported from link table: {'; '.join(missing_f)}. If you want them to be imported, please modify the parameters.yml file."
            )

        # Adding new fields to AequilibraE nodes table / Preparing it to receive information from GMNS table.

        n_fields = self.nodes.fields
        n_fields.add("notes", description="More information about the node", data_type="TEXT")

        other_ndict = {}
        other_nfields = [x for x in gmns_n_fields if not gmns_n_fields[x]["required"]]
        for fld in other_nfields:
            if fld in self.node_df.columns.to_list() and fld not in l_fields.all_fields():
                n_fields.add(
                    f"{fld}",
                    description=f"{gmns_n_fields[fld]['description']}",
                    data_type=f"{gmns_n_fields[fld]['type']}",
                )
                other_ndict.update({f"{fld}": self.node_df[fld]})

        n_fields.save()

        all_fields = list(gmns_n_fields)
        missing_f = [c for c in list(self.node_df.columns) if c not in all_fields]
        if missing_f != []:
            print(
                f"Fields not imported from node table: {'; '.join(missing_f)}. If you want them to be imported, please modify the parameters.yml file."
            )

        # Getting information from some optinal GMNS fields

        gmns_name = self.l_equiv["name"]
        name_list = (
            self.link_df[gmns_name].to_list()
            if gmns_name in self.link_df.columns.to_list()
            else ["" for _ in range(len(self.link_df))]
        )

        # Creating link_type and modes list
        link_types_list = self.save_types_to_aeq()
        mode_ids_list = self.save_modes_to_aeq()

        # Checking if the links boundaries coordinates match the "from" and "to" nodes coordinates
        self.correct_geometries()

        # Setting centroid equals 1 when informed in the 'node_type' node table field
        centroid_flag = (
            [1 if x == "centroid" else 0 for x in self.node_df["node_type"].to_list()]
            if "node_type" in self.node_df.columns.to_list()
            else 0
        )

        # Creating dataframes for adding nodes and links information to AequilibraE model

        nodes_fields = {
            "node_id": self.node_df[self.n_equiv["node_id"]],
            "is_centroid": centroid_flag,
            "x_coord": self.node_df.x_coord,
            "y_coord": self.node_df.y_coord,
            "notes": "from GMNS file",
        }

        links_fields = {
            "link_id": self.link_df[self.l_equiv["link_id"]],
            "a_node": self.link_df[self.l_equiv["a_node"]],
            "b_node": self.link_df[self.l_equiv["b_node"]],
            "direction": direction,
            "modes": mode_ids_list,
            "link_type": link_types_list,
            "name": name_list,
            "speed_ab": speed_ab,
            "speed_ba": speed_ba,
            "capacity_ab": capacity_ab,
            "capacity_ba": capacity_ba,
            "geometry": self.link_df.geometry,
            "lanes_ab": lanes_ab,
            "lanes_ba": lanes_ba,
            "notes": "from GMNS file",
        }

        nodes_fields.update(other_ndict)
        links_fields.update(other_ldict)

        self.save_to_database(links_fields, nodes_fields)

    def maybe_transform_srid(self, srid):
        if srid == 4326:
            return

        transformer = Transformer.from_crs(f"epsg:{self.srid}", "epsg:4326", always_xy=True)

        # For node table
        lons, lats = transformer.transform(self.node_df.loc[:, "x_coord"], self.node_df.loc[:, "y_coord"])
        self.node_df = self.node_df.astype({"x_coord": np.float64, "y_coord": np.float64})
        self.node_df.loc[:, "x_coord"] = np.around(lons, decimals=10)
        self.node_df.loc[:, "y_coord"] = np.around(lats, decimals=10)

        # For link table
        for idx, row in self.link_df.iterrows():
            geom = shapely.wkt.loads(row.geometry)
            x_points = [int(x[0]) for x in list(geom.coords)]
            y_points = [int(x[1]) for x in list(geom.coords)]

            lons, lats = transformer.transform(x_points, y_points)
            new_points = list(zip(np.around(lons, decimals=10), np.around(lats, decimals=10)))

            self.link_df.loc[idx, "geometry"] = LineString(new_points).wkt

    def get_aeq_direction(self):
        gmns_dir = self.l_equiv["direction"]
        gmns_cap = self.l_equiv["capacity"]
        gmns_lanes = self.l_equiv["lanes"]

        # Creating a direction list containing information in the AequilibraE standard (0 value for two-way links)
        self.link_df[gmns_dir] = [0 if x not in [1, True] else 1 for x in list(self.link_df[gmns_dir])]
        sorted_df = self.link_df.sort_values("link_id")

        to_drop_lst = []
        for idx, row in sorted_df.iterrows():
            same_dir_df = sorted_df[
                (sorted_df[["from_node_id", "to_node_id"]].apply(tuple, 1) == (row["from_node_id"], row["to_node_id"]))
                & (sorted_df.link_id > row.link_id)
            ]
            if same_dir_df.shape[0] > 0 and row[gmns_dir]:
                if gmns_lanes in sorted_df.columns.to_list():
                    self.link_df.at[idx, gmns_lanes] = row[gmns_lanes] + same_dir_df[gmns_lanes].sum()

                if gmns_cap in sorted_df.columns.to_list():
                    self.link_df.at[idx, gmns_cap] = row[gmns_cap] + same_dir_df[gmns_cap].sum()

                if "row_width" in sorted_df.columns.to_list():
                    self.link_df.at[idx, "row_width"] = row["row_width"] + same_dir_df["row_width"].sum()

            opp_dir_df = sorted_df[
                (sorted_df[["from_node_id", "to_node_id"]].apply(tuple, 1) == (row["to_node_id"], row["from_node_id"]))
                & (sorted_df.link_id > row.link_id)
            ]
            if opp_dir_df.shape[0] > 0:
                self.link_df.at[idx, gmns_dir] = 0

                if "row_width" in sorted_df.columns.to_list() and row[gmns_dir] in [1, True]:
                    self.link_df.at[idx, "row_width"] = row["row_width"] + opp_dir_df["row_width"].sum()

            to_drop_lst += list(same_dir_df.index) + list(opp_dir_df.index)

        sorted_df.drop(to_drop_lst, axis=0, inplace=True)
        self.link_df = self.link_df[self.link_df.link_id.isin(list(sorted_df.link_id))]
        self.link_df.reset_index(drop=True, inplace=True)
        direction = list(self.link_df[gmns_dir])

        return direction

    def get_ab_lists(self, direction):
        gmns_speed = self.l_equiv["speed"]
        gmns_cap = self.l_equiv["capacity"]
        gmns_lanes = self.l_equiv["lanes"]

        speed_ab = ["" for _ in range(len(self.link_df))]
        speed_ba = ["" for _ in range(len(self.link_df))]
        capacity_ab = ["" for _ in range(len(self.link_df))]
        capacity_ba = ["" for _ in range(len(self.link_df))]
        lanes_ab = ["" for _ in range(len(self.link_df))]
        lanes_ba = ["" for _ in range(len(self.link_df))]
        toll_ab = ["" for _ in range(len(self.link_df))]
        toll_ba = ["" for _ in range(len(self.link_df))]

        for idx, row in self.link_df.iterrows():
            if gmns_speed in self.link_df.columns.to_list():
                [speed_ab[idx], speed_ba[idx]] = (
                    [row[gmns_speed], ""] if direction[idx] == 1 else [row[gmns_speed], row[gmns_speed]]
                )

            if gmns_cap in self.link_df.columns.to_list():
                [capacity_ab[idx], capacity_ba[idx]] = (
                    [row[gmns_cap], ""] if direction[idx] == 1 else [row[gmns_cap], row[gmns_cap]]
                )

            if gmns_lanes in self.link_df.columns.to_list():
                [lanes_ab[idx], lanes_ba[idx]] = (
                    [row[gmns_lanes], ""] if direction[idx] == 1 else [row[gmns_lanes], row[gmns_lanes]]
                )

            if "toll" in self.link_df.columns.to_list():
                [toll_ab[idx], toll_ba[idx]] = [row["toll"], ""] if direction[idx] == 1 else [row["toll"], row["toll"]]

        return speed_ab, speed_ba, capacity_ab, capacity_ba, lanes_ab, lanes_ba, toll_ab, toll_ba

    def save_types_to_aeq(self):
        gmns_ltype = self.l_equiv["link_type"]

        # Setting link_type = 'unclassified' if there is no information about it in the GMNS links table
        if gmns_ltype not in self.link_df.columns.to_list():
            gmns_ltype = "link_type_name"
            if gmns_ltype not in self.link_df.columns.to_list():
                link_types_list = ["unclassified" for _ in range(len(self.link_df))]
            else:
                link_types_list = self.link_df[gmns_ltype].to_list()
        else:
            link_types_list = self.link_df[gmns_ltype].to_list()

        ## Adding link_types to AequilibraE model

        link_types_list = [s.replace("-", "_") for s in link_types_list]
        type_saved = ""
        all_types = list(self.link_types.all_types())
        for lt_name in list(dict.fromkeys(link_types_list)):
            letters = lt_name.lower() + lt_name.upper() + string.ascii_letters
            letters = "".join([lt for lt in letters if lt not in all_types + [type_saved]])

            link_types = self.link_types
            new_type = link_types.new(letters[0])
            new_type.link_type = lt_name
            new_type.description = "Link type from GMNS link table"
            new_type.save()
            type_saved = letters[0]

        return link_types_list

    def save_modes_to_aeq(self):
        gmns_modes = self.l_equiv["modes"]

        if gmns_modes in self.link_df.columns.to_list():
            modes_list = self.link_df[gmns_modes].to_list()
            for i in range(len(modes_list)):
                if modes_list[i] == "":
                    modes_list[i] = "unspecified_mode"
        else:
            modes_list = ["unspecified_mode" for _ in range(len(self.link_df))]

        char_replaces = {
            "0": "zero",
            "1": "one",
            "2": "two",
            "3": "three",
            "4": "four",
            "5": "five",
            "6": "six",
            "7": "seven",
            "8": "eight",
            "9": "nine",
        }
        pattern = re.compile(r"\d")

        if self.uses_df is not None:
            uses = csv.reader(self.uses_df.uses)
            uses = [
                {
                    pattern.sub(lambda x: "_" + char_replaces[x.group(0)], use)
                    .strip()
                    .replace("+", "")
                    .replace("-", "_")
                    .upper()
                    for use in line
                }
                for line in uses
            ]
            groups_dict = dict(zip(self.uses_df.use_group.map(str.upper), uses))
        else:
            groups_dict = {}

        resolved_groups = resolve_recusive_dict(groups_dict)

        modes_list = [
            pattern.sub(lambda x: "_" + char_replaces[x.group()], s).replace("+", "").replace("-", "_")
            for s in modes_list
        ]

        with commit_and_close(self.__path_file) as conn:
            existing_modes = dict(conn.execute("select mode_name, mode_id from modes").fetchall())

        # Invert the resolved_groups dictionary, we're interested in which use_groups contain our "use"
        resolved_use_groups = defaultdict(list)
        for group, values in resolved_groups.items():
            for v in values:
                resolved_use_groups[v].append(group)

        modes = deepcopy(existing_modes)
        unused_chars = {x for x in string.ascii_letters if x not in existing_modes.values()}

        # Create a new mode for the ones that don't exist. Use their first char (lower or upper), or a random unused one
        # if neither are available.
        for m in resolved_use_groups.keys():
            if m not in existing_modes:
                if m[0].lower() in unused_chars:
                    char = m[0].lower()
                    unused_chars.remove(char)
                elif m[0].upper() in unused_chars:
                    char = m[0].upper()
                    unused_chars.remove(char)
                else:
                    char = unused_chars.pop()

                new_mode = self.modes.new(char)
                new_mode.mode_name = m
                new_mode.description = f"GMNS use groups: {', '.join(resolved_use_groups[m])}"
                self.modes.add(new_mode)
                new_mode.save()

                modes[m] = char

        # For each mode specified in the links we need to parse the mode, then attempt to resolve the real modes it
        # corresponds to, i.e. map 'ALL' to all real modes and 'BIKE' to the bike mode
        modes_gathered = []
        for mode in modes_list:
            ids = set()
            if "," in mode:
                unresolved_modes = (mode.strip() for mode in list(csv.reader([mode]))[0])
            else:
                unresolved_modes = [mode]

            for mode in unresolved_modes:
                if mode in resolved_groups:
                    for resolved_mode in resolved_groups[mode]:
                        ids.add(modes[resolved_mode])
                else:
                    ids.add(modes[mode])
            modes_gathered.append("".join(sorted(ids)))

        return modes_gathered

    def correct_geometries(self):
        p = self.p
        gmns_lid = self.l_equiv["link_id"]
        gmns_a_node = self.l_equiv["a_node"]
        gmns_b_node = self.l_equiv["b_node"]
        gmns_nid = self.n_equiv["node_id"]
        critical_dist = p.parameters["network"]["gmns"]["critical_dist"]

        for idx, row in self.link_df.iterrows():
            [from_point_x, from_point_y] = (
                self.node_df.loc[self.node_df[gmns_nid] == row[gmns_a_node], ["x_coord", "y_coord"]]
                .apply(list, 1)
                .item()
            )
            [to_point_x, to_point_y] = (
                self.node_df.loc[self.node_df[gmns_nid] == row[gmns_b_node], ["x_coord", "y_coord"]]
                .apply(list, 1)
                .item()
            )

            link_geom = shapely.wkt.loads(row.geometry)
            link_points = list(link_geom.coords)
            link_start_boundary = link_points[0]
            link_end_boundary = link_points[-1]

            if link_start_boundary != (from_point_x, from_point_y):
                start_to_from_dist = (
                    Point(link_start_boundary).distance(Point(from_point_x, from_point_y)) * math.pi * 6371000 / 180
                )

                link_points = (
                    [(from_point_x, from_point_y)] + link_points[1:]
                    if start_to_from_dist <= critical_dist
                    else [(from_point_x, from_point_y)] + link_points[:]
                )

                new_link = LineString(link_points)
                self.link_df.loc[idx, "geometry"] = new_link.wkt
                logger.info(
                    f"Geometry for link_id = {row[gmns_lid]} has just been corrected. It was not connected to its start node."
                )

            if link_end_boundary != (to_point_x, to_point_y):
                end_to_to_dist = (
                    Point(link_end_boundary).distance(Point(to_point_x, to_point_y)) * math.pi * 6371000 / 180
                )

                link_points = (
                    link_points[:-1] + [(to_point_x, to_point_y)]
                    if end_to_to_dist <= critical_dist
                    else link_points[:] + [(to_point_x, to_point_y)]
                )

                new_link = LineString(link_points)
                self.link_df.loc[idx, "geometry"] = new_link.wkt
                logger.info(
                    f"Geometry for link_id = {row[gmns_lid]} has just been corrected. It was not connected to its end node."
                )

    def save_to_database(self, links_fields, nodes_fields):
        aeq_nodes_df = pd.DataFrame(nodes_fields)
        aeq_links_df = pd.DataFrame(links_fields)

        nodes_fields_list = list(nodes_fields.keys())
        nodes_fields_list.pop(nodes_fields_list.index("y_coord"))
        nodes_fields_list = ["geometry" if x == "x_coord" else x for x in nodes_fields_list]

        n_query = "insert into nodes(" + ", ".join(nodes_fields_list) + ")"
        n_query += (
            " values("
            + ", ".join(["MakePoint(?,?, 4326)" if x == "geometry" else "?" for x in nodes_fields_list])
            + ")"
        )
        n_params_list = aeq_nodes_df.to_records(index=False)

        with commit_and_close(self.__path_file, spatial=True) as conn:
            conn.executemany(n_query, n_params_list)

            l_query = "insert into links(" + ", ".join(list(links_fields.keys())) + ")"
            l_query += (
                " values("
                + ", ".join(["GeomFromTEXT(?,4326)" if x == "geometry" else "?" for x in list(links_fields.keys())])
                + ")"
            )
            l_params_list = aeq_links_df.to_records(index=False)

            conn.executemany(l_query, l_params_list)
