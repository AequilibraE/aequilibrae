import re
import numpy as np
import pandas as pd
import string
import shapely.wkb
import shapely.wkt
from shapely.geometry import LineString, Point
from pyproj import Transformer
from ...utils import WorkerThread

from aequilibrae import logger
from aequilibrae.parameters import Parameters


class GMNSBuilder(WorkerThread):

    def __init__(self, net) -> None:
        WorkerThread.__init__(self, None)
        self.links = net.links
        self.nodes = net.nodes
        self.link_types = net.link_types
        self.modes = net.modes

    def doWork(self, link_file_path: str, node_file_path: str, use_group_path: str = None, geometry_path: str = None, srid: int = 4326):

        p = Parameters()
        gmns_l_fields = p.parameters["network"]["gmns"]["link_fields"]
        gmns_n_fields = p.parameters["network"]["gmns"]["node_fields"]

        # Collecting GMNS fields names

        gmns_link_id_field = gmns_l_fields["link_id"]
        gmns_a_node_field = gmns_l_fields["a_node"]
        gmns_b_node_field = gmns_l_fields["b_node"]
        gmns_direction_field = gmns_l_fields["direction"]
        gmns_speed_field = gmns_l_fields["speed"]
        gmns_capacity_field = gmns_l_fields["capacity"]
        gmns_lanes_field = gmns_l_fields["lanes"]
        gmns_name_field = gmns_l_fields["name"]
        gmns_link_type_field = gmns_l_fields["link_type"]
        gmns_modes_field = gmns_l_fields["modes"]
        gmns_geometry_field = gmns_l_fields["geometry"]

        gmns_node_id_field = gmns_n_fields["node_id"]

        other_lfields = p.parameters["network"]["gmns"]["other_link_fields"]
        other_nfields = p.parameters["network"]["gmns"]["other_node_fields"]

        # Loading GMNS link and node files

        gmns_links_df = pd.read_csv(link_file_path).fillna("")
        gmns_nodes_df = pd.read_csv(node_file_path).fillna("")

        # Checking if all required fields are in GMNS links and nodes files

        for field in p.parameters["network"]["gmns"]["required_node_fields"]:
            if field not in gmns_nodes_df.columns.to_list():
                raise ValueError(f"In GMNS nodes file: field '{field}' required, but not found.")

        for field in p.parameters["network"]["gmns"]["required_link_fields"]:
            if field not in gmns_links_df.columns.to_list():
                raise ValueError(f"In GMNS links file: field '{field}' required, but not found.")

        if gmns_geometry_field not in gmns_links_df.columns.to_list():
            if geometry_path is None:
                raise ValueError(
                    "To create an aequilibrae links table, geometries information must be provided either in the GMNS link table or in a separate file ('geometry_path' attribute)."
                )
            else:
                geometry_df = pd.read_csv(geometry_path)
                gmns_links_df = gmns_links_df.merge(geometry_df, on="geometry_id", how="left")

        # Checking if it is needed to change the spatial reference system.

        if srid != 4326:
            transformer = Transformer.from_crs(f"epsg:{srid}", "epsg:4326", always_xy=True)

            # For node table
            lons, lats = transformer.transform(gmns_nodes_df.loc[:, "x_coord"], gmns_nodes_df.loc[:, "y_coord"])
            gmns_nodes_df.loc[:, "x_coord"] = np.around(lons, decimals=10)
            gmns_nodes_df.loc[:, "y_coord"] = np.around(lats, decimals=10)

            # For link table
            for idx, row in gmns_links_df.iterrows():
                geom = shapely.wkt.loads(row.geometry)
                x_points = [int(x[0]) for x in list(geom.coords)]
                y_points = [int(x[1]) for x in list(geom.coords)]

                lons, lats = transformer.transform(x_points, y_points)
                new_points = list(zip(np.around(lons, decimals=10), np.around(lats, decimals=10)))

                gmns_links_df.loc[idx, "geometry"] = LineString(new_points).wkt

        # Creating direction list based on list of two-way links
        # Editing gmns direction field so it contains information in the AequilibraE standard (0 value for two-way links)

        if gmns_direction_field in gmns_links_df.columns.to_list():

            gmns_links_df[gmns_direction_field] = [1 if x not in [-1, 1] else x for x in list(gmns_links_df[gmns_direction_field])]
            sorted_df = gmns_links_df.sort_values("link_id")

            idx_ = (sorted_df[gmns_direction_field] == -1)
            sorted_df.loc[idx_, ["from_node_id", "to_node_id"]] = sorted_df.loc[idx_, ["to_node_id", "from_node_id"]].values
            sorted_df.loc[idx_, gmns_direction_field] = 1

            to_drop_lst = []
            for idx, row in sorted_df.iterrows():

                same_dir_df = sorted_df[(sorted_df[["from_node_id", "to_node_id"]].apply(tuple, 1) == (row["from_node_id"], row["to_node_id"])) & (sorted_df.link_id > row.link_id)]
                if same_dir_df.shape[0] > 0:
                    if gmns_lanes_field in sorted_df.columns.to_list():
                        gmns_links_df.at[idx, gmns_lanes_field] = row[gmns_lanes_field] + same_dir_df[gmns_lanes_field].sum()

                    if gmns_capacity_field in sorted_df.columns.to_list():
                        gmns_links_df.at[idx, gmns_capacity_field] = row[gmns_capacity_field] + same_dir_df[gmns_capacity_field].sum()

                opp_dir_df = sorted_df[(sorted_df[["from_node_id", "to_node_id"]].apply(tuple, 1) == (row["to_node_id"], row["from_node_id"])) & (sorted_df.link_id > row.link_id)]
                if opp_dir_df.shape[0] > 0:
                    gmns_links_df.at[idx, gmns_direction_field] = 0

                to_drop_lst += list(same_dir_df.index) + list(opp_dir_df.index)

            sorted_df.drop(to_drop_lst, axis=0, inplace=True)
            gmns_links_df = gmns_links_df[gmns_links_df.link_id.isin(list(sorted_df.link_id))]
            gmns_links_df.reset_index(drop=True, inplace=True)
            direction = [x for x in list(gmns_links_df[gmns_direction_field])]

        else:
            direction = [1 for _ in range(len(gmns_links_df))]
            logger.info(
                f"All directions were considered equal to 1 (a_node to b_node) because the field '{gmns_direction_field}' has not been found in the GMNS link table."
            )

        # Adding new fields to AequilibraE links table / Preparing it to receive information from GMNS table.

        l_fields = self.links.fields
        l_fields.add("notes", description="More information about the link", data_type="TEXT")

        if gmns_lanes_field in gmns_links_df.columns.to_list():
            l_fields.add("lanes_ab", description="Lanes", data_type="NUMERIC")
            l_fields.add("lanes_ba", description="Lanes", data_type="NUMERIC")

        other_ldict = {}
        for fld in list(other_lfields.keys()):
            if fld in gmns_links_df.columns.to_list() and fld not in l_fields.all_fields():
                l_fields.add(
                    f"{fld}_gmns", description=f"{fld} field from GMNS link table", data_type=f"{other_lfields[fld]}"
                )
                other_ldict.update({f"{fld}_gmns": gmns_links_df[fld]})

        l_fields.save()

        # Adding new fields to AequilibraE nodes table / Preparing it to receive information from GMNS table.

        n_fields = self.nodes.fields
        n_fields.add("notes", description="More information about the node", data_type="TEXT")

        other_ndict = {}
        for fld in list(other_nfields.keys()):
            if fld in gmns_nodes_df.columns.to_list() and fld not in l_fields.all_fields():
                n_fields.add(
                    f"{fld}_gmns", description=f"{fld} field from GMNS node table", data_type=f"{other_nfields[fld]}"
                )
                other_ndict.update({f"{fld}_gmns": gmns_nodes_df[fld]})

        n_fields.save()

        # Creating speeds, capacities and lanes lists based on direction list

        speed_ab = ["" for _ in range(len(gmns_links_df))]
        speed_ba = ["" for _ in range(len(gmns_links_df))]
        capacity_ab = ["" for _ in range(len(gmns_links_df))]
        capacity_ba = ["" for _ in range(len(gmns_links_df))]
        lanes_ab = ["" for _ in range(len(gmns_links_df))]
        lanes_ba = ["" for _ in range(len(gmns_links_df))]

        for idx, row in gmns_links_df.iterrows():
            if gmns_speed_field in gmns_links_df.columns.to_list():
                if direction[idx] == 1:
                    speed_ab[idx] = row[gmns_speed_field]
                elif direction[idx] == -1:
                    speed_ba[idx] = row[gmns_speed_field]
                else:
                    speed_ab[idx] = row[gmns_speed_field]
                    speed_ba[idx] = row[gmns_speed_field]

            if gmns_capacity_field in gmns_links_df.columns.to_list():
                if direction[idx] == 1:
                    capacity_ab[idx] = row[gmns_capacity_field]
                elif direction[idx] == -1:
                    capacity_ba[idx] = row[gmns_capacity_field]
                else:
                    capacity_ab[idx] = row[gmns_capacity_field]
                    capacity_ba[idx] = row[gmns_capacity_field]

            if gmns_lanes_field in gmns_links_df.columns.to_list():
                if direction[idx] == 1:
                    lanes_ab[idx] = row[gmns_lanes_field]
                elif direction[idx] == -1:
                    lanes_ba[idx] = row[gmns_lanes_field]
                else:
                    lanes_ab[idx] = row[gmns_lanes_field]
                    lanes_ba[idx] = row[gmns_lanes_field]

        # Getting information from some optinal GMNS fields

        if gmns_name_field in gmns_links_df.columns.to_list():
            name_list = gmns_links_df[gmns_name_field].to_list()
        else:
            name_list = ["" for _ in range(len(gmns_links_df))]

        # Creating link_type list
        # Setting link_type = 'unclassified' if there is no information about it in the GMNS links table

        if gmns_link_type_field not in gmns_links_df.columns.to_list():
            gmns_link_type_field = "link_type_name"
            if gmns_link_type_field not in gmns_links_df.columns.to_list():
                link_types_list = ["unclassified" for _ in range(len(gmns_links_df))]
            else:
                link_types_list = gmns_links_df[gmns_link_type_field].to_list()
        else:
            link_types_list = gmns_links_df[gmns_link_type_field].to_list()

        ## Adding link_types to AequilibraE model

        link_types_list = [s.replace("-", "_") for s in link_types_list]
        for lt_name in list(dict.fromkeys(link_types_list)):

            all_types = list(self.link_types.all_types())
            letters = lt_name.lower() + lt_name.upper() + string.ascii_letters
            letters = "".join([lt for lt in letters if lt not in all_types])

            link_types = self.link_types
            new_type = link_types.new(letters[0])
            new_type.link_type = lt_name
            new_type.description = "Link type from GMNS link table"
            new_type.save()

        # Creating modes list and saving modes to AequilibraE model

        if gmns_modes_field in gmns_links_df.columns.to_list():
            modes_list = gmns_links_df[gmns_modes_field].to_list()
            if "" in modes_list:
                modes_list = ["unspecified_mode" if x == "" else x for x in modes_list]

        else:
            modes_list = ["unspecified_mode" for _ in range(len(gmns_links_df))]

        if use_group_path is not None:

            use_group = pd.read_csv(use_group_path)
            groups_dict = dict(zip(use_group.use_group, use_group.uses))
            for k, use in groups_dict.items():
                for group in list(groups_dict.keys()):
                    if group in use:
                        groups_dict[k] = use.replace(group, groups_dict[group])
        else:
            groups_dict = {}

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
        pattern = re.compile("|".join(char_replaces.keys()))
        modes_list = [
            pattern.sub(lambda x: char_replaces[x.group()], s).replace("+", "").replace("-", "_") for s in modes_list
        ]
        mode_ids_list = [x for x in modes_list]

        for mode in list(dict.fromkeys(modes_list)):

            if mode in groups_dict.keys():
                modes_gathered = [
                    m.replace("+", "").replace("-", "_").replace("2", "two").replace("3", "three").replace(" ", "")
                    for m in groups_dict[mode].split(sep=",")
                ]
                desc = use_group.loc[use_group.use_group == mode, "description"].item() + f" - GMNS use group: {mode}"

            else:
                modes_gathered = [m.replace("+", "").replace("-", "_").replace(" ", "") for m in mode.split(sep=",")]
                if mode == "unspecified_mode":
                    desc = "Mode not specified"
                else:
                    desc = "Mode from GMNS link table"

            mode_to_add = ""
            for m in modes_gathered:
                letters = m.lower() + m.upper() + string.ascii_letters
                all_modes = list(self.modes.all_modes())
                letters = "".join([lt for lt in letters if lt not in all_modes and lt != "_"])

                if m in [list(x.keys())[0] for x in p.parameters["network"]["modes"]]:
                    m += "_gmns"

                modes = self.modes
                new_mode = modes.new(letters[0])
                new_mode.mode_name = m
                modes.add(new_mode)
                new_mode.description = desc
                new_mode.save()

                mode_to_add += letters[0]

            mode_ids_list = [mode_to_add if x == mode else x for x in mode_ids_list]

        # Checking if the links boundaries coordinates match the "from" and "to" nodes coordinates

        critical_dist = p.parameters["network"]["gmns"]["critical_dist"]

        for idx, row in gmns_links_df.iterrows():

            from_point_x = gmns_nodes_df.loc[
                gmns_nodes_df[gmns_node_id_field] == row[gmns_a_node_field], "x_coord"
            ].item()
            from_point_y = gmns_nodes_df.loc[
                gmns_nodes_df[gmns_node_id_field] == row[gmns_a_node_field], "y_coord"
            ].item()
            to_point_x = gmns_nodes_df.loc[
                gmns_nodes_df[gmns_node_id_field] == row[gmns_b_node_field], "x_coord"
            ].item()
            to_point_y = gmns_nodes_df.loc[
                gmns_nodes_df[gmns_node_id_field] == row[gmns_b_node_field], "y_coord"
            ].item()

            link_geom = shapely.wkt.loads(row.geometry)
            link_points = list(link_geom.coords)
            link_start_boundary = link_points[0]
            link_end_boundary = link_points[-1]

            if link_start_boundary != (from_point_x, from_point_y):
                start_to_from_dist = Point(link_start_boundary).distance(Point(from_point_x, from_point_y))

                if start_to_from_dist <= critical_dist:
                    link_points = [(from_point_x, from_point_y)] + link_points[1:]

                else:
                    link_points = [(from_point_x, from_point_y)] + link_points[:]

                new_link = LineString(link_points)
                gmns_links_df.loc[idx, "geometry"] = new_link.wkt
                logger.info(
                    f"Geometry from link whose link_id = {row[gmns_link_id_field]} has just been corrected. It was not connected to its start node."
                )

            if link_end_boundary != (to_point_x, to_point_y):
                end_to_to_dist = Point(link_end_boundary).distance(Point(to_point_x, to_point_y))

                if end_to_to_dist <= critical_dist:
                    link_points = link_points[:-1] + [(to_point_x, to_point_y)]

                else:
                    link_points = link_points[:] + [(to_point_x, to_point_y)]

                new_link = LineString(link_points)
                gmns_links_df.loc[idx, "geometry"] = new_link.wkt
                logger.info(
                    f"Geometry from link whose link_id = {row[gmns_link_id_field]} has just been corrected. It was not connected to its end node."
                )

        # Setting centroid equals 1 when informed in the 'node_type' node table field

        if "node_type" in gmns_nodes_df.columns.to_list():
            centroid_flag = [1 if x == "centroid" else 0 for x in gmns_nodes_df["node_type"].to_list()]
        else:
            centroid_flag = 0
        # Creating dataframes for adding nodes and links information to AequilibraE model

        nodes_dict = {
            "node_id": gmns_nodes_df[gmns_node_id_field],
            "is_centroid": centroid_flag,
            "x_coord": gmns_nodes_df.x_coord,
            "y_coord": gmns_nodes_df.y_coord,
            "notes": "from GMNS file",
        }
        nodes_dict.update(other_ndict)
        aeq_nodes_df = pd.DataFrame(nodes_dict)

        links_dict = {
            "link_id": gmns_links_df[gmns_link_id_field],
            "a_node": gmns_links_df[gmns_a_node_field],
            "b_node": gmns_links_df[gmns_b_node_field],
            "direction": direction,
            "modes": mode_ids_list,
            "link_type": link_types_list,
            "name": name_list,
            "speed_ab": speed_ab,
            "speed_ba": speed_ba,
            "capacity_ab": capacity_ab,
            "capacity_ba": capacity_ba,
            "geometry": gmns_links_df.geometry,
            "lanes_ab": lanes_ab,
            "lanes_ba": lanes_ba,
            "notes": "from GMNS file",
        }
        links_dict.update(other_ldict)
        aeq_links_df = pd.DataFrame(links_dict)

        return aeq_nodes_df, aeq_links_df, nodes_dict, links_dict
