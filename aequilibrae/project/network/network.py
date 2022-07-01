import importlib.util as iutil
import math
import re
from re import A
from sqlite3 import Connection as sqlc
from tracemalloc import start
from typing import Dict
from unicodedata import decimal
from warnings import warn

import numpy as np
import pandas as pd
import string
import shapely.wkb
import shapely.wkt
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
from pyproj import Transformer

from aequilibrae import logger
from aequilibrae.parameters import Parameters
from aequilibrae.project.database_connection import database_connection
from aequilibrae.project.network import OSMDownloader
from aequilibrae.project.network.haversine import haversine
from aequilibrae.project.network.link_types import LinkTypes
from aequilibrae.project.network.links import Links
from aequilibrae.project.network.modes import Modes
from aequilibrae.project.network.nodes import Nodes
from aequilibrae.project.network.osm_builder import OSMBuilder
from aequilibrae.project.network.osm_utils.place_getter import placegetter
from aequilibrae.project.project_creation import req_link_flds, req_node_flds, protected_fields
from aequilibrae.utils import WorkerThread

spec = iutil.find_spec("PyQt5")
pyqt = spec is not None
if pyqt:
    from PyQt5.QtCore import pyqtSignal as SIGNAL


class Network(WorkerThread):
    """
    Network class. Member of an AequilibraE Project
    """

    if pyqt:
        netsignal = SIGNAL(object)

    req_link_flds = req_link_flds
    req_node_flds = req_node_flds
    protected_fields = protected_fields
    link_types: LinkTypes = None

    def __init__(self, project) -> None:
        from aequilibrae.paths import Graph

        WorkerThread.__init__(self, None)
        self.conn = project.conn  # type: sqlc
        self.source = project.source  # type: sqlc
        self.graphs = {}  # type: Dict[Graph]
        self.modes = Modes(self)
        self.link_types = LinkTypes(self)
        self.links = Links()
        self.nodes = Nodes()

    def skimmable_fields(self):
        """
        Returns a list of all fields that can be skimmed

        Returns:
            :obj:`list`: List of all fields that can be skimmed
        """
        curr = self.conn.cursor()
        curr.execute("PRAGMA table_info(links);")
        field_names = curr.fetchall()
        ignore_fields = ["ogc_fid", "geometry"] + self.req_link_flds

        skimmable = [
            "INT",
            "INTEGER",
            "TINYINT",
            "SMALLINT",
            "MEDIUMINT",
            "BIGINT",
            "UNSIGNED BIG INT",
            "INT2",
            "INT8",
            "REAL",
            "DOUBLE",
            "DOUBLE PRECISION",
            "FLOAT",
            "DECIMAL",
            "NUMERIC",
        ]
        all_fields = []

        for f in field_names:
            if f[1] in ignore_fields:
                continue
            for i in skimmable:
                if i in f[2].upper():
                    all_fields.append(f[1])
                    break

        all_fields.append("distance")
        real_fields = []
        for f in all_fields:
            if f[-2:] == "ab":
                if f[:-2] + "ba" in all_fields:
                    real_fields.append(f[:-3])
            elif f[-3:] != "_ba":
                real_fields.append(f)

        return real_fields

    def list_modes(self):
        """
        Returns a list of all the modes in this model

        Returns:
            :obj:`list`: List of all modes
        """
        curr = self.conn.cursor()
        curr.execute("""select mode_id from modes""")
        return [x[0] for x in curr.fetchall()]

    def create_from_osm(
        self,
        west: float = None,
        south: float = None,
        east: float = None,
        north: float = None,
        place_name: str = None,
        modes=["car", "transit", "bicycle", "walk"],
    ) -> None:
        """
        Downloads the network from Open-Street Maps

        Args:
            *west* (:obj:`float`, Optional): West most coordinate of the download bounding box

            *south* (:obj:`float`, Optional): South most coordinate of the download bounding box

            *east* (:obj:`float`, Optional): East most coordinate of the download bounding box

            *place_name* (:obj:`str`, Optional): If not downloading with East-West-North-South boundingbox, this is
            required

            *modes* (:obj:`list`, Optional): List of all modes to be downloaded. Defaults to the modes in the parameter
            file

            p = Project()
            p.new(nm)

        ::

            from aequilibrae import Project, Parameters
            p = Project()
            p.new('path/to/project')

            # We now choose a different overpass endpoint (say a deployment in your local network)
            par = Parameters()
            par.parameters['osm']['overpass_endpoint'] = "http://192.168.1.234:5678/api"

            # Because we have our own server, we can set a bigger area for download (in M2)
            par.parameters['osm']['max_query_area_size'] = 10000000000

            # And have no pause between successive queries
            par.parameters['osm']['sleeptime'] = 0

            # Save the parameters to disk
            par.write_back()

            # And do the import
            p.network.create_from_osm(place_name=my_beautiful_hometown)
            p.close()
        """

        if self.count_links() > 0:
            raise FileExistsError("You can only import an OSM network into a brand new model file")

        curr = self.conn.cursor()
        curr.execute("""ALTER TABLE links ADD COLUMN osm_id integer""")
        curr.execute("""ALTER TABLE nodes ADD COLUMN osm_id integer""")
        self.conn.commit()

        if isinstance(modes, (tuple, list)):
            modes = list(modes)
        elif isinstance(modes, str):
            modes = [modes]
        else:
            raise ValueError("'modes' needs to be string or list/tuple of string")

        if place_name is None:
            if min(east, west) < -180 or max(east, west) > 180 or min(north, south) < -90 or max(north, south) > 90:
                raise ValueError("Coordinates out of bounds")
            bbox = [west, south, east, north]
        else:
            bbox, report = placegetter(place_name)
            west, south, east, north = bbox
            if bbox is None:
                msg = f'We could not find a reference for place name "{place_name}"'
                warn(msg)
                logger.warning(msg)
                return
            for i in report:
                if "PLACE FOUND" in i:
                    logger.info(i)

        # Need to compute the size of the bounding box to not exceed it too much
        height = haversine((east + west) / 2, south, (east + west) / 2, north)
        width = haversine(east, (north + south) / 2, west, (north + south) / 2)
        area = height * width

        par = Parameters().parameters["osm"]
        max_query_area_size = par["max_query_area_size"]

        if area < max_query_area_size:
            polygons = [bbox]
        else:
            polygons = []
            parts = math.ceil(area / max_query_area_size)
            horizontal = math.ceil(math.sqrt(parts))
            vertical = math.ceil(parts / horizontal)
            dx = (east - west) / horizontal
            dy = (north - south) / vertical
            for i in range(horizontal):
                xmin = max(-180, west + i * dx)
                xmax = min(180, west + (i + 1) * dx)
                for j in range(vertical):
                    ymin = max(-90, south + j * dy)
                    ymax = min(90, south + (j + 1) * dy)
                    box = [xmin, ymin, xmax, ymax]
                    polygons.append(box)
        logger.info("Downloading data")
        self.downloader = OSMDownloader(polygons, modes)
        if pyqt:
            self.downloader.downloading.connect(self.signal_handler)

        self.downloader.doWork()

        logger.info("Building Network")
        self.builder = OSMBuilder(self.downloader.json, self.source)
        if pyqt:
            self.builder.building.connect(self.signal_handler)
        self.builder.doWork()

        logger.info("Network built successfully")

    def create_from_gmns(
        self,
        link_file_path: str,
        node_file_path: str,
        use_group_path: str = None,
        geometry_path: str = None,
        srid: int = 4326,
    ) -> None:
        """
        Creates AequilibraE model from links and nodes in GMNS format.

        Args:
            *link_file_path* (:obj:`str`): Path to a links csv file in GMNS format

            *node_file_path* (:obj:`str`): Path to a nodes csv file in GMNS format

            *use_group_path* (:obj:`str`): Optional argument. Path to a csv table containing groupings of uses. This helps AequilibraE
            know when a GMNS use is actually a group of other GMNS uses

            *geometry_path* (:obj:`str`): Optional argument. Path to a csv file containing geometry information for a line object, if not
            specified in the link table
        """

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

        # Getting list of two-way links

        gmns_links_df[["from_node_id", "to_node_id"]] = np.sort(gmns_links_df[["from_node_id", "to_node_id"]], axis=1)
        df_count = gmns_links_df.groupby(["from_node_id", "to_node_id"], as_index=False).count()

        df_two_way_count = df_count[df_count.link_id >= 2]
        if df_two_way_count.shape[0] > 0:
            two_way_nodes = list(
                zip(df_two_way_count.from_node_id, df_two_way_count.to_node_id)
            )  # list not in the correct from-to order
            two_way_df = gmns_links_df[
                gmns_links_df[["from_node_id", "to_node_id"]].apply(tuple, 1).isin(two_way_nodes)
            ]  # dataframe in the correct from-to order
            two_way_links = (
                two_way_df.sort_values("link_id")
                .drop_duplicates(subset=["from_node_id", "to_node_id"], keep="first")
                .link_id.to_list()
            )  # list in the correct from-to order

            gmns_links_df = gmns_links_df.sort_values("link_id").drop_duplicates(
                subset=["from_node_id", "to_node_id"], keep="first"
            )
            gmns_links_df.reset_index(drop=True, inplace=True)

            two_way_indices = gmns_links_df.index[gmns_links_df.link_id.isin(two_way_links)].to_list()

        else:
            two_way_indices = []

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

        # Creating direction list based on list of two-way links

        if gmns_direction_field not in gmns_links_df.columns.to_list():
            direction = [1 for _ in range(len(gmns_links_df))]

        else:
            ## Assuming direction from 'from_node_id' to 'to_node_id' (direction=1) in case there is no information about it
            direction = [1 if x not in [-1, 1] else x for x in gmns_links_df[gmns_direction_field].to_list()]

        if two_way_indices != []:
            for idx in two_way_indices:
                direction[idx] = 0

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

            letters = lt_name.lower() + lt_name.upper() + string.ascii_letters
            letters = "".join([lt for lt in letters if lt not in list(self.link_types.all_types())])

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
                letters = "".join([lt for lt in letters if lt not in self.list_modes() and lt != "_"])

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

        nodes_fields_list = list(nodes_dict.keys())
        nodes_fields_list.pop(nodes_fields_list.index("y_coord"))
        nodes_fields_list = ["geometry" if x == "x_coord" else x for x in nodes_fields_list]

        n_query = "insert into nodes(" + ", ".join(nodes_fields_list) + ")"
        n_query += (
            " values("
            + ", ".join(["MakePoint(?,?, 4326)" if x == "geometry" else "?" for x in nodes_fields_list])
            + ")"
        )
        n_params_list = aeq_nodes_df.to_records(index=False)

        self.conn.executemany(n_query, n_params_list)
        self.conn.commit()

        l_query = "insert into links(" + ", ".join(list(links_dict.keys())) + ")"
        l_query += (
            " values("
            + ", ".join(["GeomFromTEXT(?,4326)" if x == "geometry" else "?" for x in list(links_dict.keys())])
            + ")"
        )
        l_params_list = aeq_links_df.to_records(index=False)

        self.conn.executemany(l_query, l_params_list)
        self.conn.commit()

        logger.info("Network built successfully")

    def signal_handler(self, val):
        if pyqt:
            self.netsignal.emit(val)

    def build_graphs(self, fields: list = None, modes: list = None) -> None:
        """Builds graphs for all modes currently available in the model

        When called, it overwrites all graphs previously created and stored in the networks'
        dictionary of graphs

        Args:
            *fields* (:obj:`list`, optional): When working with very large graphs with large number of fields in the
                                              database, it may be useful to specify which fields to use
            *modes* (:obj:`list`, optional): When working with very large graphs with large number of fields in the
                                              database, it may be useful to generate only those we need

        To use the *fields* parameter, a minimalistic option is the following
        ::

            p = Project()
            p.open(nm)
            fields = ['distance']
            p.network.build_graphs(fields, modes = ['c', 'w'])

        """
        from aequilibrae.paths import Graph

        curr = self.conn.cursor()

        if fields is None:
            curr.execute("PRAGMA table_info(links);")
            field_names = curr.fetchall()

            ignore_fields = ["ogc_fid", "geometry"]
            all_fields = [f[1] for f in field_names if f[1] not in ignore_fields]
        else:
            fields.extend(["link_id", "a_node", "b_node", "direction", "modes"])
            all_fields = list(set(fields))

        if modes is None:
            modes = curr.execute("select mode_id from modes;").fetchall()
            modes = [m[0] for m in modes]
        elif isinstance(modes, str):
            modes = [modes]

        sql = f"select {','.join(all_fields)} from links"

        df = pd.read_sql(sql, self.conn).fillna(value=np.nan)
        valid_fields = list(df.select_dtypes(np.number).columns) + ["modes"]
        curr.execute("select node_id from nodes where is_centroid=1 order by node_id;")
        centroids = np.array([i[0] for i in curr.fetchall()], np.uint32)

        data = df[valid_fields]
        for m in modes:
            net = pd.DataFrame(data, copy=True)
            net.loc[~net.modes.str.contains(m), "b_node"] = net.loc[~net.modes.str.contains(m), "a_node"]
            g = Graph()
            g.mode = m
            g.network = net
            g.prepare_graph(centroids)
            g.set_blocked_centroid_flows(True)
            self.graphs[m] = g

    def set_time_field(self, time_field: str) -> None:
        """
        Set the time field for all graphs built in the model

        Args:
            *time_field* (:obj:`str`): Network field with travel time information
        """
        for m, g in self.graphs.items():
            if time_field not in list(g.graph.columns):
                raise ValueError(f"{time_field} not available. Check if you have NULL values in the database")
            g.free_flow_time = time_field
            g.set_graph(time_field)
            self.graphs[m] = g

    def count_links(self) -> int:
        """
        Returns the number of links in the model

        Returns:
            :obj:`int`: Number of links
        """
        return self.__count_items("link_id", "links", "link_id>=0")

    def count_centroids(self) -> int:
        """
        Returns the number of centroids in the model

        Returns:
            :obj:`int`: Number of centroids
        """
        return self.__count_items("node_id", "nodes", "is_centroid=1")

    def count_nodes(self) -> int:
        """
        Returns the number of nodes in the model

        Returns:
            :obj:`int`: Number of nodes
        """
        return self.__count_items("node_id", "nodes", "node_id>=0")

    def extent(self):
        """Queries the extent of the network included in the model

        Returns:
            *model extent* (:obj:`Polygon`): Shapely polygon with the bounding box of the model network.
        """
        curr = self.conn.cursor()
        curr.execute('Select ST_asBinary(GetLayerExtent("Links"))')
        poly = shapely.wkb.loads(curr.fetchone()[0])
        return poly

    def convex_hull(self) -> Polygon:
        """Queries the model for the convex hull of the entire network

        Returns:
            *model coverage* (:obj:`Polygon`): Shapely (Multi)polygon of the model network.
        """
        curr = self.conn.cursor()
        curr.execute('Select ST_asBinary("geometry") from Links where ST_Length("geometry") > 0;')
        links = [shapely.wkb.loads(x[0]) for x in curr.fetchall()]
        return unary_union(links).convex_hull

    def refresh_connection(self):
        """Opens a new database connection to avoid thread conflict"""
        self.conn = database_connection()

    def __count_items(self, field: str, table: str, condition: str) -> int:
        c = self.conn.execute(f"select count({field}) from {table} where {condition};").fetchone()[0]
        return c
