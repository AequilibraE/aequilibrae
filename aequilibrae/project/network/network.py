import math
import os
from warnings import warn
from sqlite3 import Connection as sqlc
from typing import List
import numpy as np
from aequilibrae.project.network import OSMDownloader
from aequilibrae.project.network.osm_builder import OSMBuilder
from aequilibrae.project.network.osm_utils.place_getter import placegetter
from aequilibrae.project.network.osm_utils.osm_params import max_query_area_size
from aequilibrae.project.network.haversine import haversine
from aequilibrae.paths import Graph
from aequilibrae.parameters import Parameters
from aequilibrae import logger

from ...utils import WorkerThread


class Network(WorkerThread):
    def __init__(self, project):
        WorkerThread.__init__(self, None)

        self.conn = project.conn  # type: sqlc
        self.graphs = {}

    def _check_if_exists(self):
        curr = self.conn.cursor()
        curr.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='links';")
        tbls = curr.fetchone()[0]
        if tbls > 0:
            return True
        return False

    def modes(self):
        curr = self.conn.cursor()
        curr.execute("""select mode_name from modes""")
        return [x[0] for x in curr.fetchall()]

    def create_from_osm(
        self,
        west: float = None,
        south: float = None,
        east: float = None,
        north: float = None,
        place_name: str = None,
        modes=["car", "transit", "bicycle", "walk"],
        spatial_index=False,
    ) -> None:

        if self._check_if_exists():
            raise FileExistsError("You can only import an OSM network into a brand new model file")

        self.create_empty_tables()

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
                msg = 'We could not find a reference for place name "{}"'.format(place_name)
                warn(msg)
                logger.warn(msg)
                return
            for i in report:
                if "PLACE FOUND" in i:
                    logger.info(i)

        # Need to compute the size of the bounding box to not exceed it too much
        height = haversine((east + west) / 2, south, (east + west) / 2, north)
        width = haversine(east, (north + south) / 2, west, (north + south) / 2)
        area = height * width

        if area < max_query_area_size:
            polygons = [bbox]
        else:
            polygons = []
            parts = math.ceil(area / max_query_area_size)
            horizontal = math.ceil(math.sqrt(parts))
            vertical = math.ceil(parts / horizontal)
            dx = east - west
            dy = north - south
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
        self.downloader.doWork()

        logger.info("Building Network")
        self.builder = OSMBuilder(self.downloader.json, self.conn)
        self.builder.doWork()

        if spatial_index:
            logger.info("Adding spatial indices")
            self.add_spatial_index()

        self.add_triggers()
        logger.info("Network built successfully")

    def create_empty_tables(self) -> None:
        curr = self.conn.cursor()
        # Create the links table
        p = Parameters()
        fields = p.parameters["network"]["links"]["fields"]

        mandatory = ["LINK_ID", "A_NODE", "B_NODE", "DIRECTION", "DISTANCE", "MODES", "LINK_TYPE"]
        sql = """CREATE TABLE 'links' (
                          ogc_fid INTEGER PRIMARY KEY,
                          link_id INTEGER UNIQUE NOT NULL,
                          a_node INTEGER NOT NULL,
                          b_node INTEGER NOT NULL,
                          direction INTEGER NOT NULL DEFAULT 0,
                          distance NUMERIC NOT NULL,
                          modes TEXT NOT NULL,
                          link_type TEXT NOT NULL DEFAULT 'link type not defined'
                          {});"""

        flds = fields["one-way"]

        owlf = [
            "{} {}".format(list(f.keys())[0], f[list(f.keys())[0]]["type"])
            for f in flds
            if list(f.keys())[0].upper() not in mandatory
        ]

        flds = fields["two-way"]
        twlf = []
        for f in flds:
            nm = list(f.keys())[0]
            tp = f[nm]["type"]
            twlf.extend(["{}_ab {}".format(nm, tp), "{}_ba {}".format(nm, tp)])

        link_fields = owlf + twlf

        if link_fields:
            sql = sql.format("," + ",".join(link_fields))
        else:
            sql = sql.format("")

        curr.execute(sql)

        sql = """CREATE TABLE 'nodes' (ogc_fid INTEGER PRIMARY KEY,
                                 node_id INTEGER UNIQUE NOT NULL,
                                 is_centroid INTEGER NOT NULL DEFAULT 0 {});"""

        flds = p.parameters["network"]["nodes"]["fields"]

        default_fields = ["NODE_ID", "IS_CENTROID"]
        ndflds = [
            "{} {}".format(list(f.keys())[0], f[list(f.keys())[0]]["type"])
            for f in flds
            if list(f.keys())[0].upper() not in default_fields
        ]

        if ndflds:
            sql = sql.format("," + ",".join(ndflds))
        else:
            sql = sql.format("")
        curr.execute(sql)

        curr.execute("""SELECT AddGeometryColumn( 'links', 'geometry', 4326, 'LINESTRING', 'XY' )""")
        curr.execute("""SELECT AddGeometryColumn( 'nodes', 'geometry', 4326, 'POINT', 'XY' )""")
        self.conn.commit()

    def build_graphs(self, modes: List[str], fields=["distance"]) -> None:
        # curr.execute("select * from links where link_id < 0")
        for mode in modes:
            print(mode)

    def custom_graph(self, mode: str, centroids: np.array, fields=["distance"]) -> Graph:
        curr = self.conn.cursor()
        curr.execute("select * from links where link_id < 0")
        available_fields = [x[0] for x in curr.description if x[0] not in ["ogc_fid", "geometry"]]

        required_fields = ["link_id", "a_node", "b_node", "direction"]
        for f in fields:
            if f in available_fields:
                required_fields.append(f)
            else:
                if "{}_ab".format(f) not in available_fields or "{}_ba".format(f) not in available_fields:
                    raise ValueError("Field {} does not exist on your network".format(f))
                required_fields.extend(["{}_ab".format(f), "{}_ba".format(f)])

        curr.execute("select {} from links where  instr(modes, '{}') > 0;".format(",".join(required_fields), mode))

        # data = curr.fetchall()

        g = Graph()

        return g

    def count_links(self) -> int:
        c = self.conn.cursor()
        c.execute("""select count(*) from links""")
        return c.fetchone()[0]

    def count_nodes(self) -> int:
        c = self.conn.cursor()
        c.execute("""select count(*) from nodes""")
        return c.fetchone()[0]

    def add_triggers(self):
        self.__add_network_triggers()
        self.__add_mode_triggers()

    def __add_network_triggers(self) -> None:
        logger.info("Adding network triggers")
        pth = os.path.dirname(os.path.realpath(__file__))
        qry_file = os.path.join(pth, "database_triggers", "network_triggers.sql")
        self.__add_trigger_from_file(qry_file)

    def __add_mode_triggers(self) -> None:
        logger.info("Adding mode table triggers")
        pth = os.path.dirname(os.path.realpath(__file__))
        qry_file = os.path.join(pth, "database_triggers", "modes_table_triggers.sql")
        self.__add_trigger_from_file(qry_file)

    def __add_trigger_from_file(self, qry_file: str):
        curr = self.conn.cursor()
        sql_file = open(qry_file, "r")
        query_list = sql_file.read()
        sql_file.close()

        # Run one query/command at a time
        for cmd in query_list.split("#"):
            try:
                curr.execute(cmd)
            except Exception as e:
                msg = "Error creating trigger: {}".format(e.args)
                logger.error(msg)
                logger.info(cmd)
        self.conn.commit()

    def add_spatial_index(self) -> None:
        curr = self.conn.cursor()
        curr.execute("""SELECT CreateSpatialIndex( 'links' , 'geometry' );""")
        curr.execute("""SELECT CreateSpatialIndex( 'nodes' , 'geometry' );""")
        self.conn.commit()
