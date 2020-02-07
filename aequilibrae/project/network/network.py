import math
import os
from warnings import warn
from sqlite3 import Connection as sqlc
from aequilibrae.project.network import OSMDownloader
from aequilibrae.project.network.osm_builder import OSMBuilder
from aequilibrae.project.network.osm_utils.place_getter import placegetter
from aequilibrae.project.network.osm_utils.osm_params import max_query_area_size
from aequilibrae.project.network.haversine import haversine
from aequilibrae.parameters import Parameters
from aequilibrae import logger

from ...utils import WorkerThread


class Network(WorkerThread):
    def __init__(self, project):
        WorkerThread.__init__(self, None)

        self.conn = project.conn  # type: sqlc

    def _check_if_exists(self):
        curr = self.conn.cursor()
        curr.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='links';")
        tbls = curr.fetchone()[0]
        return tbls > 0

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
                msg = f'We could not find a reference for place name "{place_name}"'
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

        self.add_network_triggers()
        logger.info("Network built successfully")

    def create_empty_tables(self) -> None:
        curr = self.conn.cursor()
        # Create the links table
        p = Parameters()
        fields = p.parameters["network"]["links"]["fields"]

        sql = """CREATE TABLE 'links' (
                          ogc_fid INTEGER PRIMARY KEY,
                          link_id INTEGER UNIQUE NOT NULL,
                          {});"""

        flds = fields["one-way"]

        # returns first key in the dictionary
        def fkey(f):
            return list(f.keys())[0]

        owlf = ["{} {}".format(fkey(f), f[fkey(f)]["type"]) for f in flds if fkey(f).upper() != "LINK_ID"]

        flds = fields["two-way"]
        twlf = []
        for f in flds:
            nm = fkey(f)
            tp = f[nm]["type"]
            twlf.extend([f"{nm}_ab {tp}", f"{nm}_ba {tp}"])

        link_fields = owlf + twlf

        sql = sql.format(",".join(link_fields))
        curr.execute(sql)

        sql = """CREATE TABLE 'nodes' (ogc_fid INTEGER PRIMARY KEY,
                                 node_id INTEGER UNIQUE NOT NULL, {});"""

        flds = p.parameters["network"]["nodes"]["fields"]

        ndflds = ["{} {}".format(fkey(f), f[fkey(f)]["type"]) for f in flds if fkey(f).upper() != "NODE_ID"]

        sql = sql.format(",".join(ndflds))
        curr.execute(sql)

        curr.execute("""SELECT AddGeometryColumn( 'links', 'geometry', 4326, 'LINESTRING', 'XY' )""")
        curr.execute("""SELECT AddGeometryColumn( 'nodes', 'geometry', 4326, 'POINT', 'XY' )""")
        self.conn.commit()

    def count_links(self) -> int:
        c = self.conn.cursor()
        c.execute("""select count(*) from links""")
        return c.fetchone()[0]

    def count_nodes(self) -> int:
        c = self.conn.cursor()
        c.execute("""select count(*) from nodes""")
        return c.fetchone()[0]

    def add_network_triggers(self) -> None:
        curr = self.conn.cursor()
        logger.info("Adding data indices")

        curr.execute("""CREATE INDEX links_a_node_idx ON links (a_node);""")
        curr.execute("""CREATE INDEX links_b_node_idx ON links (b_node);""")

        pth = os.path.dirname(os.path.realpath(__file__))
        qry_file = os.path.join(pth, "network_triggers.sql")
        with open(qry_file, "r") as sql_file:
            query_list = sql_file.read()
        logger.info("Adding network triggers")
        # Run one query/command at a time
        for cmd in query_list.split("#"):
            try:
                curr.execute(cmd)
            except Exception as e:
                msg = f"Error creating trigger: {e.args}"
                logger.error(msg)
                logger.info(cmd)
        self.conn.commit()

    def add_spatial_index(self) -> None:
        curr = self.conn.cursor()
        curr.execute("""SELECT CreateSpatialIndex( 'links' , 'geometry' );""")
        curr.execute("""SELECT CreateSpatialIndex( 'nodes' , 'geometry' );""")
        self.conn.commit()
