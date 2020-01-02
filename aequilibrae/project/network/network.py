import math
from warnings import warn
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

        self.conn = project.conn

    def _check_if_exists(self):
        curr = self.conn.cursor()
        curr.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='links';")
        tbls = curr.fetchone()[0]
        if tbls > 0:
            return True
        return False

    def create_from_osm(
        self,
        west: float = None,
        south: float = None,
        east: float = None,
        north: float = None,
        place_name: str = None,
        modes=["car", "transit", "bicycle", "walk"],
    ):

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
                warn('We could not find a reference for place name "{}"'.format(place_name))
                return
            for i in report:
                if "PLACE FOUND" in i:
                    print(i)

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
                xmin = west + i * dx
                xmax = west + (i + 1) * dx
                for j in range(vertical):
                    ymin = south + j * dy
                    ymax = south + (j + 1) * dy
                    box = [xmin, ymin, xmax, ymax]
                    polygons.append(box)

        logger.info("Downloading data")
        self.downloader = OSMDownloader(polygons, modes)
        self.downloader.doWork()

        logger.info("Building Network")
        self.builder = OSMBuilder(self.downloader.json, self.conn)
        self.builder.doWork()
        logger.info("Network built successfully")

    def create_empty_tables(self):
        curr = self.conn.cursor()
        # Create the links table
        p = Parameters()
        fields = p.parameters["network"]["links"]["fields"]

        sql = """CREATE TABLE 'links' (
                          ogc_fid INTEGER PRIMARY KEY,
                          link_id INTEGER UNIQUE NOT NULL,
                          {});"""

        flds = fields["one-way"]

        owlf = [
            "{} {}".format(list(f.keys())[0], f[list(f.keys())[0]]["type"])
            for f in flds
            if list(f.keys())[0].upper() != "LINK_ID"
        ]

        flds = fields["two-way"]
        twlf = []
        for f in flds:
            nm = list(f.keys())[0]
            tp = f[nm]["type"]
            twlf.extend(["{}_ab {}".format(nm, tp), "{}_ba {}".format(nm, tp)])

        link_fields = owlf + twlf

        sql = sql.format(",".join(link_fields))
        curr.execute(sql)

        sql = """CREATE TABLE 'nodes' (ogc_fid INTEGER PRIMARY KEY,
                                 node_id INTEGER UNIQUE NOT NULL, {});"""

        flds = p.parameters["network"]["nodes"]["fields"]

        ndflds = [
            "{} {}".format(list(f.keys())[0], f[list(f.keys())[0]]["type"])
            for f in flds
            if list(f.keys())[0].upper() != "NODE_ID"
        ]

        sql = sql.format(",".join(ndflds))
        curr.execute(sql)

        curr.execute("""SELECT AddGeometryColumn( 'links', 'geometry', 4326, 'LINESTRING', 'XY' )""")
        curr.execute("""SELECT AddGeometryColumn( 'nodes', 'geometry', 4326, 'POINT', 'XY' )""")
        self.conn.commit()

    def count_links(self):
        print("quick query on number of links")

    def count_nodes(self):
        print("quick query on number of nodes")

    def many_more_queries_like_that(self):
        print("With all the work that goes with it")
