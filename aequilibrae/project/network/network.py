import math
import os
from warnings import warn
from sqlite3 import Connection as sqlc
from typing import List, Dict
import numpy as np
from aequilibrae.project.network import OSMDownloader
from aequilibrae.project.network.osm_builder import OSMBuilder
from aequilibrae.project.network.osm_utils.place_getter import placegetter
from aequilibrae.project.network.osm_utils.osm_params import max_query_area_size
from aequilibrae.project.network.haversine import haversine
from aequilibrae.paths import Graph
from aequilibrae.parameters import Parameters
from aequilibrae import logger


class Network():
    """
    Network class. Member of an AequilibraE Project
    """
    req_link_flds = ["link_id", "a_node", "b_node", "direction", "distance", "modes", "link_type"]
    req_node_flds = ["node_id", "is_centroid"]
    protected_fields = ['ogc_fid', 'geometry']

    def __init__(self, project):
        """
        Instantiates the network with the project it is member of

        Args:
            *project* (:obj:`Project`): Project
        """
        self.conn = project.conn  # type: sqlc
        self.source = project.source  # type: sqlc
        self.graphs = {}  # type: Dict[Graph]

    def _check_if_exists(self):
        curr = self.conn.cursor()
        curr.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='links';")
        tbls = curr.fetchone()[0]
        return tbls > 0

    # TODO: DOCUMENT THESE FUNCTIONS
    def skimmable_fields(self):
        """
        Returns a list of all fields that can be skimmed

        Returns:
            :obj:`list`: List of all fields that can be skimmed
        """
        curr = self.conn.cursor()
        curr.execute('PRAGMA table_info(links);')
        field_names = curr.fetchall()
        ignore_fields = ['ogc_fid', 'geometry'] + self.req_link_flds

        skimmable = ['INT', 'INTEGER', 'TINYINT', 'SMALLINT', 'MEDIUMINT', 'BIGINT', 'UNSIGNED BIG INT',
                     'INT2', 'INT8', 'REAL', 'DOUBLE', 'DOUBLE PRECISION', 'FLOAT', 'DECIMAL', 'NUMERIC']
        all_fields = []

        for f in field_names:
            if f[1] in ignore_fields:
                continue
            for i in skimmable:
                if i in f[2].upper():
                    all_fields.append(f[1])
                    break

        all_fields.append('distance')
        real_fields = []
        for f in all_fields:
            if f[-2:] == "ab":
                if f[:-2] + 'ba' in all_fields:
                    real_fields.append(f[:-3])
            elif f[-3:] == "_ba":
                pass
            else:
                real_fields.append(f)

        return real_fields

    def modes(self):
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
            spatial_index=False,
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

            *spatial_index* (:obj:`bool`, Optional): Creates spatial index. Defaults to zero. REQUIRES SQLITE WITH RTREE
        """

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
                logger.warning(msg)
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
        self.builder = OSMBuilder(self.downloader.json, self.source)
        self.builder.doWork()

        if spatial_index:
            logger.info("Adding spatial indices")
            self.add_spatial_index()

        self.add_triggers()
        logger.info("Network built successfully")

    def create_empty_tables(self) -> None:
        """Creates empty network tables for future filling"""
        curr = self.conn.cursor()
        # Create the links table
        p = Parameters()
        fields = p.parameters["network"]["links"]["fields"]

        sql = """CREATE TABLE 'links' (
                          ogc_fid INTEGER PRIMARY KEY,
                          link_id INTEGER UNIQUE,
                          a_node INTEGER,
                          b_node INTEGER,
                          direction INTEGER NOT NULL DEFAULT 0,
                          distance NUMERIC,
                          modes TEXT NOT NULL,
                          link_type TEXT NOT NULL DEFAULT 'link type not defined'
                          {});"""

        flds = fields["one-way"]

        # returns first key in the dictionary
        def fkey(f):
            return list(f.keys())[0]

        owlf = ["{} {}".format(fkey(f), f[fkey(f)]["type"]) for f in flds if fkey(f).lower() not in self.req_link_flds]

        flds = fields["two-way"]
        twlf = []
        for f in flds:
            nm = fkey(f)
            tp = f[nm]["type"]
            twlf.extend([f"{nm}_ab {tp}", f"{nm}_ba {tp}"])

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
        ndflds = [f"{fkey(f)} {f[fkey(f)]['type']}" for f in flds if fkey(f).lower() not in self.req_node_flds]

        if ndflds:
            sql = sql.format("," + ",".join(ndflds))
        else:
            sql = sql.format("")
        curr.execute(sql)

        curr.execute("""SELECT AddGeometryColumn( 'links', 'geometry', 4326, 'LINESTRING', 'XY' )""")
        curr.execute("""SELECT AddGeometryColumn( 'nodes', 'geometry', 4326, 'POINT', 'XY' )""")
        self.conn.commit()

    def build_graphs(self) -> None:
        """Builds graphs for all modes currently available in the model"""
        curr = self.conn.cursor()
        curr.execute('PRAGMA table_info(links);')
        field_names = curr.fetchall()

        ignore_fields = ['ogc_fid', 'geometry']
        all_fields = [f[1] for f in field_names if f[1] not in ignore_fields]

        raw_links = curr.execute(f"select {','.join(all_fields)} from links").fetchall()
        links = []
        for l in raw_links:
            lk = list(map(lambda x: np.nan if x is None else x, l))
            links.append(lk)
        # links =

        data = np.core.records.fromrecords(links, names=all_fields)

        valid_fields = []
        removed_fields = []
        for f in all_fields:
            if np.issubdtype(data[f].dtype, np.floating) or np.issubdtype(data[f].dtype, np.integer):
                valid_fields.append(f)
            else:
                removed_fields.append(f)
        if len(removed_fields) > 1:
            warn(f'Fields were removed form Graph for being non-numeric: {",".join(removed_fields)}')

        curr.execute('select node_id from nodes where is_centroid=1;')
        centroids = np.array([i[0] for i in curr.fetchall()], np.uint32)

        modes = curr.execute('select mode_id from modes;').fetchall()
        modes = [m[0] for m in modes]

        for m in modes:
            w = np.core.defchararray.find(data['modes'], m)
            net = np.array(data[valid_fields], copy=True)
            net['b_node'][w < 0] = net['a_node'][w < 0]

            g = Graph()
            g.mode = m
            g.network = net
            g.network_ok = True
            g.status = 'OK'
            g.prepare_graph(centroids)
            g.set_blocked_centroid_flows(True)
            self.graphs[m] = g

    def set_time_field(self, time_field: str) -> None:
        """
        Set the time field for all graphs built in the model

        Args:
            *time_field* (:obj:`str`): Network field with travel time information
        """
        for m, g in self.graphs.items():  # type: str, Graph
            if time_field not in list(g.graph.dtype.names):
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
        c = self.conn.cursor()
        c.execute("""select count(link_id) from links""")
        return c.fetchone()[0]

    def count_centroids(self) -> int:
        """
        Returns the number of centroids in the model

        Returns:
            :obj:`int`: Number of centroids
        """
        c = self.conn.cursor()
        c.execute("""select count(node_id) from nodes where is_centroid=1;""")
        return c.fetchone()[0]

    def count_nodes(self) -> int:
        """
        Returns the number of nodes in the model

        Returns:
            :obj:`int`: Number of nodes
        """
        c = self.conn.cursor()
        c.execute("""select count(node_id) from nodes""")
        return c.fetchone()[0]

    def add_triggers(self):
        """Adds consistency triggers to the project"""
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
                msg = f"Error creating trigger: {e.args}"
                logger.error(msg)
                logger.info(cmd)
        self.conn.commit()

    def add_spatial_index(self) -> None:
        """Adds spatial indices to links and nodes table

        Requires an Sqlite3 distribution with RTree (not the Python standard).
        Use with caution"""
        curr = self.conn.cursor()
        curr.execute("""SELECT CreateSpatialIndex( 'links' , 'geometry' );""")
        curr.execute("""SELECT CreateSpatialIndex( 'nodes' , 'geometry' );""")
        self.conn.commit()
