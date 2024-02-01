# %%
import importlib.util as iutil
import math
from sqlite3 import Connection as sqlc
from typing import Dict

import numpy as np
import pandas as pd
import shapely.wkb
import shapely.wkt
from shapely.geometry import Polygon
from shapely.ops import unary_union

from aequilibrae.context import get_logger
from aequilibrae.parameters import Parameters
from aequilibrae.project.network import OSMDownloader
from aequilibrae.project.network.gmns_builder import GMNSBuilder
from aequilibrae.project.network.gmns_exporter import GMNSExporter
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
        self.project = project
        self.logger = project.logger
        self.modes = Modes(self)
        self.link_types = LinkTypes(self)
        self.links = Links(self)
        self.nodes = Nodes(self)

    def skimmable_fields(self):
        """
        Returns a list of all fields that can be skimmed

        :Returns:
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

        :Returns:
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

        :Arguments:
            **west** (:obj:`float`, Optional): West most coordinate of the download bounding box

            **south** (:obj:`float`, Optional): South most coordinate of the download bounding box

            **east** (:obj:`float`, Optional): East most coordinate of the download bounding box

            **place_name** (:obj:`str`, Optional): If not downloading with East-West-North-South boundingbox, this is
            required

            **modes** (:obj:`list`, Optional): List of all modes to be downloaded. Defaults to the modes in the parameter
            file

        .. code-block:: python

            >>> from aequilibrae import Project

            >>> p = Project()
            >>> p.new("/tmp/new_project")

            # We now choose a different overpass endpoint (say a deployment in your local network)
            >>> par = Parameters()
            >>> par.parameters['osm']['overpass_endpoint'] = "http://192.168.1.234:5678/api"

            # Because we have our own server, we can set a bigger area for download (in M2)
            >>> par.parameters['osm']['max_query_area_size'] = 10000000000

            # And have no pause between successive queries
            >>> par.parameters['osm']['sleeptime'] = 0

            # Save the parameters to disk
            >>> par.write_back()

            # Now we can import the network for any place we want
            # p.network.create_from_osm(place_name="my_beautiful_hometown")

            >>> p.close()
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
                self.logger.warning(msg)
                return
            for i in report:
                if "PLACE FOUND" in i:
                    self.logger.info(i)

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
        self.logger.info("Downloading data")
        self.downloader = OSMDownloader(polygons, modes, logger=self.logger)
        if pyqt:
            self.downloader.downloading.connect(self.signal_handler)

        self.downloader.doWork()

        self.logger.info("Building Network")
        self.builder = OSMBuilder(self.downloader.json, self.source, project=self.project)

        if pyqt:
            self.builder.building.connect(self.signal_handler)
        self.builder.doWork()

        self.logger.info("Network built successfully")

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

        :Arguments:
            **link_file_path** (:obj:`str`): Path to a links csv file in GMNS format

            **node_file_path** (:obj:`str`): Path to a nodes csv file in GMNS format

            **use_group_path** (:obj:`str`, Optional): Path to a csv table containing groupings of uses. This helps AequilibraE
            know when a GMNS use is actually a group of other GMNS uses

            **geometry_path** (:obj:`str`, Optional): Path to a csv file containing geometry information for a line object, if not
            specified in the link table

            **srid** (:obj:`int`, Optional): Spatial Reference ID in which the GMNS geometries were created
        """

        gmns_builder = GMNSBuilder(self, link_file_path, node_file_path, use_group_path, geometry_path, srid)
        gmns_builder.doWork()

        self.logger.info("Network built successfully")

    def export_to_gmns(self, path: str):
        """
        Exports AequilibraE network to csv files in GMNS format.

        :Arguments:
            **path** (:obj:`str`): Output folder path.
        """

        gmns_exporter = GMNSExporter(self, path)
        gmns_exporter.doWork()

        self.logger.info("Network exported successfully")

    def signal_handler(self, val):
        if pyqt:
            self.netsignal.emit(val)

    def build_graphs(self, fields: list = None, modes: list = None) -> None:
        """Builds graphs for all modes currently available in the model

        When called, it overwrites all graphs previously created and stored in the networks'
        dictionary of graphs

        :Arguments:
            **fields** (:obj:`list`, optional): When working with very large graphs with large number of fields in the
                                              database, it may be useful to specify which fields to use
            **modes** (:obj:`list`, optional): When working with very large graphs with large number of fields in the
                                              database, it may be useful to generate only those we need

        To use the *fields* parameter, a minimalistic option is the following

        .. code-block:: python

            >>> from aequilibrae import Project

            >>> p = Project.from_path("/tmp/test_project")
            >>> fields = ['distance']
            >>> p.network.build_graphs(fields, modes = ['c', 'w'])

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
            if centroids.shape[0]:
                g.prepare_graph(centroids)
                g.set_blocked_centroid_flows(True)
            else:
                get_logger().warning("Your graph has no centroids")
            self.graphs[m] = g

    def set_time_field(self, time_field: str) -> None:
        """
        Set the time field for all graphs built in the model

        :Arguments:
            **time_field** (:obj:`str`): Network field with travel time information
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

        :Returns:
            :obj:`int`: Number of links
        """
        return self.__count_items("link_id", "links", "link_id>=0")

    def count_centroids(self) -> int:
        """
        Returns the number of centroids in the model

        :Returns:
            :obj:`int`: Number of centroids
        """
        return self.__count_items("node_id", "nodes", "is_centroid=1")

    def count_nodes(self) -> int:
        """
        Returns the number of nodes in the model

        :Returns:
            :obj:`int`: Number of nodes
        """
        return self.__count_items("node_id", "nodes", "node_id>=0")

    def extent(self):
        """Queries the extent of the network included in the model

        :Returns:
            **model extent** (:obj:`Polygon`): Shapely polygon with the bounding box of the model network.
        """
        curr = self.conn.cursor()
        curr.execute('Select ST_asBinary(GetLayerExtent("Links"))')
        poly = shapely.wkb.loads(curr.fetchone()[0])
        return poly

    def convex_hull(self) -> Polygon:
        """Queries the model for the convex hull of the entire network

        :Returns:
            **model coverage** (:obj:`Polygon`): Shapely (Multi)polygon of the model network.
        """
        curr = self.conn.cursor()
        curr.execute('Select ST_asBinary("geometry") from Links where ST_Length("geometry") > 0;')
        links = [shapely.wkb.loads(x[0]) for x in curr.fetchall()]
        return unary_union(links).convex_hull

    def refresh_connection(self):
        """Opens a new database connection to avoid thread conflict"""
        self.conn = self.project.connect()

    def __count_items(self, field: str, table: str, condition: str) -> int:
        c = self.conn.execute(f"select count({field}) from {table} where {condition};").fetchone()[0]
        return c

# %%
# Imports
from uuid import uuid4
from tempfile import gettempdir
from os.path import join
from aequilibrae import Project
import folium
# sphinx_gallery_thumbnail_path = 'images/nauru.png'

# %%
# We create an empty project on an arbitrary folder
fldr = join(gettempdir(), uuid4().hex)
project = Project()
project.new(fldr)

# %%
project.network.create_from_osm(place_name="Airlie Beach")


# %%
links = project.network.links.data
links

# %%
project.network.nodes.data

#%%
project.network.count_links()

# %%
project.network.count_nodes()

# %%
curr = project.network.conn.cursor()
project.network.conn.commit()

# %%
modes = Modes(project.network)
modes=["car", "transit", "bicycle", "walk"]
if isinstance(modes, (tuple, list)):
    modes = list(modes)
elif isinstance(modes, str):
    modes = [modes]
else:
    raise ValueError("modes needs to be string or list/tuple of string")
modes

# %%
par = Parameters().parameters["osm"]
max_query_area_size = par["max_query_area_size"]
par

# %%
Parameters().__dict__

# %%
Parameters().parameters["network"]

# %%
place_name = "Nauru"

# %%
north: float = None
east: float = None
south: float = None
west: float = None

if place_name is None:
    if min(east, west) < -180 or max(east, west) > 180 or min(north, south) < -90 or max(north, south) > 90:
        raise ValueError("Coordinates out of bounds")
    bbox = [west, south, east, north]
else:
    bbox, report = placegetter(place_name)
    west, south, east, north = bbox
    if bbox is None:
        msg = f'We could not find a reference for place name "{place_name}"'
        project.network.logger.warning(msg)
    
    for i in report:
        if "PLACE FOUND" in i:
            project.network.logger.info(i)

# %%
bbox

# %%
height = haversine((east + west) / 2, south, (east + west) / 2, north)
width = haversine(east, (north + south) / 2, west, (north + south) / 2)
area = height * width
area

# %%
area < max_query_area_size

# %%
polygons = []
parts = math.ceil(area / max_query_area_size)
horizontal = math.ceil(math.sqrt(parts))
vertical = math.ceil(parts / horizontal)
dx = (east - west) / horizontal
dy = (north - south) / vertical

# %%
parts
# %%
horizontal
# %%
vertical
# %%
dx
# %%
dy
# %%
for i in range(horizontal):
    xmin = max(-180, west + i * dx)
    xmax = min(180, west + (i + 1) * dx)
    for j in range(vertical):
        ymin = max(-90, south + j * dy)
        ymax = min(90, south + (j + 1) * dy)
        box = [xmin, ymin, xmax, ymax]
        polygons.append(box)

polygons
# %%
ymax
# %%
ymin
# %%
xmax
# %%
xmin
# %%
project.network.downloader = OSMDownloader(polygons, modes, logger= project.network.logger)
project.network.downloader

# %%
print(project.network.downloader.json)
print(project.network.downloader.report)
print(polygons)
print(project.network.logger)

# %%
if pyqt:
    project.network.builder.building.connect(project.network.signal_handler)

# project.network.builder.doWork()
# we will come back to the builder

# %%
project.network.downloader.doWork()
project.network.downloader.json

# %%

# looking through downloader's doWork()
import logging
import time
import re
import requests
from aequilibrae.project.network.osm_utils.osm_params import http_headers, memory
from aequilibrae.parameters import Parameters
from aequilibrae.context import get_logger
import gc
import importlib.util as iutil
from aequilibrae.utils import WorkerThread

spec = iutil.find_spec("PyQt5")
pyqt = spec is not None
if pyqt:
    from PyQt5.QtCore import pyqtSignal

# %%
infrastructure = 'way["highway"]'
query_template = (
            "{memory}[out:json][timeout:{timeout}];({infrastructure}{filters}({south:.6f},{west:.6f},"
            "{north:.6f},{east:.6f});>;);out;"
        )

# %%
query_template

# %%
project.__getstate__()

# %%
WorkerThread.__init__(project.network.downloader, None)
project.network.downloader.logger = get_logger()
project.network.downloader.polygons = polygons
project.network.downloader.filter = project.network.downloader.get_osm_filter(modes)
project.network.downloader.report = []
project.network.downloader.json = []
par = Parameters().parameters["osm"]
project.network.downloader.overpass_endpoint = par["overpass_endpoint"]
project.network.downloader.timeout = par["timeout"]
project.network.downloader.sleeptime = par["sleeptime"]

# %%
project.network.downloader.downloading.emit(["maxValue", len(project.network.downloader.polygons)])
project.network.downloader.downloading.emit(["Value", 0])

# %%
m = ""
if memory > 0:
    m = f"[maxsize: {memory}]"
memory
# %%
query_str = query_template.format(
                north=north,
                south=south,
                east=east,
                west=west,
                infrastructure=infrastructure,
                filters=project.network.downloader.filter,
                timeout=project.network.downloader.timeout,
                memory=m,
            )
query_str
# %%
project.network.downloader.overpass_request(data={"data": query_str}, timeout=project.network.downloader.timeout)

# %%
json = project.network.downloader.overpass_request(data={"data": query_str}, timeout=project.network.downloader.timeout)
json
# %%
if json["elements"]:
    project.network.downloader.json.extend(json["elements"])
del json
project.network.downloader.json

# %%
gc.collect()

# %%
for counter, poly in enumerate(project.network.downloader.polygons):
    msg = f"Downloading polygon {counter + 1} of {len(project.network.downloader.polygons)}"
    project.network.downloader.logger.debug(msg)
    project.network.downloader.downloading.emit(["Value", counter])
    project.network.downloader.downloading.emit(["text", msg])
    west, south, east, north = poly
    query_str = query_template.format(
                north=north,
                south=south,
                east=east,
                west=west,
                infrastructure=infrastructure,
                filters=project.network.downloader.filter,
                timeout=project.network.downloader.timeout,
                memory=m,
    )
    json = project.network.downloader.overpass_request(data={"data": query_str}, timeout=project.network.downloader.timeout)
    if json["elements"]:
        project.network.downloader.json.extend(json["elements"])
    del json
    gc.collect()
project.network.downloader.downloading.emit(["Value", len(project.network.downloader.polygons)])
project.network.downloader.downloading.emit(["FinishedDownloading", 0])
project.network.downloader.json

# %%
project.network.logger.info("Downloading data")
project.network.downloader = OSMDownloader(polygons, modes, logger=project.network.logger)
print(project.network.downloader.json)
project.network.downloader.doWork()
project.network.downloader.json

# %%
project.network.logger.info("Building Network")
project.network.builder = OSMBuilder(project.network.downloader.json, project.network.source, project=project.network.project)
if pyqt:
    project.network.builder.building.connect(project.network.signal_handler)

# project.network.builder.doWork()
# can't access doWork() in this file, so we will go through each step of the function separately here
project.network.builder.nodes

# %%

# looking through builder's doWork()
import gc
import importlib.util as iutil
import sqlite3
import string
from typing import List

import numpy as np
import pandas as pd

from aequilibrae.context import get_active_project
from aequilibrae.parameters import Parameters
from aequilibrae.project.network.link_types import LinkTypes
from aequilibrae.utils.spatialite_utils import connect_spatialite
from aequilibrae.project.network.haversine import haversine
from aequilibrae.utils import WorkerThread

spec = iutil.find_spec("PyQt5")
pyqt = spec is not None
if pyqt:
    from PyQt5.QtCore import pyqtSignal

spec = iutil.find_spec("qgis")
isqgis = spec is not None
if isqgis:
    import qgis

# %%
WorkerThread.__init__(project.network.builder, None)
project.network.builder.project = project or get_active_project()
project.network.builder.logger = project.network.builder.project.logger
project.network.builder.conn = None
project.network.builder.__link_types = None  # type: LinkTypes
project.network.builder.report = []
project.network.builder.__model_link_types = []
project.network.builder.__model_link_type_ids = []
project.network.builder.__link_type_quick_reference = {}
project.network.builder.nodes = {}
project.network.builder.node_df = []
project.network.builder.links = {}
project.network.builder.insert_qry = """INSERT INTO {} ({}, geometry) VALUES({}, GeomFromText(?, 4326))"""
# %%
project.network.builder.conn = connect_spatialite(project.network.builder.path)
print(project.network.builder.conn)
project.network.builder.curr = project.network.builder.conn.cursor()
print(project.network.builder.curr)

# project.network.builder.__worksetup()
# can't access so will go through each step

# %%
project.network.builder.__link_types = project.network.builder.project.network.link_types
lts = project.network.builder.__link_types.all_types()
for lt_id, lt in lts.items():
    project.network.builder.__model_link_types.append(lt.link_type)
    project.network.builder.__model_link_type_ids.append(lt_id)
# %%
print(project.network.builder.__model_link_types)
print(project.network.builder.__model_link_type_ids)
lts

# %%

# back to doWork
node_count = project.network.builder.data_structures()
node_count

# %%

# project.network.builder.importing_links(node_count)
# this would have been the next step in doWork but again we don't have access to this function
# in this particual case so we will go step by step
node_ids = {}

vars = {}
vars["link_id"] = 1
table = "links"
# fields = project.network.builder.get_link_fields()
# same again no access, so step by step
p = Parameters()
fields = p.parameters["network"]["links"]["fields"]
owf = [list(x.keys())[0] for x in fields["one-way"]]

twf1 = ["{}_ab".format(list(x.keys())[0]) for x in fields["two-way"]]
twf2 = ["{}_ba".format(list(x.keys())[0]) for x in fields["two-way"]]

return_get_link_fields =  owf + twf1 + twf2 + ["osm_id"]
print(fields)
print(owf)
print(twf1)
print(twf2)
print(return_get_link_fields)

# %%
