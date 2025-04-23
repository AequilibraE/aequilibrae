import math
from typing import Dict, Optional

import numpy as np
import pandas as pd
import shapely.wkb
import shapely.wkt
from shapely.geometry import Polygon, box
from shapely import union_all

from aequilibrae.context import get_logger
from aequilibrae.parameters import Parameters
from aequilibrae.project.network.gmns_builder import GMNSBuilder
from aequilibrae.project.network.gmns_exporter import GMNSExporter
from aequilibrae.project.network.haversine import haversine
from aequilibrae.project.network.link_types import LinkTypes
from aequilibrae.project.network.links import Links
from aequilibrae.project.network.modes import Modes
from aequilibrae.project.network.nodes import Nodes
from aequilibrae.project.network.osm.osm_builder import OSMBuilder
from aequilibrae.project.network.osm.osm_downloader import OSMDownloader
from aequilibrae.project.network.osm.place_getter import placegetter
from aequilibrae.project.network.periods import Periods
from aequilibrae.project.project_creation import req_link_flds, req_node_flds, protected_fields
from aequilibrae.utils.aeq_signal import SIGNAL
from aequilibrae.utils.interface.worker_thread import WorkerThread


class Network(WorkerThread):
    """
    Network class. Member of an AequilibraE Project
    """

    req_link_flds = req_link_flds
    req_node_flds = req_node_flds
    protected_fields = protected_fields
    link_types: LinkTypes = None
    signal = SIGNAL(object)

    def __init__(self, project) -> None:
        WorkerThread.__init__(self, None)
        from aequilibrae.paths import Graph

        self.graphs = {}  # type: Dict[Graph]
        self.project = project
        self.logger = project.logger
        self.modes = Modes(self)
        self.link_types = LinkTypes(self)
        self.links = Links(self)
        self.nodes = Nodes(self)
        self.periods = Periods(self)

    def skimmable_fields(self):
        """
        Returns a list of all fields that can be skimmed

        :Returns:
            :obj:`list`: List of all fields that can be skimmed
        """

        with self.project.db_connection as conn:
            field_names = conn.execute("PRAGMA table_info(links);").fetchall()

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

        with self.project.db_connection as conn:
            all_modes = [x[0] for x in conn.execute("""select mode_id from modes""").fetchall()]
        return all_modes

    def create_from_osm(
        self,
        model_area: Optional[Polygon] = None,
        place_name: Optional[str] = None,
        modes=("car", "transit", "bicycle", "walk"),
        clean=True,
    ) -> None:
        """
        Downloads the network from OpenStreetMap (OSM)

        :Arguments:
            **area** (:obj:`Polygon`, *Optional*): Polygon for which the network will be downloaded. If not provided,
            a place name would be required

            **place_name** (:obj:`str`, *Optional*): If not downloading with East-West-North-South boundingbox, this is
            required

            **modes** (:obj:`tuple`, *Optional*): List of all modes to be downloaded. Defaults to the modes in the
            parameter file

            **clean** (:obj:`bool`, *Optional*): Keeps only the links that intersects the model area polygon.
            Defaults to ``True``. Does not apply to networks downloaded with a place name

        .. code-block:: python

            >>> project = Project()
            >>> project.new(project_path)

            # Now we can import the network for any place we want
            >>> project.network.create_from_osm(place_name="my_beautiful_hometown") # doctest: +SKIP

            >>> project.close()
        """

        if self.count_links() > 0:
            raise FileExistsError("You can only import an OSM network into a brand new model file")

        with self.project.db_connection as conn:
            conn.execute("""ALTER TABLE links ADD COLUMN osm_id integer""")
            conn.execute("""ALTER TABLE nodes ADD COLUMN osm_id integer""")

        if isinstance(modes, (tuple, list)):
            modes = list(modes)
        elif isinstance(modes, str):
            modes = [modes]
        else:
            raise ValueError("'modes' needs to be string or list/tuple of string")

        if place_name is None:
            if (
                model_area.bounds[0] < -180
                or model_area.bounds[2] > 180
                or model_area.bounds[1] < -90
                or model_area.bounds[3] > 90
            ):
                raise ValueError("Coordinates out of bounds. Polygon must be in WGS84")
            west, south, east, north = model_area.bounds
        else:
            clean = False
            bbox, report = placegetter(place_name)
            if bbox is None:
                msg = f'We could not find a reference for place name "{place_name}"'
                self.logger.warning(msg)
                return
            for i in report:
                if "PLACE FOUND" in i:
                    self.logger.info(i)
            model_area = box(*bbox)
            west, south, east, north = bbox

        # Need to compute the size of the bounding box to not exceed it too much
        height = haversine((east + west) / 2, south, (east + west) / 2, north)
        width = haversine(east, (north + south) / 2, west, (north + south) / 2)
        area = height * width

        par = Parameters().parameters["osm"]
        max_query_area_size = par["max_query_area_size"]

        if area < max_query_area_size:
            polygons = [model_area]
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
                    subarea = box(xmin, ymin, xmax, ymax)
                    if subarea.intersects(model_area):
                        polygons.append(subarea)
        self.logger.info("Downloading data")
        dwnloader = OSMDownloader(polygons, modes, logger=self.logger)
        dwnloader.signal = self.signal
        dwnloader.doWork()

        self.logger.info("Building Network")
        self.builder = OSMBuilder(dwnloader.data, project=self.project, model_area=model_area, clean=clean)

        self.builder.signal = self.signal
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

            **use_group_path** (:obj:`str`, *Optional*): Path to a csv table containing groupings of uses.
            This helps AequilibraE know when a GMNS use is actually a group of other GMNS uses

            **geometry_path** (:obj:`str`, *Optional*): Path to a csv file containing geometry information for a line
            object, if not specified in the link table

            **srid** (:obj:`int`, *Optional*): Spatial Reference ID in which the GMNS geometries were created
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

    def build_graphs(self, fields: list = None, modes: list = None, limit_to_area: Polygon = None) -> None:
        """Builds graphs for all modes currently available in the model

        When called, it overwrites all graphs previously created and stored in the networks'
        dictionary of graphs

        :Arguments:
            **fields** (:obj:`list`, *Optional*): When working with very large graphs with large number of fields in the
            database, it may be useful to specify which fields to use

            **modes** (:obj:`list`, *Optional*): When working with very large graphs with large number of fields in the
            database, it may be useful to generate only those we need

            **limit_to_area** (:obj:`Polygon`, *Optional*): When working with a very large model area, you may want to
            filter your database to a small area for your computation, which you can do by providing a polygon.
            The search is limited to a spatial index search, so it is very fast but NOT PRECISE.

        To use the 'fields' parameter, a minimalistic option is the following

        .. code-block:: python

            >>> project = create_example(project_path)

            >>> fields = ['distance']
            >>> project.network.build_graphs(fields, modes = ['c', 'w'])

        """
        from aequilibrae.paths import Graph

        with self.project.db_connection as conn:
            if fields is None:
                field_names = conn.execute("PRAGMA table_info(links);").fetchall()

                ignore_fields = ["ogc_fid", "geometry"]
                all_fields = [f[1] for f in field_names if f[1] not in ignore_fields]
            else:
                fields.extend(["link_id", "a_node", "b_node", "direction", "modes"])
                all_fields = list(set(fields))

            if modes is None:
                modes = conn.execute("select mode_id from modes;").fetchall()
                modes = [m[0] for m in modes]
            elif isinstance(modes, str):
                modes = [modes]

            if limit_to_area is not None:
                spatial_add = """ WHERE links.rowid in (
                                        select rowid from SpatialIndex where f_table_name = 'links' and
                                       search_frame = GeomFromWKB(?, 4326))"""

            sql = f"select {','.join(all_fields)} from links"

            sql_centroids = "select node_id from nodes where is_centroid=1 order by node_id;"
            centroids = np.array([i[0] for i in conn.execute(sql_centroids).fetchall()], np.uint32)
            centroids = centroids if centroids.shape[0] else None

            with pd.option_context("future.no_silent_downcasting", True):
                if limit_to_area is None:
                    df = pd.read_sql(sql, conn).fillna(value=np.nan).infer_objects(False)
                else:
                    sql += spatial_add
                    df = (
                        pd.read_sql_query(sql, conn, params=(limit_to_area.wkb,))
                        .fillna(value=np.nan)
                        .infer_objects(False)
                    )

                    # We filter to centroids existing in our filtered area
                    centroids = centroids[np.isin(centroids, df.a_node) | np.isin(centroids, df.b_node)]

            valid_fields = list(df.select_dtypes(np.number).columns) + ["modes"]

        lonlat = self.nodes.lonlat.set_index("node_id")
        data = df[valid_fields]
        for m in modes:
            # For any link in net that doesn't support mode 'm', set a_node = b_node (these will be culled when
            # the compressed graph representation is created)
            net = pd.DataFrame(data, copy=True)
            net.loc[~net.modes.str.contains(m), "b_node"] = net.loc[~net.modes.str.contains(m), "a_node"]

            g = Graph()
            g.mode = m
            g.network = net
            g.prepare_graph(centroids)
            g.set_blocked_centroid_flows(True)
            if centroids is None:
                get_logger().warning("Your graph has no centroids")
            g.lonlat_index = lonlat.loc[g.all_nodes]
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
        with self.project.db_connection as conn:
            poly = shapely.wkb.loads(conn.execute('Select ST_asBinary(GetLayerExtent("Links"))').fetchone()[0])
        return poly

    def convex_hull(self) -> Polygon:
        """Queries the model for the convex hull of the entire network

        :Returns:
            **model coverage** (:obj:`Polygon`): Shapely (Multi)polygon of the model network.
        """
        with self.project.db_connection as conn:
            sql = 'Select ST_asBinary("geometry") from Links where ST_Length("geometry") > 0;'
            links = [shapely.wkb.loads(x[0]) for x in conn.execute(sql).fetchall()]
        return union_all(links).convex_hull

    def __count_items(self, field: str, table: str, condition: str) -> int:
        with self.project.db_connection as conn:
            c = conn.execute(f"select count({field}) from {table} where {condition};").fetchone()[0]
        return c
