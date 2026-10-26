import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import shapely.wkb
from shapely import union_all
from shapely.geometry import Polygon

from aequilibrae.project.network.gmns_exporter import GMNSExporter
from aequilibrae.project.network.importers import Importer
from aequilibrae.project.network.link_types import LinkTypes
from aequilibrae.project.network.links import Links
from aequilibrae.project.network.modes import Modes
from aequilibrae.project.network.nodes import Nodes
from aequilibrae.project.network.periods import Periods
from aequilibrae.project.network.turn_restrictions import TurnRestrictions
from aequilibrae.project.network.zones import Zones
from aequilibrae.project.project_creation import protected_fields, req_link_flds, req_node_flds
from aequilibrae.utils.aeq_signal import SIGNAL
from aequilibrae.utils.db_utils import ConnectionClosure
from aequilibrae.utils.interface.worker_thread import WorkerThread
from aequilibrae.utils.spatialite_utils import load_spatialite_extension

if TYPE_CHECKING:
    from aequilibrae.paths.graph import Graph
    from aequilibrae.project import Project

logger = logging.getLogger(__name__)


class Network(WorkerThread):
    """
    Network class. Member of an AequilibraE Project
    """

    req_link_flds = req_link_flds
    req_node_flds = req_node_flds
    protected_fields = protected_fields
    link_types: LinkTypes
    signal = SIGNAL(object)

    def __init__(self, connections: ConnectionClosure, project=None) -> None:
        WorkerThread.__init__(self, None)

        self.__connections = connections
        self.__project = project  # HACK

        self.modes = Modes(self.__connections.db_connection)
        self.link_types = LinkTypes(self.__connections.db_connection)
        self.links = Links(self.__connections.db_connection)
        self.nodes = Nodes(self.__connections.db_connection)
        self.periods = Periods(self.__connections.db_connection)
        self.zones = Zones(self.__connections.db_connection)
        self.turn_restrictions = TurnRestrictions(self.__connections.db_connection)
        self.importer = Importer(self)

    @property
    def project(self) -> "Project":
        """The project this network belongs to."""
        return self.__project

    @property
    def connections(self) -> ConnectionClosure:
        """The database connections for the scenario this network belongs to."""
        return self.__connections

    def skimmable_fields(self) -> list:
        """
        Returns a list of all fields that can be skimmed

        :Returns:
            :obj:`list`: List of all fields that can be skimmed
        """

        with self.__connections.db_connection as conn:
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

    def list_modes(self) -> list:
        """
        Returns a list of all the modes in this model

        :Returns:
            :obj:`list`: List of all modes
        """

        with self.__connections.db_connection as conn:
            all_modes = [x[0] for x in conn.execute("""select mode_id from modes""").fetchall()]
        return all_modes

    def export_to_gmns(self, path: str):
        """
        Exports AequilibraE network to csv files in GMNS format.

        :Arguments:
            **path** (:obj:`str`): Output folder path.
        """

        gmns_exporter = GMNSExporter(self.links, self.nodes, self.modes, path)
        gmns_exporter.doWork()

        logger.info("Network exported successfully")

    def build_graph(
        self,
        mode: str,
        cost: str,
        *,
        fields=None,
        centroids=None,
        skimming=None,
        block_centroid_flows=True,
        limit_to_area=None,
    ) -> "Graph":
        """Load and prepare a graph for standalone paths, skimming, or route choice."""
        from aequilibrae.paths.graph import Graph, _prepare_network_graph

        if mode not in self.list_modes():
            raise ValueError(f"Mode '{mode}' is not available in this network")
        if fields is not None:
            fields = list(dict.fromkeys([*fields, cost, *(skimming or [])]))
        links, project_centroids, restrictions, allow_uturns = self._load_graph_data(fields, limit_to_area)
        graph = Graph(network=links)
        _prepare_network_graph(
            graph,
            mode,
            project_centroids if centroids is None else np.asarray(centroids),
            restrictions,
            block_centroid_flows,
            allow_uturns,
        )
        graph.lonlat_index = self.nodes.lonlat.set_index("node_id").loc[graph.all_nodes]
        graph.set_graph(cost)
        graph.set_skimming(list(skimming or []))
        return graph

    def _load_graph_data(self, fields=None, limit_to_area=None):
        """Read source tables once for a graph or an assignment execution."""
        with self.__connections.db_connection as conn:
            if fields is None:
                field_names = conn.execute("PRAGMA table_info(links);").fetchall()

                ignore_fields = ["ogc_fid", "geometry"]
                all_fields = [f[1] for f in field_names if f[1] not in ignore_fields]
            else:
                available = {f[1] for f in conn.execute("PRAGMA table_info(links);").fetchall()}
                requested = ["link_id", "a_node", "b_node", "direction", "modes", "link_type"]
                for field in fields:
                    if field in available:
                        requested.append(field)
                    elif field + "_ab" in available and field + "_ba" in available:
                        requested.extend([field + "_ab", field + "_ba"])
                    else:
                        raise ValueError(f"'{field}' not in network")
                all_fields = list(dict.fromkeys(requested))

            if limit_to_area is not None:
                load_spatialite_extension(conn)
                spatial_add = """ WHERE links.rowid in (
                                        select rowid from SpatialIndex where f_table_name = 'links' and
                                       search_frame = GeomFromWKB(?, 4326))"""

            sql = f"select {','.join(all_fields)} from links"

            sql_centroids = "select node_id from nodes where is_centroid=1 order by node_id;"
            centroids = np.array([i[0] for i in conn.execute(sql_centroids).fetchall()], np.uint32)
            centroids = centroids if centroids.shape[0] else None

            if limit_to_area is None:
                df = pd.read_sql(sql, conn).fillna(value=np.nan)
            else:
                sql += spatial_add
                df = pd.read_sql_query(sql, conn, params=(limit_to_area.wkb,)).fillna(value=np.nan)

                # We filter to centroids existing in our filtered area
                if centroids is not None:
                    centroids = centroids[np.isin(centroids, df.a_node) | np.isin(centroids, df.b_node)]

            # Load turn restrictions and allow_uturns setting
            turn_restrictions_df = None
            allow_uturns = False

            table_check = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='turn_restrictions'"
            ).fetchone()
            if table_check:
                turn_restrictions_df = pd.read_sql("SELECT * FROM turn_restrictions", conn)
            else:
                logger.warning(
                    "This project does not have a 'turn_restrictions' table, consider upgrading the project "
                    "with 'project.upgrade()'."
                )

            # Load allow_uturns setting from about table
            allow_uturns_r = conn.execute("SELECT infovalue FROM about WHERE infoname = 'allow_uturns'").fetchone()
            if allow_uturns_r:
                allow_uturns = allow_uturns_r[0] == "1"

        assert turn_restrictions_df is None or "modes" in turn_restrictions_df.columns

        if limit_to_area is not None and turn_restrictions_df is not None:
            # Only retain turns whose two directed legs are included in the area.
            forward = df.loc[df.direction >= 0, ["a_node", "b_node"]]
            backward = df.loc[df.direction <= 0, ["b_node", "a_node"]].set_axis(["a_node", "b_node"], axis=1)
            edges = pd.MultiIndex.from_frame(pd.concat([forward, backward]))
            incoming = pd.MultiIndex.from_frame(turn_restrictions_df[["from_node", "via_node"]])
            outgoing = pd.MultiIndex.from_frame(turn_restrictions_df[["via_node", "to_node"]])
            turn_restrictions_df = turn_restrictions_df.loc[incoming.isin(edges) & outgoing.isin(edges)].copy()

        return df[all_fields], centroids, turn_restrictions_df, allow_uturns

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
        with self.__connections.db_connection as conn:
            poly = shapely.wkb.loads(conn.execute('Select ST_asBinary(GetLayerExtent("Links"))').fetchone()[0])
        return poly

    def convex_hull(self) -> Polygon:
        """Queries the model for the convex hull of the entire network

        :Returns:
            **model coverage** (:obj:`Polygon`): Shapely (Multi)polygon of the model network.
        """
        with self.__connections.db_connection as conn:
            sql = 'Select ST_asBinary("geometry") from Links where ST_Length("geometry") > 0;'
            links = [shapely.wkb.loads(x[0]) for x in conn.execute(sql).fetchall()]
        return union_all(links).convex_hull

    def __count_items(self, field: str, table: str, condition: str) -> int:
        with self.__connections.db_connection as conn:
            c = conn.execute(f"select count({field}) from {table} where {condition};").fetchone()[0]
        return c
