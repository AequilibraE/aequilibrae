import logging

import pandas as pd
from shapely.geometry.base import BaseGeometry

from aequilibrae.project.network.connector_creation import connector_creation
from aequilibrae.project.network.links import Links
from aequilibrae.project.project_table import SpatialProjectTable

logger = logging.getLogger(__name__)


class Nodes(SpatialProjectTable):
    """
    Access to the API resources to manipulate the nodes table in the network

    .. code-block:: python

        >>> from shapely.geometry import Point

        >>> project = create_example(project_path)

        >>> nodes = project.network.nodes

        # We can get a single node as an immutable record
        >>> node = nodes.get(21)

        # and write changes explicitly
        >>> nodes.update(21, geometry=Point(1, 2))

        # Centroids are created with their id and position
        >>> nodes.insert(node_id=998877, is_centroid=1, geometry=Point(1, 3))
        998877

        >>> project.close()
    """

    name = "nodes"
    key = "node_id"
    record_name = "NodeRecord"

    def new_centroid(self, node_id: int, geometry: BaseGeometry) -> int:
        """Creates a new centroid with a given ID at the given position

        :Arguments:
            **node_id** (:obj:`int`): ID of the centroid to be created

            **geometry** (:obj:`Point`): Position of the centroid
        """
        return self.insert(node_id=node_id, is_centroid=1, geometry=geometry)

    def renumber(self, node_id: int, new_id: int) -> None:
        """Renumbers a node in the network

        :Arguments:
            **node_id** (:obj:`int`): Current node_id

            **new_id** (:obj:`int`): New node_id
        """
        new_id = int(new_id)
        if new_id == node_id:
            logger.warning("This is already the node number")
            return
        self._change_key(node_id, new_id)
        logger.info(f"Node {node_id} was renumbered to {new_id}")

    def connect_mode(
        self,
        node_id: int,
        mode_id: str,
        link_types: str = "",
        connectors: int = 1,
        area: BaseGeometry | None = None,
    ) -> None:
        """Adds centroid connectors for the desired mode to the network file

        Centroid connectors are created by connecting the zone centroid to one or more nodes selected from
        all those that satisfy the mode and link_types criteria and are inside the provided area.

        The selection of the nodes that will be connected is done simply by computing running the
        `KMeans2 <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans2.html>`_
        clustering algorithm from SciPy and selecting the nodes closest to each cluster centroid.

        When there are no node candidates inside the provided area, is it progressively expanded until
        at least one candidate is found.

        If fewer candidates than required connectors are found, all candidates are connected.

        :Arguments:
            **node_id** (:obj:`int`): Id of the centroid to connect

            **mode_id** (:obj:`str`): Mode ID we are trying to connect

            **link_types** (:obj:`str`, *Optional*): String with all the link type IDs that can
            be considered. eg: yCdR. Defaults to ALL link types

            **connectors** (:obj:`int`, *Optional*): Number of connectors to add. Defaults to 1

            **area** (:obj:`Polygon`, *Optional*): Area limiting the search for connectors
        """
        if self.get(node_id).is_centroid != 1:
            logger.warning("Connecting a mode only makes sense for centroids and not for regular nodes")
            return

        links = Links(self._transactions)
        connector_creation(
            zone_id=node_id,
            mode_id=mode_id,
            link_types=link_types,
            connectors=connectors,
            transactions=self._transactions,
            links=links,
            delimiting_area=area,
            proj_nodes=self.data,
            proj_links=links.data,
        )

    @property
    def lonlat(self) -> pd.DataFrame:
        """Returns all nodes lon/lat coords as a Pandas DataFrame

        :Returns:
            **table** (:obj:`DataFrame`): Pandas DataFrame with all the nodes, with geometry as lon/lat
        """
        frame = pd.read_sql(
            "SELECT node_id, ST_X(geometry) AS lon, ST_Y(geometry) AS lat FROM nodes", self._transactions
        )
        return frame.set_index("node_id")
