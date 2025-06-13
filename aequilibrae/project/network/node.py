from sqlite3 import Connection
from typing import Optional

from shapely.geometry import Polygon
from .safe_class import SafeClass
from .connector_creation import connector_creation


class Node(SafeClass):
    """A Node object represents a single record in the *nodes* table

    .. code-block:: python

        >>> from shapely.geometry import Point

        >>> project = create_example(project_path)

        >>> all_nodes = project.network.nodes

        # We can just get one link in specific
        >>> node1 = all_nodes.get(7)

        # We can find out which fields exist for the links
        >>> which_fields_do_we_have = node1.data_fields()

        # It success if the node_id already does not exist
        >>> node1.renumber(998877)

        >>> node1.geometry = Point(1,2)

        # We can just save the node
        >>> node1.save()
    """

    def __init__(self, dataset, project):
        super().__init__(dataset, project)
        self.__new = dataset["geometry"] is None
        self.__fields = list(dataset.keys())
        self._table = "nodes"

    def save(self):
        """Saves node to database"""
        with self.project.db_connection as conn:

            if self.node_id != self.__original__["node_id"]:
                raise ValueError("One cannot change the node_id")

            if self.__new:
                data, sql = self._save_new_with_geometry()
            else:
                data, sql = self.__save_existing_node()

            if data:
                conn.execute(sql, data)

            self.__new = False

    def data_fields(self) -> list:
        """lists all data fields for the node, as available in the database

        :Returns:
            **data fields** (:obj:`list`): list of all fields available for editing
        """

        return list(self.__original__.keys())

    def renumber(self, new_id: int):
        """Renumbers the node in the network

        Logs a warning if another node already exists with this node_id

        :Arguments:
            **new_id** (:obj:`int`): New node_id
        """

        new_id = int(new_id)
        if new_id == self.node_id:
            self._logger.warning("This is already the node number")
            return

        with self.project.db_connection as conn:
            conn.execute("Update Nodes set node_id=? where node_id=?", [new_id, self.node_id])

        self._logger.info(f"Node {self.node_id} was renumbered to {new_id}")
        self.__dict__["node_id"] = new_id
        self.__original__["node_id"] = new_id

    def __save_existing_node(self):
        data = []
        txts = []
        for key, val in self.__dict__.items():
            if key not in self.__original__:
                continue
            if val != self.__original__[key]:
                if key == "geometry" and val is not None:
                    data.append(val.wkb)
                    txts.append(f"geometry=GeomFromWKB(?, {self.__srid__})")
                else:
                    data.append(val)
                    txts.append(f'"{key}"=?')

        if not data:
            self._logger.warning(f"Nothing to update for node {self.node_id}")
            return [], ""

        txts = ",".join(txts) + " where node_id=?"
        data.append(self.node_id)
        sql = f"Update Nodes set {txts}"
        return data, sql

    def connect_mode(
        self,
        mode_id: str,
        link_types="",
        connectors=1,
        conn: Optional[Connection] = None,
        area: Optional[Polygon] = None,
    ):
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

            **mode_id** (:obj:`str`): Mode ID we are trying to connect

            **link_types** (:obj:`str`, *Optional*): String with all the link type IDs that can
            be considered. eg: yCdR. Defaults to ALL link types

            **connectors** (:obj:`int`, *Optional*): Number of connectors to add. Defaults to 1

            **area** (:obj:`Polygon`, *Optional*): Area limiting the search for connectors
        """
        if self.is_centroid != 1 or self.__original__["is_centroid"] != 1:
            self._logger.warning("Connecting a mode only makes sense for centroids and not for regular nodes")
            return

        connector_creation(
            zone_id=self.node_id,
            mode_id=mode_id,
            link_types=link_types,
            connectors=connectors,
            network=self.project.network,
            conn=conn,
            delimiting_area=area,
            proj_nodes=self.project.network.nodes.data,
            proj_links=self.project.network.links.data,
        )

    def __setattr__(self, instance, value) -> None:
        if instance not in self.__dict__ and instance[:1] != "_":
            raise AttributeError(f'"{instance}" is not a valid attribute for a node')
        elif instance == "node_id":
            raise AttributeError("Setting node_id is not allowed")
        elif instance == "link_types":
            raise AttributeError("Setting link_types is not allowed")
        elif instance == "modes":
            raise AttributeError("Setting modes is not allowed")
        elif instance == "is_centroid":
            if value not in [0, 1]:
                raise ValueError("The is_centroid must be either 1 or 0")
        self.__dict__[instance] = value
