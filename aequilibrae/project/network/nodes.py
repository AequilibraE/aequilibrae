from copy import deepcopy

import pandas as pd

from aequilibrae.project.basic_table import BasicTable
from aequilibrae.project.data_loader import DataLoader
from aequilibrae.project.network.node import Node
from aequilibrae.project.table_loader import TableLoader
from aequilibrae.utils.db_utils import commit_and_close
from aequilibrae.utils.spatialite_utils import connect_spatialite


class Nodes(BasicTable):
    """
    Access to the API resources to manipulate the links table in the network

    .. code-block:: python

        >>> from aequilibrae import Project

        >>> proj = Project.from_path("/tmp/test_project")

        >>> all_nodes = proj.network.nodes

        # We can just get one link in specific
        >>> node = all_nodes.get(21)

        # We can save changes for all nodes we have edited so far
        >>> all_nodes.save()
    """

    #: Query sql for retrieving nodes
    sql = ""

    def __init__(self, net):
        super().__init__(net.project)
        self.__table_type__ = "nodes"
        self.__items = {}
        self.__fields = []

        if self.sql == "":
            self.refresh_fields()

    def get(self, node_id: int) -> Node:
        """Get a node from the network by its **node_id**

        It raises an error if node_id does not exist

        :Arguments:
            **node_id** (:obj:`int`): Id of a node to retrieve

        :Returns:
            **node** (:obj:`Node`): Node object for requested node_id
        """

        if node_id in self.__items:
            node = self.__items[node_id]

            # If this element has not been renumbered, we return it. Otherwise we
            # store the object under its new number and carry on
            if node.node_id == node_id:
                return node
            else:
                self.__items[node.node_id] = self.__items.pop(node_id)

        with commit_and_close(connect_spatialite(self.project.path_to_file)) as conn:
            data = conn.execute(f"{self.sql} where node_id=?", [node_id]).fetchone()
        if data:
            data = {key: val for key, val in zip(self.__fields, data)}
            node = Node(data, self.project)
            self.__items[node.node_id] = node
            return node

        raise ValueError(f"Node {node_id} does not exist in the model")

    def refresh_fields(self) -> None:
        """After adding a field one needs to refresh all the fields recognized by the software"""
        tl = TableLoader()
        with commit_and_close(connect_spatialite(self.project.path_to_file)) as conn:
            tl.load_structure(conn, "nodes")
        self.sql = tl.sql
        self.__fields = deepcopy(tl.fields)

    def refresh(self):
        """Refreshes all the nodes in memory"""
        lst = list(self.__items.keys())
        for k in lst:
            del self.__items[k]

    def new_centroid(self, node_id: int) -> Node:
        """Creates a new centroid with a given ID

        :Arguments:
            **node_id** (:obj:`int`): Id of the centroid to be created
        """

        with commit_and_close(connect_spatialite(self.project.path_to_file)) as conn:
            ct = conn.execute("select count(*) from nodes where node_id=?", [node_id]).fetchone()[0]
        if ct > 0:
            raise Exception("Node_id already exists. Failed to create it")

        data = {key: None for key in self.__fields}
        data["node_id"] = node_id
        data["is_centroid"] = 1
        node = Node(data, self.project)
        self.__items[node_id] = node
        return node

    def save(self):
        for item in self.__items.values():
            item.save()

    @property
    def data(self) -> pd.DataFrame:
        """Returns all nodes data as a Pandas DataFrame

        :Returns:
            **table** (:obj:`DataFrame`): Pandas DataFrame with all the nodes, complete with Geometry
        """
        dl = DataLoader(self.project.path_to_file, "nodes")
        return dl.load_table()

    @property
    def lonlat(self) -> pd.DataFrame:
        """Returns all nodes lon/lat coords as a Pandas DataFrame

        :Returns:
            **table** (:obj:`DataFrame`): Pandas DataFrame with all the nodes, with geometry as lon/lat
        """
        with commit_and_close(connect_spatialite(self.project.path_to_file)) as conn:
            df = pd.read_sql("SELECT node_id, ST_X(geometry) AS lon, ST_Y(geometry) AS lat FROM nodes", conn)
        return df

    def __del__(self):
        self.__items.clear()
