from copy import deepcopy

import pandas as pd

from aequilibrae.project.basic_table import BasicTable
from aequilibrae.project.data_loader import DataLoader
from aequilibrae.project.network.node import Node
from aequilibrae.project.table_loader import TableLoader


class Nodes(BasicTable):
    """
    Access to the API resources to manipulate the links table in the network

    ::

        from aequilibrae import Project

        proj = Project()
        proj.open('path/to/project/folder')

        all_nodes = proj.network.nodes

        # We can just get one link in specific
        node = all_nodes.get(7894)

        # We can save changes for all nodes we have edited so far
        all_nodes.save()
    """

    __items = {}
    __fields = []

    #: Query sql for retrieving nodes
    sql = ""

    def __init__(self):
        super().__init__()
        self.__table_type__ = 'nodes'
        self.__all_nodes = []
        if self.sql == "":
            self.refresh_fields()

    def get(self, node_id: int) -> Node:
        """Get a node from the network by its **node_id**

        It raises an error if node_id does not exist

        Args:
            *node_id* (:obj:`int`): Id of a node to retrieve

        Returns:
            *node* (:obj:`Node`): Node object for requested node_id
            """

        if node_id in self.__items:
            node = self.__items[node_id]

            # If this element has not been renumbered, we return it. Otherwise we
            # store the object under its new number and carry on
            if node.node_id == node_id:
                return node
            else:
                self.__items[node.node_id] = self.__items.pop(node_id)

        self._curr.execute(f"{self.sql} where node_id=?", [node_id])
        data = self._curr.fetchone()
        if data:
            data = {key: val for key, val in zip(self.__fields, data)}
            node = Node(data)
            self.__items[node.node_id] = node
            return node

        raise ValueError(f"Node {node_id} does not exist in the model")

    def refresh_fields(self) -> None:
        """After adding a field one needs to refresh all the fields recognized by the software"""
        tl = TableLoader()
        tl.load_structure(self._curr, "nodes")
        self.sql = tl.sql
        self.__fields = deepcopy(tl.fields)

    def refresh(self):
        """Refreshes all the nodes in memory"""
        lst = list(self.__items.keys())
        for k in lst:
            del self.__items[k]

    def new_centroid(self, node_id: int) -> Node:
        """Creates a new centroid with a given ID

        Args:
            *node_id* (:obj:`int`): Id of the centroid to be created
        """

        self._curr.execute("select count(*) from nodes where node_id=?", [node_id])
        if self._curr.fetchone()[0] > 0:
            raise Exception("Node_id already exists. Failed to create it")

        data = {key: None for key in self.__fields}
        data["node_id"] = node_id
        data["is_centroid"] = 1
        node = Node(data)
        self.__items[node_id] = node
        return node

    def save(self):
        for item in self.__items.values():
            item.save()

    @property
    def data(self) -> pd.DataFrame:
        """ Returns all nodes data as a Pandas dataFrame

        Returns:
            *table* (:obj:`DataFrame`): Pandas dataframe with all the nodes, complete with Geometry
        """
        dl = DataLoader(self.conn, "nodes")
        return dl.load_table()

    def __del__(self):
        self.__items.clear()
