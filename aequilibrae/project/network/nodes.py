from sqlite3 import Connection
from copy import deepcopy
import pandas as pd
from aequilibrae.project.network.node import Node
from aequilibrae.project.field_editor import FieldEditor
from aequilibrae.project.table_loader import TableLoader
from aequilibrae.project.data_loader import DataLoader
from aequilibrae.project.database_connection import database_connection


class Nodes:
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
        self.__all_nodes = []
        self.conn = database_connection()
        self.curr = self.conn.cursor()
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

        self.curr.execute(f"{self.sql} where node_id=?", [node_id])
        data = self.curr.fetchone()
        if data:
            data = {key: val for key, val in zip(self.__fields, data)}
            node = Node(data)
            self.__items[node.node_id] = node
            return node

        raise ValueError(f"Node {node_id} does not exist in the model")

    def save(self):
        """Saves all nodes that have been retrieved (and edited) so far"""
        nodes = [node for node in self.__items.values()]
        for node in nodes:  # type: Node
            node.save()

    def refresh_fields(self) -> None:
        """After adding a field one needs to refresh all the fields recognized by the software"""
        tl = TableLoader()
        tl.load_structure(self.curr, "nodes")
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

        self.curr.execute("select count(*) from nodes where node_id=?", [node_id])
        if self.curr.fetchone()[0] > 0:
            raise Exception("Node_id already exists. Failed to create it")

        data = {key: None for key in self.__fields}
        data["node_id"] = node_id
        data["is_centroid"] = 1
        node = Node(data)
        self.__items[node_id] = node
        return node

    @property
    def data(self) -> pd.DataFrame:
        """ Returns all nodes data as a Pandas dataFrame

        Returns:
            *table* (:obj:`DataFrame`): Pandas dataframe with all the nodes, complete with Geometry
        """
        dl = DataLoader(self.conn, "nodes")
        return dl.load_table()

    @staticmethod
    def fields() -> FieldEditor:
        """Returns a FieldEditor class instance to edit the Links table fields and their metadata

        Returns:
            *field_editor* (:obj:`FieldEditor`): A field editor configured for editing the Links table
            """
        return FieldEditor("nodes")

    def __copy__(self):
        raise Exception("Links object cannot be copied")

    def __deepcopy__(self, memodict=None):
        raise Exception("Links object cannot be copied")

    def __del__(self):
        self.__items.clear()
