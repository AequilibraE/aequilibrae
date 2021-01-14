import os
from copy import deepcopy
import shapely.wkb
import pandas as pd
from aequilibrae.project.network.link import Link
from aequilibrae import logger
from aequilibrae.project.field_editor import FieldEditor
from aequilibrae.project.table_loader import TableLoader
from aequilibrae.project.data_loader import DataLoader
from aequilibrae.project.database_connection import database_connection


class Links:
    """
    Access to the API resources to manipulate the links table in the network

    ::

        from aequilibrae import Project

        proj = Project()
        proj.open('path/to/project/folder')

        all_links = proj.network.links

        # We can just get one link in specific
        link = all_links.get(4523)

        # We can save changes for all links we have edited so far
        all_links.save()
    """

    __items = {}
    __all_links = []
    __fields = []
    __max_id = -1

    #: Query sql for retrieving links
    sql = ""

    def __init__(self):
        self.conn = database_connection()
        self.curr = self.conn.cursor()
        if self.sql == "":
            self.refresh_fields()

    def get(self, link_id: int) -> Link:
        """Get a link from the network by its **link_id**

        It raises an error if link_id does not exist

        Args:
            *link_id* (:obj:`int`): Id of a link to retrieve

        Returns:
            *link* (:obj:`Link`): Link object for requested link_id
            """
        link_id = int(link_id)
        if link_id in self.__items:
            link = self.__items[link_id]
            if not link._exists():
                raise Exception("Link was deleted")
            return link
        data = self.__link_data(link_id)
        if data:
            return self.__create_return_link(data)
        self.__existence_error(link_id)

    def new(self) -> Link:
        """Creates a new link

            Returns:
                *link* (:obj:`Link`): A new link object populated only with link_id (not saved in the model yet)
                """

        data = {key: None for key in self.__fields}
        data["direction"] = 0
        data["link_type"] = "default"
        data["link_id"] = self.__new_link_id()
        return Link(data)
        # return self.__create_return_link(data)

    def copy_link(self, link_id: int) -> Link:
        """Creates a copy of a link with a new id

        It raises an error if link_id does not exist

        Args:
            *link_id* (:obj:`int`): Id of the link to copy

        Returns:
            *link* (:obj:`Link`): Link object for requested link_id
            """

        data = self.__link_data(int(link_id))
        data["link_id"] = self.__new_link_id()

        # The geometry wrangling is just a workaround to signalize that the link is new
        # That allows saving of the link to work properly
        geo = data["geometry"]
        data["geometry"] = None
        link = self.__create_return_link(data)
        link.geometry = shapely.wkb.loads(geo)

        return link

    def delete(self, link_id: int) -> None:
        """Removes the link with **link_id** from the project

        Args:
            *link_id* (:obj:`int`): Id of a link to delete"""
        d = 1
        link_id = int(link_id)
        if link_id in self.__items:
            link = self.__items.pop(link_id)  # type: Link
            link.delete()
        else:
            self.curr.execute("Delete from Links where link_id=?", [link_id])
            d = self.curr.rowcount
            self.conn.commit()
        if d:
            logger.warning(f"Link {link_id} was successfully removed from the project database")
        else:
            self.__existence_error(link_id)

    def save(self):
        for link in self.__items.values():  # type: Link
            link.save()

    def refresh_fields(self) -> None:
        """After adding a field one needs to refresh all the fields recognized by the software"""
        self.curr.execute("select max(link_id) from Links")
        self.__max_id = self.curr.fetchone()[0]
        tl = TableLoader()
        tl.load_structure(self.curr, "links")
        self.sql = tl.sql
        self.__fields = deepcopy(tl.fields)

    @property
    def data(self) -> pd.DataFrame:
        """ Returns all links data as a Pandas dataFrame

        Returns:
            *table* (:obj:`DataFrame`): Pandas dataframe with all the links, complete with Geometry
        """
        dl = DataLoader(self.conn, "links")
        return dl.load_table()

    def refresh(self):
        """Refreshes all the links in memory"""
        lst = list(self.__items.keys())
        for k in lst:
            del self.__items[k]

    @staticmethod
    def fields() -> FieldEditor:
        """Returns a FieldEditor class instance to edit the Links table fields and their metadata

        Returns:
            *field_editor* (:obj:`FieldEditor`): A field editor configured for editing the Links table
            """
        return FieldEditor("links")

    def __copy__(self):
        raise Exception("Links object cannot be copied")

    def __deepcopy__(self, memodict=None):
        raise Exception("Links object cannot be copied")

    def __del__(self):
        self.__items.clear()

    def __existence_error(self, link_id):
        raise ValueError(f"Link {link_id} does not exist in the model")

    def __link_data(self, link_id: int) -> dict:
        self.curr.execute(f"{self.sql} where link_id=?", [link_id])
        data = self.curr.fetchone()
        if data:
            return {key: val for key, val in zip(self.__fields, data)}
        raise ValueError("Link_id does not exist on the network")

    def __new_link_id(self):
        self.__max_id += 1
        return self.__max_id

    def __create_return_link(self, data):
        link = Link(data)
        self.__items[link.link_id] = link
        return link
