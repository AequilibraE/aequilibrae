from sqlite3 import Connection
from copy import deepcopy
from aequilibrae.project.network.link import Link
from aequilibrae import logger
from aequilibrae.project.field_editor import FieldEditor
from aequilibrae.project.table_loader import TableLoader
from aequilibrae.project.network.link_types import LinkTypes
from aequilibrae.project.network.modes import Modes


class Links:
    """
    Access to the API resources to manipulate the links table in the network

    ::

        from aequilibrae import Project

        p = Project()
        p.open('path/to/project/folder')

        all_links = p.network.links

        # We can just get one link in specific
        link = all_links.get(4523)

        # We can find out which fields exist for the links
        link.fields()

        # And edit each one like this
        link.lanes_ab = 3
        link.lanes_ba = 2


        # we can drop a mode from the link
        link.drop_mode('c')

        # we can add a mode to the link
        link.add_mode('m')

        # Or set all modes at once
        link.set_modes('cmtw')

        # We can just save the link
        link.save()

        # We can save changes for all links we have edited so far
        all_links.save()
    """
    __items = {}

    #: Query sql for retrieving links
    sql = ''

    def __init__(self, net):
        self.__all_links = []
        self.conn = net.conn  # type: Connection
        self.curr = net.conn.cursor()
        self.__link_types = net.link_types  # type: LinkTypes
        self.__modes = net.modes  # type: Modes
        tl = TableLoader()
        tl.load_structure(self.curr, 'links')
        self.sql = tl.sql

        self.__fields = deepcopy(tl.fields)

    def get(self, link_id: int) -> Link:
        """Get a link from the network by its **link_id**

        It raises an error if link_id does not exist"""

        self.curr.execute(f'{self.sql} where link_id=?', [link_id])
        data = self.curr.fetchone()
        if data:
            data = {key: val for key, val in zip(self.__fields, data)}
            link = Link(data, self.__link_types, self.__modes)
            self.__items[link_id] = link
        else:
            self.__existence_error(link_id)
        return link

    def new(self) -> Link:
        tp = {key: None for key in self.__fields}
        link = Link(tp, self.__link_types, self.__modes)

        self.__items[link.link_id] = link
        return link

    def delete(self, link_id: int) -> None:
        """Removes the link with **link_id** from the project"""
        d = 1
        if link_id in self.__items:
            link = self.__items[link_id]  # type: Link
            link.delete()
            del self.__items[link_id]
        else:
            self.curr.execute('Delete from Links where link_id=?', [link_id])
            d = self.curr.rowcount
            self.conn.commit()
        if d:
            logger.warning(f'Link {link_id} was successfully removed from the project database')
        else:
            self.__existence_error(link_id)

    def fields(self) -> FieldEditor:
        """Returns a FieldEditor class instance to edit the Links table fields and their metadata"""
        return FieldEditor('links')

    def save(self):
        for lt in self.__items.values():  # type: Link
            lt.save()

    def __copy__(self):
        raise Exception('Links object cannot be copied')

    def __deepcopy__(self, memodict=None):
        raise Exception('Links object cannot be copied')

    def __del__(self):
        self.__items.clear()

    def __existence_error(self, link_id):
        raise ValueError(f'Link {link_id} does not exist in the model')
