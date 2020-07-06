from sqlite3 import IntegrityError, Connection
from aequilibrae.project.network.link_type import LinkType
from aequilibrae import logger
from aequilibrae.project.field_editor import FieldEditor


class LinkTypes:
    """
    Access to the API resources to manipulate the link_types table in the network

    ::

        from aequilibrae import Project
        from aequilibrae.project.network import LinkType

        p = Project()
        p.open('path/to/project/folder')

        link_types = p.network.link_types

        # We can get a dictionary of all modes in the model
        all_link_types = modes.all_types()

        #And do a bulk change and save it
        for link_type_id, link_type_obj in all_link_types.items():
            link_type_obj.beta = 1
            link_type_obj.save()

        # or just get one link_type in specific
        default_link_type = link_types.get('y')

        # or just get it by name
        default_link_type = link_types.get('default')

        # We can change the description of the mode
        default_link_type.description = 'My own new description'

        # Let's say we are using alpha to store lane capacity during the night as 90% of the standard
        default_link_type.alpha =0.9 * default_link_type.lane_capacity

        # To save this mode we can simply
        default_link_type.save()

        # We can also create a completely new link_type and add to the model
        new_type = link_type('a')
        new_type.link_type_name = 'Arterial'  # Only ASCII letters and *_* allowed
        # other fields are not mandatory

        # We then explicitly add it to the network
        link_types.add(new_type)

        # we can even keep editing and save it directly once we have added it to the project
        new_type.lanes = 3
        new_type.lane_capacity = 1100
        new_type.save()
    """

    def __init__(self, net):
        self.__all_types = []
        self.conn = net.conn  # type: Connection
        self.curr = net.conn.cursor()
        if self.__has_table():
            self.__update_list_of_link_types()

    def add(self, link_type: LinkType) -> None:
        """ We add a new link type to the project"""
        self.__update_list_of_link_types()
        if link_type.link_type_id in self.__all_types:
            raise ValueError("Link type already exists in the model")

        self.curr.execute("insert into 'link_types'(link_type_id, link_type) Values(?,?)",
                          [link_type.link_type_id, link_type.link_type])
        self.conn.commit()
        logger.info(f'Link Type {link_type.link_type}({link_type.link_type_id}) was added to the project')

        link_type.save()
        self.__update_list_of_link_types()

    def drop(self, link_type_id: str) -> None:
        """Removes the link_type with **link_type_id** from the project"""
        try:
            self.curr.execute(f'delete from link_types where link_type_id="{link_type_id}"')
            self.conn.commit()
        except IntegrityError as e:
            logger.error(f'Failed to remove link_type {link_type_id}. {e.args}')
            raise e
        logger.warning(f'Link type {link_type_id} was successfully removed from the project database')
        self.__update_list_of_link_types()

    def get(self, link_type_id: str) -> LinkType:
        """Get a link_type from the network by its **link_type_id**"""
        self.__update_list_of_link_types()
        if link_type_id not in self.__all_types:
            raise ValueError(f'Link type {link_type_id} does not exist in the model')
        return LinkType(link_type_id)

    def get_by_name(self, link_type: str) -> LinkType:
        """Get a link_type from the network by its **link_type_id**"""
        self.__update_list_of_link_types()
        self.curr.execute(f"select link_type_id from 'link_types' where link_type='{link_type}'")
        found = self.curr.fetchone()
        if len(found) == 0:
            raise ValueError(f'Link type {link_type} does not exist in the model')
        return LinkType(found[0])

    def all_types(self) -> dict:
        """Returns a dictionary with all LinkType objects available in the model. link_type_id as key"""
        self.__update_list_of_link_types()
        return {x: LinkType(x) for x in self.__all_types}

    def fields(self) -> FieldEditor:
        """Returns a FieldEditor class instance to edit the Link_Types table fields and their metadata"""
        return FieldEditor('link_types')

    def __update_list_of_link_types(self) -> None:
        self.curr.execute("select link_type_id from 'link_types'")
        self.__all_types = [x[0] for x in self.curr.fetchall()]

    def __has_table(self):
        curr = self.conn.cursor()
        curr.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return any(['link_types' in x[0] for x in curr.fetchall()])
