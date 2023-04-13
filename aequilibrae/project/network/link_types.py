from sqlite3 import IntegrityError, Connection
from aequilibrae.project.network.link_type import LinkType
from aequilibrae.project.field_editor import FieldEditor
from aequilibrae.project.table_loader import TableLoader


class LinkTypes:
    """
    Access to the API resources to manipulate the link_types table in the network.

    .. code-block:: python

        >>> from aequilibrae import Project

        >>> p = Project.from_path("/tmp/test_project")

        >>> link_types = p.network.link_types

        # We can get a dictionary of link types in the model
        >>> all_link_types = link_types.all_types()

        # And do a bulk change and save it
        >>> for link_type_id, link_type_obj in all_link_types.items():
        ...     link_type_obj.beta = 1

        # We can save changes for all link types in one go
        >>> link_types.save()

        # or just get one link_type in specific
        >>> default_link_type = link_types.get('y')

        # or just get it by name
        >>> default_link_type = link_types.get_by_name('default')

        # We can change the description of the link types
        >>> default_link_type.description = 'My own new description'

        # Let's say we are using alpha to store lane capacity during the night as 90% of the standard
        >>> default_link_type.alpha = 0.9 * default_link_type.lane_capacity

        # To save this link types we can simply
        >>> default_link_type.save()

        # We can also create a completely new link_type and add to the model
        >>> new_type = link_types.new('a')
        >>> new_type.link_type = 'Arterial'  # Only ASCII letters and *_* allowed # other fields are not mandatory

        # We then save it to the database
        >>> new_type.save()

        # we can even keep editing and save it directly once we have added it to the project
        >>> new_type.lanes = 3
        >>> new_type.lane_capacity = 1100
        >>> new_type.save()

    """

    def __init__(self, net):
        self.__items = {}
        self.project = net.project
        self.logger = net.project.logger
        self.conn = net.conn  # type: Connection
        self.curr = net.conn.cursor()

        tl = TableLoader()
        link_types_list = tl.load_table(self.curr, "link_types")
        existing_list = [lt["link_type_id"] for lt in link_types_list]

        self.__fields = [x for x in tl.fields]
        for lt in link_types_list:
            if lt["link_type_id"] not in self.__items:
                self.__items[lt["link_type_id"]] = LinkType(lt, self.project)

        to_del = [key for key in self.__items.keys() if key not in existing_list]
        for key in to_del:
            del self.__items[key]

    def new(self, link_type_id: str) -> LinkType:
        if link_type_id in self.__items:
            raise ValueError(f"Link Type ID ({link_type_id}) already exists in the model. It must be unique.")

        tp = {key: None for key in self.__fields}
        tp["link_type_id"] = link_type_id
        lt = LinkType(tp, self.project)
        self.__items[link_type_id] = lt
        self.logger.warning("Link type has not yet been saved to the database. Do so explicitly")
        return lt

    def delete(self, link_type_id: str) -> None:
        """Removes the link_type with *link_type_id* from the project"""
        try:
            lt = self.__items[link_type_id]  # type: LinkType
            lt.delete()
            del self.__items[link_type_id]
            self.conn.commit()
        except IntegrityError as e:
            self.logger.error(f"Failed to remove link_type {link_type_id}. {e.args}")
            raise e
        self.logger.warning(f"Link type {link_type_id} was successfully removed from the project database")

    def get(self, link_type_id: str) -> LinkType:
        """Get a link_type from the network by its *link_type_id*"""
        if link_type_id not in self.__items:
            raise ValueError(f"Link type {link_type_id} does not exist in the model")
        return self.__items[link_type_id]

    def get_by_name(self, link_type: str) -> LinkType:
        """Get a link_type from the network by its *link_type* (i.e. name)"""
        for lt in self.__items.values():
            if lt.link_type.lower() == link_type.lower():
                return lt

    def fields(self) -> FieldEditor:
        """Returns a FieldEditor class instance to edit the Link_Types table fields and their metadata"""
        return FieldEditor(self.project.project_base_path, "link_types")

    def all_types(self) -> dict:
        """Returns a dictionary with all LinkType objects available in the model. link_type_id as key"""
        return self.__items

    def save(self):
        for lt in self.__items.values():  # type: LinkType
            lt.save()

    def __copy__(self):
        raise Exception("Link Types object cannot be copied")

    def __deepcopy__(self, memodict=None):
        raise Exception("Link Types object cannot be copied")

    def __del__(self):
        self.__items.clear()
