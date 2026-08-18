import string

from aequilibrae.project.project_table import ProjectTable


class LinkTypes(ProjectTable):
    """
    Access to the API resources to manipulate the link_types table in the network.

    .. code-block:: python

        >>> project = create_example(project_path)

        >>> link_types = project.network.link_types

        # We can get a link type as an immutable record
        >>> default_type = link_types.get('y')

        # or by name
        >>> default_type = link_types.get_by_name('default')

        # and write changes explicitly
        >>> link_types.update('y', description='My own new description', lanes=3)

        # Creating a new link type is a single insert
        >>> link_types.insert(link_type_id='a', link_type='Arterial', lanes=3, lane_capacity=1100)
        'a'

        # Bulk changes go in one batch
        >>> with link_types.batch() as batch:
        ...     for lt in link_types:
        ...         batch.update(lt.link_type_id, beta=1)

        >>> project.close()
    """

    name = "link_types"
    key = "link_type_id"
    record_name = "LinkTypeRecord"

    __allowed_characters = string.ascii_letters + "_"

    def __init__(self, net):
        super().__init__(net.transactions)

    def get_by_name(self, link_type: str):
        """Get a link type record from the network by its *link_type* (i.e. name)"""
        for lt in self:
            if lt.link_type == link_type:
                return lt
        raise ValueError(f"Link type {link_type} does not exist in the model")

    def _check_link_type(self, value) -> str:
        if not isinstance(value, str):
            raise ValueError("link_type must be string")
        if not len(value):
            raise ValueError("link_type cannot be zero-length")
        for letter in value:
            if letter not in self.__allowed_characters:
                raise ValueError('link_type can only contain letters and "_"')
        return value
