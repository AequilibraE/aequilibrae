from typing import Any

from aequilibrae.project.project_table import _SELECT_ONE_SQL, NonSpatialProjectTable


class LinkTypes(NonSpatialProjectTable):
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

        # Coordinate several writes with the project transaction
        >>> with project.transaction():
        ...     for lt in link_types:
        ...         link_types.update(lt.link_type_id, beta=1)

        >>> project.close()
    """

    name = "link_types"
    key = "link_type_id"
    record_name = "LinkTypeRecord"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._select_by_name_sql = _SELECT_ONE_SQL.format(table=self.name, key="link_type", columns="*")

    def get_by_name(self, link_type: str) -> Any:
        """Get a link-type record by its descriptive name.

        :Arguments:
            **link_type** (:obj:`str`): Descriptive link-type name.

        :Returns:
            **link type** (:obj:`Any`): Generated frozen link-type record.
        """
        self._refresh_record_type()
        row = self._connection._connection.execute(self._select_by_name_sql, [link_type]).fetchone()
        if row is None:
            raise ValueError(f"Link type {link_type} does not exist in the model")
        return self._build_record(row)
