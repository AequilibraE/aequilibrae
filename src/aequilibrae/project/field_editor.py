import logging
import re
import string
from typing import List, NoReturn

ALLOWED_CHARACTERS = string.ascii_letters + "_0123456789"

logger = logging.getLogger(__name__)


class FieldEditor:
    """Allows user to edit the project data tables

    The field editor is used for two different purposes:

    * Managing data tables (adding and removing fields)
    * Editing the tables' metadata (description of each field)

    This is a general class used to manage all project's data tables accessible
    to the user and but it should be accessed directly from within the module
    corresponding to the data table one wants to edit. Example:

    .. code-block:: python

        >>> project = create_example(project_path)

        # To edit the fields of the link_types table
        >>> lt_fields = project.tables.link_types.fields

        # To edit the fields of the modes table
        >>> m_fields = project.tables.modes.fields

        >>> project.close()

    Field descriptions are kept in the table *attributes_documentation* and
    are accessible as lowercase attributes on the editor.
    """

    _allowed_characters = ALLOWED_CHARACTERS

    def __init__(self, connection, table_name: str) -> None:
        self._transaction_manager = connection
        self._table = table_name.lower()
        self._table_fields = {}
        self._original_values = {}
        # Keep the stored spelling for metadata writes, even though editor attributes are lowercase.
        self._attribute_names = {}
        self.__update_table_fields()
        self._populate()
        self._check_completeness()

    def _populate(self) -> None:
        self._original_values.clear()
        self._attribute_names.clear()
        qry = "SELECT attribute, description FROM attributes_documentation WHERE name_table=?"
        for attribute, description in self.__run_query_fetch_all(qry, (self._table,)):
            field_key = attribute.lower()
            self.__dict__[field_key] = description
            self._original_values[field_key] = description
            self._attribute_names[field_key] = attribute

    def add(self, field_name: str, description: str, data_type="NUMERIC") -> None:
        """Adds new field to the data table

        :Arguments:
            **field_name** (:obj:`str`): Field to be added to the table. Must be a valid SQLite field name

            **description** (:obj:`str`): Description of the field to be inserted in the metadata

            **data_type** (:obj:`str`, *Optional*): Valid SQLite Data type. Default: "NUMERIC"
        """
        field_key = field_name.lower()
        if field_key in self._original_values:
            raise ValueError("attribute_name already exists")
        if field_key in self.__dict__:
            raise ValueError("attribute_name not allowed")

        has_forbidden = [letter for letter in field_name if letter not in self._allowed_characters]
        if has_forbidden:
            raise ValueError('attribute_name can only contain letters, numbers and "_"')

        if field_name[0] in "0123456789":
            raise ValueError("attribute_name cannot begin with a digit")

        self.__update_table_fields()

        if field_key in self._table_fields:
            field_name = self._table_fields[field_key]
        else:
            self.__run_query_commit(f"Alter table {self._table} add column {field_name} {data_type};")
        self.__adds_to_attribute_table(field_name, description)

    def __update_table_fields(self) -> None:
        qry = "SELECT name FROM pragma_table_info(?)"
        rows = self.__run_query_fetch_all(qry, (self._table,))
        self._table_fields = {name.lower(): name for (name,) in rows if name.lower() != "ogc_fid"}

    def remove(self, field_name: str) -> NoReturn:
        raise NotImplementedError

    def save(self) -> None:
        """
        Saves any field descriptions which my have been changed to the database and update layer statistics.

        This is required for new fields to appear in applications like QGIS.
        """

        qry = "UPDATE attributes_documentation SET description=? WHERE attribute=? AND name_table=?"
        for key, val in self._original_values.items():
            new_val = self.__dict__[key]
            if new_val != val:
                self.__run_query_commit(qry, (new_val, self._attribute_names[key], self._table))
                logger.info(f"Metadata for field {key} on table {self._table} was updated to {new_val}")

        logger.info(f"Updating layer statistics for {self._table}, this may take a moment")
        with self._transaction_manager.transaction() as conn:
            conn.execute(f"SELECT InvalidateLayerStatistics('{self._table}');")
            conn.execute(f"SELECT UpdateLayerStatistics('{self._table}');")
        logger.info(f"Updated layer statistics for {self._table}")

    def all_fields(self) -> List[str]:
        """Returns the list of fields available in the database"""
        return list(self._original_values.keys())

    def _check_completeness(self) -> None:
        fields = self._table_fields
        if self._table == "links":
            fields = {}
            for column in self._table_fields.values():
                field = re.sub("_ba", "", column, flags=re.IGNORECASE)
                field = re.sub("_ab", "", field, flags=re.IGNORECASE)
                fields.setdefault(field.lower(), field)

        for field_key, field in fields.items():
            if field_key not in self._original_values:
                self.__adds_to_attribute_table(field, "not provided")

        for field_key in list(self._original_values):
            if field_key not in fields:
                qry = "DELETE FROM attributes_documentation WHERE attribute=? AND name_table=?"
                self.__run_query_commit(qry, (self._attribute_names[field_key], self._table))
                del self.__dict__[field_key]
                del self._original_values[field_key]
                del self._attribute_names[field_key]

    def __adds_to_attribute_table(self, attribute_name: str, attribute_value: str) -> None:
        qry = "insert into attributes_documentation VALUES(?,?,?)"
        vals = (self._table, attribute_name, attribute_value)
        self.__run_query_commit(qry, vals)
        field_key = attribute_name.lower()
        self.__dict__[field_key] = attribute_value
        self._original_values[field_key] = attribute_value
        self._attribute_names[field_key] = attribute_name

    def __run_query_fetch_all(self, qry: str, values=()):
        return self._transaction_manager._connection.execute(qry, values).fetchall()

    def __run_query_commit(self, qry: str, values=None) -> None:
        with self._transaction_manager.transaction() as conn:
            if values is None:
                conn.execute(qry)
            else:
                conn.execute(qry, values)
