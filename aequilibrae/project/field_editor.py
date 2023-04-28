import re
import string
from typing import List

ALLOWED_CHARACTERS = string.ascii_letters + "_0123456789"


class FieldEditor:
    """Allows user to edit the project data tables

    The field editor is used for two different purposes:

    * Managing data tables (adding and removing fields)
    * Editing the tables' metadata (description of each field)

    This is a general class used to manage all project's data tables accessible
    to the user and but it should be accessed directly from within the module
    corresponding to the data table one wants to edit. Example:

    .. code-block:: python

        >>> from aequilibrae import Project

        >>> proj = Project.from_path("/tmp/test_project")

        # To edit the fields of the link_types table
        >>> lt_fields = proj.network.link_types.fields

        # To edit the fields of the modes table
        >>> m_fields = proj.network.modes.fields

    Field descriptions are kept in the table *attributes_documentation*
    """

    _alowed_characters = ALLOWED_CHARACTERS

    def __init__(self, project, table_name: str) -> None:
        self.project = project
        self.logger = project.logger
        self._table = table_name.lower()
        self._table_fields = []
        self._original_values = {}
        self.__update_table_fields()
        self._populate()
        self._check_completeness()

    def _populate(self):
        self._original_values.clear()
        qry = f'Select attribute, description from attributes_documentation where name_table="{self._table}"'
        dt = self.__run_query_fetch_all(qry)

        for attr, descr in dt:
            self.__dict__[attr.lower()] = descr
            self._original_values[attr.lower()] = descr

    def add(self, field_name: str, description: str, data_type="NUMERIC") -> None:
        """Adds new field to the data table

        :Arguments:
            **field_name** (:obj:`str`): Field to be added to the table. Must be a valid SQLite field name
            **description** (:obj:`str`): Description of the field to be inserted in the metadata
            **data_type** (:obj:`str`, optional): Valid SQLite Data type. Default: "NUMERIC"
        """
        if field_name.lower() in self._original_values.keys():
            raise ValueError("attribute_name already exists")
        if field_name in self.__dict__.keys():
            raise ValueError("attribute_name not allowed")

        has_forbidden = [letter for letter in field_name if letter not in self._alowed_characters]
        if has_forbidden:
            raise ValueError('attribute_name can only contain letters, numbers and "_"')

        if field_name[0] in "0123456789":
            raise ValueError("attribute_name cannot begin with a digit")

        self.__update_table_fields()

        if field_name not in self._table_fields:
            self.__run_query_commit(f"Alter table {self._table} add column {field_name} {data_type};")
        self.__adds_to_attribute_table(field_name, description)

    def __update_table_fields(self):
        qry = f"pragma table_info({self._table})"
        dt = self.__run_query_fetch_all(qry)
        self._table_fields = [x[1] for x in dt if x[1] != "ogc_fid"]

    def remove(self, field_name: str) -> None:
        raise NotImplementedError

    def save(self) -> None:
        """Saves any field descriptions which my have been changed to the database"""

        qry = 'update attributes_documentation set description="{}" where attribute="{}" and name_table="{}"'
        for key, val in self._original_values.items():
            new_val = self.__dict__[key]
            if new_val != val:
                self.__run_query_commit(qry.format(new_val, key, self._table))
                self.logger.info(f"Metadata for field {key} on table {self._table} was updated to {new_val}")

    def all_fields(self) -> List[str]:
        """Returns the list of fields available in the database"""
        return list(self._original_values.keys())

    def _check_completeness(self) -> None:
        raw_fields = self._table_fields

        if self._table == "links":
            fields = list({re.sub("_ab", "", re.sub("_ba", "", f)) for f in raw_fields})
        else:
            fields = raw_fields

        for field in fields:
            if field not in self._original_values.keys():
                self.__adds_to_attribute_table(field, "not provided")

        original_fields = list(self._original_values.keys())
        for field in original_fields:
            if field not in fields:
                qry = f'DELETE FROM attributes_documentation where attribute="{field}" and name_table="{self._table}"'
                self.__run_query_commit(qry)
                del self.__dict__[field]
                del self._original_values[field]

    def __adds_to_attribute_table(self, attribute_name, attribute_value):
        self.__dict__[attribute_name] = attribute_value
        self._original_values[attribute_name] = attribute_value
        qry = "insert into attributes_documentation VALUES(?,?,?)"
        vals = (self._table, attribute_name, attribute_value)
        self.__run_query_commit(qry, vals)

    def __run_query_fetch_all(self, qry: str):
        conn = self.project.connect()
        curr = conn.cursor()
        curr.execute(qry)
        dt = curr.fetchall()
        conn.close()
        return dt

    def __run_query_commit(self, qry: str, values=None) -> None:
        conn = self.project.connect()
        if values is None:
            conn.execute(qry)
        else:
            conn.execute(qry, values)
        conn.commit()
        conn.close()
