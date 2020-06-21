import string
from typing import List
from aequilibrae.project.database_connection import database_connection

allowed_characters = string.ascii_letters + '_'


class MetaFields:
    '''Allows user to edit the description to each field for each table

    The results are kept in the table *attributes_documentation*'''
    _alowed_characters = allowed_characters

    def __init__(self, table_name: str) -> None:
        self.__table = table_name
        self._original_values = {}
        self._populate()
        self._check_completeness()

    def _populate(self):
        self._original_values.clear()
        qry = f'Select attribute, description from attributes_documentation where name_table="{self.__table}"'
        dt = self.__run_query_fetch_all(qry)

        for attr, descr in dt:
            self.__dict__[attr] = descr
            self._original_values[attr] = descr

    def add(self, field_name: str, description: str, data_type="NUMERIC") -> None:
        """Adds new field to the data table

        Args:
            *field_name* (:obj:`str`): Field to be added to the table. Must be a valid SQLite field name
            *description* (:obj:`str`): Description of the field to be inserted in the metadata
            *data_type* (:obj:`str`, optional): Valid SQLite Data type. Default: "NUMERIC"
        """
        if field_name in self._original_values.keys():
            raise ValueError('attribute_name already exists')
        if field_name in self.__dict__.keys():
            raise ValueError('attribute_name not allowed')

        has_forbidden = [letter for letter in field_name if letter not in self._alowed_characters]
        if has_forbidden:
            raise ValueError('attribute_name can only contain letters and "_"')

        qry = f'pragma table_info({self.__table})'
        dt = self.__run_query_fetch_all(qry)
        fields = [x[1] for x in dt]
        if field_name not in fields:
            self.__run_query_commit(f'Alter table {self.__table} add column {field_name} {data_type};')
        self.__adds_to_attribute_table(field_name, description)

    def save(self) -> None:
        """Saves any field descriptions which my have been changed to the database"""

        qry = 'update attributes_documentation set description="{}" where attribute="{}" and name_table="{}"'
        for key, val in self._original_values.items():
            new_val = self.__dict__[key]
            if new_val != val:
                self.__run_query_commit(qry.format(new_val, key, self.__table))

    def all_fields(self) -> List[str]:
        """Returns the list of fields available in the database"""
        return list(self._original_values.keys())

    def __adds_to_attribute_table(self, attribute_name, attribute_value):
        self.__dict__[attribute_name] = attribute_value
        self._original_values[attribute_name] = attribute_value
        qry = 'insert into attributes_documentation VALUES(?,?,?)'
        vals = (self.__table, attribute_name, attribute_value)
        self.__run_query_commit(qry, vals)

    def __run_query_fetch_all(self, qry: str):
        conn = database_connection()
        curr = conn.cursor()
        curr.execute(qry)
        dt = curr.fetchall()
        conn.close()
        return dt

    def __run_query_commit(self, qry: str, values=None) -> None:
        conn = database_connection()
        if values is None:
            conn.execute(qry)
        else:
            conn.execute(qry, values)
        conn.commit()
        conn.close()

    def _check_completeness(self) -> None:
        qry = f'pragma table_info({self.__table})'
        dt = self.__run_query_fetch_all(qry)
        fields = [x[1] for x in dt if x[1] != 'ogc_fid']
        for field in fields:
            if field not in self._original_values.keys():
                self.__adds_to_attribute_table(field, 'not provided')

        original_fields = list(self._original_values.keys())
        for field in original_fields:
            if field not in fields:
                qry = f'DELETE FROM attributes_documentation where attribute="{field}" and name_table="{self.__table}"'
                self.__run_query_commit(qry)
                del self.__dict__[field]
                del self._original_values[field]
