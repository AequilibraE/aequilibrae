import logging
import string
from sqlite3 import Connection
from typing import TYPE_CHECKING, NoReturn

from aequilibrae.utils.db_utils import escape_identifier

if TYPE_CHECKING:
    from aequilibrae.project.project_table import ProjectTable

ALLOWED_CHARACTERS = string.ascii_letters + "_0123456789"
_DIRECTION_SUFFIXES = ("_ab", "_ba")
_ADD_COLUMN_SQL = "ALTER TABLE {table} ADD COLUMN {column} {data_type}"
_GET_DESCRIPTION_SQL = (
    "SELECT description FROM attributes_documentation WHERE name_table=? AND attribute=? COLLATE NOCASE "
    "ORDER BY attribute != ? LIMIT 1"
)
_UPDATE_DESCRIPTION_SQL = (
    "UPDATE attributes_documentation SET description=? WHERE name_table=? AND attribute=? COLLATE NOCASE"
)
_INSERT_DESCRIPTION_SQL = "INSERT INTO attributes_documentation (name_table, attribute, description) VALUES (?, ?, ?)"

_FieldSpec = tuple[str, str] | tuple[str, str, str] | tuple[str, str, str, bool]

logger = logging.getLogger(__name__)


class FieldEditor:
    """
    Allows user to edit the project data tables

    The field editor is used for two different purposes:

    * Managing data tables (adding fields)
    * Editing the tables' metadata (description of each field)

    This is a general class used to manage all project's data tables accessible
    to the user, but it should be accessed directly from within the module
    corresponding to the data table one wants to edit. Example:

    .. code-block:: python

        >>> project = create_example(project_path)

        # To edit the fields of the link_types table
        >>> lt_fields = project.tables.link_types.fields

        # To edit the fields of the modes table
        >>> m_fields = project.tables.modes.fields

        >>> project.close()

    Field descriptions are kept in the table *attributes_documentation*. Use
    :meth:`get_description` and :meth:`set_description` to read and edit them.
    Field names are case-insensitive.
    """

    __slots__ = ("_table",)

    def __init__(self, table: "ProjectTable") -> None:
        self._table = table

    def add(self, field_name: str, description: str, data_type: str = "NUMERIC", *, directional: bool = False) -> None:
        """
        Adds new field to the data table

        :param field_name: Field to be added to the table. Must be a valid
            SQLite field name.
        :param description: Description of the field to be inserted in the
            metadata.
        :param data_type: Valid SQLite data type. Default: "NUMERIC".
        :param directional: On the links table, add both ``_ab`` and ``_ba``
            columns for the given base name.

        Both directions share a description. When adding the second direction
        separately, use :meth:`set_description` to change an existing
        description.  A directional link column must have a name before its
        ``_ab`` or ``_ba`` suffix.
        """
        self.add_many([(field_name, description, data_type, directional)])

    def add_many(self, fields: list[_FieldSpec]) -> None:
        """
        Add fields to the data table in one transaction, in the order supplied.

        :param fields: Tuples of ``(field_name, description[, data_type[,
            directional]])``.  Omitted values use the defaults from
            :meth:`add`.
        """
        if not fields:
            return

        # ProjectTable imports FieldEditor to expose the table's fields property.
        from aequilibrae.project.project_table import SpatialProjectTable

        table_name = self._table.name
        with self._table._connection.transaction() as conn:
            columns_by_name = self._columns_by_name()
            for field_spec in fields:
                if not isinstance(field_spec, tuple):
                    raise TypeError("each field specification must be a tuple")
                self._add_field(conn, columns_by_name, *field_spec)

            if isinstance(self._table, SpatialProjectTable):
                # Layer statistics need updating for new fields to appear in applications like QGIS.
                logger.info(f"Updating layer statistics for {table_name}, this may take a moment")
                conn.execute("SELECT InvalidateLayerStatistics(?)", (table_name,)).fetchone()
                conn.execute("SELECT UpdateLayerStatistics(?)", (table_name,)).fetchone()
                logger.info(f"Updated layer statistics for {table_name}")

            self._table._refresh_record_type()

    def remove(self, field_name: str) -> NoReturn:
        raise NotImplementedError

    def all_fields(self) -> list[str]:
        """
        Returns the list of fields available in the database.

        Directional link fields appear once under their lowercase base name.
        Use the table's ``columns`` property to list the physical columns.
        """
        names = []
        for column in self._table.columns:
            suffix = self._link_direction_suffix(column)
            name = column[: -len(suffix)] if suffix else column
            names.append(name.lower())

        return list(dict.fromkeys(names))

    def get_description(self, field_name: str) -> str | None:
        """
        Return the stored description, or ``"not provided"`` if it has not been documented.

        :param field_name: Column name, or ``name_*`` for a directional link
            field.  Use the base name when there is no unsuffixed column of
            that name.  :raises ValueError: If the field does not exist.
        """
        keys = self._description_keys(field_name, self._columns_by_name())
        existing = self._find_description(self._table._connection._connection, keys)
        return "not provided" if existing is None else existing[0]

    def set_description(self, field_name: str, description: str) -> None:
        """
        Write the description of an existing field immediately.

        :param field_name: Column name or directional link field, as in
            :meth:`get_description`.
        :param description: New description for the field.

        :raises ValueError: If the field does not exist.
        """
        self.set_descriptions([(field_name, description)])

    def set_descriptions(self, descriptions: list[tuple[str, str]]) -> None:
        """
        Write descriptions of existing fields in one transaction, in the order supplied.

        :param descriptions: Tuples of ``(field_name, description)``. Field
            names use the same rules as :meth:`get_description`.

        :raises ValueError: If a field does not exist.
        """
        if not descriptions:
            return

        updates = []
        with self._table._connection.transaction() as conn:
            columns_by_name = self._columns_by_name()
            for entry in descriptions:
                if not isinstance(entry, tuple):
                    raise TypeError("each field description must be a tuple")

                field_name, description = entry
                keys = self._description_keys(field_name, columns_by_name)
                attribute = keys[0]
                self._write_description(conn, attribute, description)
                updates.append((attribute, description))

            self._table._refresh_record_type()

        for attribute, description in updates:
            logger.info(f"Metadata for field {attribute} on table {self._table.name} was updated to {description}")

    def _add_field(
        self,
        conn: Connection,
        columns_by_name: dict[str, str],
        field_name: str,
        description: str | None,
        data_type: str = "NUMERIC",
        directional: bool = False,
    ) -> None:
        """Add one field in the batch transaction and update its column lookup."""
        if not isinstance(field_name, str):
            raise TypeError("field_name must be a string")
        if not field_name or any(letter not in ALLOWED_CHARACTERS for letter in field_name):
            raise ValueError('attribute_name can only contain letters, numbers and "_"')
        if field_name[0].isdigit():
            raise ValueError("attribute_name cannot begin with a digit")
        if field_name.lower() == "ogc_fid":
            raise ValueError("attribute_name not allowed")

        table_name = self._table.name
        suffix = self._link_direction_suffix(field_name)
        base = field_name[: -len(suffix)] if suffix else field_name
        if not base:
            raise ValueError("directional link columns require a non-empty base name")
        if directional and (table_name != "links" or suffix is not None):
            raise ValueError("directional fields require a base field name on the links table")
        if not isinstance(data_type, str):
            raise TypeError("data_type must be a SQLite type name")

        new_columns = (field_name,)
        if directional:
            new_columns = tuple(base + direction for direction in _DIRECTION_SUFFIXES)

        quoted_table = escape_identifier(table_name)
        quoted_type = escape_identifier(data_type)
        if any(column.lower() in columns_by_name for column in new_columns):
            raise ValueError("attribute_name already exists")

        attribute = field_name
        if directional:
            attribute = base + "_*"
        elif table_name == "links":
            attribute, description = self._link_description_for_add(conn, columns_by_name, base, suffix, description)

        for column in new_columns:
            sql = _ADD_COLUMN_SQL.format(
                table=quoted_table,
                column=escape_identifier(column),
                data_type=quoted_type,
            )
            conn.execute(sql)
            columns_by_name[column.lower()] = column

        self._write_description(conn, attribute, description)

    def _link_description_for_add(
        self,
        conn: Connection,
        columns_by_name: dict[str, str],
        base: str,
        suffix: str | None,
        description: str | None,
    ) -> tuple[str, str | None]:
        """Resolve the metadata write without replacing an existing shared description."""
        if suffix is not None:
            opposite_suffix = "_ba" if suffix == "_ab" else "_ab"
            if base.lower() + opposite_suffix in columns_by_name:
                keys = self._directional_description_keys(base, columns_by_name)
                existing = self._find_description(conn, keys)
                if existing is not None:
                    return keys[0], existing[0]
            return base + "_*", description

        # An unsuffixed column takes over its own name. Move any legacy shared
        # description to the wildcard key before writing the description for
        # the new column.
        keys = self._directional_description_keys(base, columns_by_name)
        existing = self._find_description(conn, keys)
        if existing is not None:
            shared_description, matched_key = existing
            wildcard_key = keys[0]
            if matched_key.lower() != wildcard_key.lower():
                self._write_description(conn, wildcard_key, shared_description)

        return base, description

    def _columns_by_name(self) -> dict[str, str]:
        return {column.lower(): column for column in self._table.columns}

    def _link_direction_suffix(self, column: str) -> str | None:
        suffix = column[-3:].lower()
        if self._table.name == "links" and suffix in _DIRECTION_SUFFIXES:
            return suffix
        return None

    def _description_keys(self, field_name: str, columns_by_name: dict[str, str]) -> tuple[str, ...]:
        """Resolve a field name to its write key first, then any read fallbacks."""
        if not isinstance(field_name, str):
            raise TypeError("field_name must be a string")

        column = columns_by_name.get(field_name.lower())
        if column is not None:
            suffix = self._link_direction_suffix(column)
            if suffix is None:
                return (column,)
            field_name = column[: -len(suffix)] + "_*"

        if self._table.name == "links":
            base = field_name.removesuffix("_*")
            keys = self._directional_description_keys(base, columns_by_name)
            if keys:
                return keys

        raise ValueError(f"{self._table.name} has no field {field_name!r}")

    @staticmethod
    def _directional_description_keys(base: str, columns_by_name: dict[str, str]) -> tuple[str, ...]:
        """Return the wildcard key first, followed by legacy keys in lookup order."""
        base_key = base.lower()
        direction_columns = []
        for suffix in _DIRECTION_SUFFIXES:
            column = columns_by_name.get(base_key + suffix)
            if column is not None:
                direction_columns.append(column)

        if not direction_columns:
            return ()

        # Older editors stored descriptions under the base name or individual
        # columns.  Only use the base name as a fallback if it is not a
        # physical column in its own right.
        stored_base = direction_columns[0][:-3]
        if base_key in columns_by_name:
            return (stored_base + "_*", *direction_columns)
        return (stored_base + "_*", stored_base, *direction_columns)

    def _find_description(self, conn: Connection, keys: tuple[str, ...]) -> tuple[str | None, str] | None:
        """Return the description and the first metadata key it matched."""
        for key in keys:
            row = conn.execute(_GET_DESCRIPTION_SQL, (self._table.name, key, key)).fetchone()
            if row is not None:
                return row[0], key
        return None

    def _write_description(self, conn: Connection, attribute: str, description: str | None) -> None:
        # Keep the stored spelling for metadata writes, even though field names
        # are case-insensitive.
        updated = conn.execute(_UPDATE_DESCRIPTION_SQL, (description, self._table.name, attribute))
        if updated.rowcount == 0:
            conn.execute(_INSERT_DESCRIPTION_SQL, (self._table.name, attribute, description))
