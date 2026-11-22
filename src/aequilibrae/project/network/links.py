import logging
from typing import Any, Mapping

import pandas as pd

from aequilibrae.project.network.links_view import LINK_ATTRIBUTES_TABLE, LINKS_VIEW, links_source, refresh_links_view
from aequilibrae.project.project_table import SpatialProjectTable

logger = logging.getLogger(__name__)


class Links(SpatialProjectTable):
    """
    Object to manipulate the links in the database.

    Links are stored in two tables: ``links`` (geometry and mandatory fields) and
    ``link_attributes`` (all other fields, including user-defined ones). This object reads
    and writes both through the editable ``links_full`` view, and adds new fields to
    ``link_attributes``.

    .. code-block:: python

        >>> project = create_example(project_path)

        >>> links = project.tables.links

        # We can get a single link as an immutable record
        >>> link = links.get(1)
        >>> link.modes
        'cMT'

        # and write any changes
        >>> links.update(1, lanes_ab=2, name="First Avenue")

        # Manipulating modes has its own helpers
        >>> links.add_mode(1, 'b')
        >>> links.drop_mode(1, 'b')

        # Coordinate many changes in one project transaction
        >>> with project.transaction():
        ...     for link_id in (2, 3, 4):
        ...         links.update(link_id, speed_ab=90.0)

        >>> project.close()
    """

    name = "links"
    key = "link_id"
    record_name = "LinkRecord"
    defaults = {"a_node": -1, "b_node": -1, "direction": 0, "link_type": "default"}
    _copy_excluded_fields = frozenset(("a_node", "b_node"))
    has_numeric_key = True

    def _resolve_source(self) -> str:
        with self._connection as conn:
            return links_source(conn)

    def _fields_table(self) -> str:
        return LINK_ATTRIBUTES_TABLE if self._source == LINKS_VIEW else self._source

    def _after_fields_added(self, conn) -> None:
        if self._source == LINKS_VIEW:
            refresh_links_view(conn)

    def _prepare_insert(self, values: Mapping[str, Any]) -> dict[str, Any]:
        if not (("a_node" in values and "b_node" in values) or "geometry" in values):
            raise ValueError("for links either a_node and b_node, or geometry must be provided")
        return super()._prepare_insert(values=values)

    def _prepare_rows(self, frame: pd.DataFrame, value_columns: tuple[str, ...]) -> list[tuple[Any, ...]]:
        if not (("a_node" in value_columns and "b_node" in value_columns) or "geometry" in value_columns):
            raise ValueError("for links either a_node and b_node, or geometry must be provided")
        return super()._prepare_rows(frame=frame, value_columns=value_columns)

    def copy(self, link_id: int) -> int:
        """Duplicate a link under a new ID and return that ID.

        :Arguments:
            **link_id** (:obj:`int`): ID of the link to duplicate.

        :Returns:
            **link ID** (:obj:`int`): Generated ID of the duplicate.
        """
        link = self.get(link_id)
        values = {k: v for k, v in vars(link).items() if k != self.key and k not in self._copy_excluded_fields}
        return self.insert(**values)

    def add_mode(self, link_id: int, mode: Any) -> None:
        """Adds a mode to a link

        Logs a warning if the mode is already allowed on the link

        :Arguments:
            **link_id** (:obj:`int`): Id of the link to change

            **mode** (:obj:`str` or mode record): mode_id or mode to be added to the link
        """
        mode_id = self.__mode_id_of(mode)
        modes = self.get(link_id).modes
        if mode_id in modes:
            logger.warning("Mode already active for this link")
            return
        self.update(link_id, modes=modes + mode_id)

    def drop_mode(self, link_id: int, mode: Any) -> None:
        """Removes a mode from a link

        Logs a warning if the mode is already NOT allowed on the link

        :Arguments:
            **link_id** (:obj:`int`): Id of the link to change

            **mode** (:obj:`str` or mode record): mode_id or mode to be removed from the link
        """
        mode_id = self.__mode_id_of(mode)
        modes = self.get(link_id).modes
        if mode_id not in modes:
            logger.warning("Mode already inactive for this link")
            return
        self.update(link_id, modes=modes.replace(mode_id, ""))

    @staticmethod
    def __mode_id_of(mode: Any) -> str:
        mode_id = getattr(mode, "mode_id", mode)
        if not isinstance(mode_id, str):
            raise TypeError("You should provide a mode_id (string) or a mode record")
        if len(mode_id) != 1:
            raise ValueError("A mode_id is a single character")
        return mode_id
