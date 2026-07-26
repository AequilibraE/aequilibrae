import logging

from aequilibrae.project.project_table import ProjectTable

logger = logging.getLogger(__name__)


class Links(ProjectTable):
    """
    Access to the API resources to manipulate the links table in the network

    .. code-block:: python

        >>> project = create_example(project_path)

        >>> links = project.network.links

        # We can get a single link as an immutable record
        >>> link = links.get(1)
        >>> link.modes
        'cMT'

        # and write any changes explicitly
        >>> links.update(1, lanes_ab=2, name="First Avenue")

        # Manipulating modes has its own helpers
        >>> links.add_mode(1, 'b')
        >>> links.drop_mode(1, 'b')

        # Many changes go in one batch (a single transaction)
        >>> with links.batch() as batch:
        ...     for link_id in (2, 3, 4):
        ...         batch.update(link_id, speed_ab=90.0)

        >>> project.close()
    """

    name = "links"
    key = "link_id"
    record_name = "LinkRecord"
    spatial = True
    protected = frozenset(("a_node", "b_node"))
    defaults = {"a_node": 0, "b_node": 0, "direction": 0, "link_type": "default"}

    def __init__(self, net):
        super().__init__(net.project)

    def copy(self, link_id: int, conn=None) -> int:
        """Duplicates a link under a new id, returning that id

        It raises an error if ``link_id`` does not exist
        """
        link = self.get(link_id, conn=conn)
        values = {k: v for k, v in vars(link).items() if k != self.key and k not in self.protected}
        return self.insert(conn=conn, **values)

    def add_mode(self, link_id: int, mode, conn=None):
        """Adds a mode to a link

        Logs a warning if the mode is already allowed on the link

        :Arguments:
            **link_id** (:obj:`int`): Id of the link to change

            **mode** (:obj:`str` or mode record): mode_id or mode to be added to the link
        """
        mode_id = self.__mode_id_of(mode)
        modes = self.get(link_id, conn=conn).modes
        if mode_id in modes:
            logger.warning("Mode already active for this link")
            return
        self.update(link_id, conn=conn, modes=modes + mode_id)

    def drop_mode(self, link_id: int, mode, conn=None):
        """Removes a mode from a link

        Logs a warning if the mode is already NOT allowed on the link

        :Arguments:
            **link_id** (:obj:`int`): Id of the link to change

            **mode** (:obj:`str` or mode record): mode_id or mode to be removed from the link
        """
        mode_id = self.__mode_id_of(mode)
        modes = self.get(link_id, conn=conn).modes
        if mode_id not in modes:
            logger.warning("Mode already inactive for this link")
            return
        self.update(link_id, conn=conn, modes=modes.replace(mode_id, ""))

    def _check_modes(self, modes) -> str:
        if not isinstance(modes, str):
            raise ValueError("Modes field needs to be a string")
        if modes == "":
            raise ValueError("Modes field needs to have at least one mode")
        return modes

    @staticmethod
    def __mode_id_of(mode) -> str:
        mode_id = getattr(mode, "mode_id", mode)
        if not isinstance(mode_id, str):
            raise TypeError("You should provide a mode_id (string) or a mode record")
        if len(mode_id) != 1:
            raise ValueError("A mode_id is a single character")
        return mode_id
