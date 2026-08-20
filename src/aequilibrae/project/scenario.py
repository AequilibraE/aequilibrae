import logging
from pathlib import Path

from aequilibrae.parameters import Parameters
from aequilibrae.project.about import About
from aequilibrae.project.data import Matrices, Results
from aequilibrae.project.network import Network
from aequilibrae.project.zoning import Zoning
from aequilibrae.transit import Transit
from aequilibrae.utils.db_utils import ConnectionClosure, safe_connect
from aequilibrae.utils.spatialite_utils import connect_spatialite


class Scenario:
    """Own all persistent database resources for one project scenario.

    A ``Scenario`` is the single source of truth for a named slice of project
    state.  It owns its paths, log handler, parameters, and all gateway
    objects (network, zoning, matrices, results, transit, about).  The
    optional results and transit databases are opened only when their files
    already exist on disk; they are never created as a side-effect of opening
    a scenario.

    Construction is one-stage: every gateway is fully wired in ``__init__``.
    Use :meth:`create` to build a scenario from an on-disk project folder.
    """

    def __init__(
        self,
        name: str,
        base_path: Path,
        connections: ConnectionClosure,
        log_handler: logging.Handler,
    ):
        self.name = name
        self.base_path = Path(base_path)
        self.path_to_file = self.base_path / "project_database.sqlite"
        self.log_handler = log_handler
        self.connections = connections
        self.project_parameters = Parameters(path=self.base_path)
        self._destroyed = False
        self._results = None
        self._transit = None

        project_manager = self.connections["project"]
        self.about = About(project_manager)
        self.matrices = Matrices(project_manager, self.base_path / "matrices")
        # Network and Transit receive *this* scenario as their owner reference so
        # that helpers (OSMBuilder, TransitGraphBuilder, …) can reach path/connection
        # attributes without creating a circular dependency with Project.
        self.network = Network(self, project_manager)
        self.zoning = Zoning(self.network)
        if "results" in self.connections:
            self._results = Results(project_manager, self.connections["results"])
        if "transit" in self.connections:
            self._transit = Transit(self, project_manager, self.connections["transit"], self.network.periods)

    @property
    def project_base_path(self) -> Path:
        """Alias for :attr:`base_path`; satisfies the network-gateway interface."""
        return self.base_path

    @property
    def parameters(self) -> Parameters:
        """The parameters loaded from this scenario's ``parameters.yml`` file."""
        return self.project_parameters

    @property
    def db_connection(self):
        """Return the project-database :class:`~aequilibrae.utils.db_utils.NestedTransactions` manager."""
        return self.connections["project"]

    @property
    def transit_connection(self):
        """Return a transaction context for the transit database."""
        if "transit" not in self.connections:
            raise RuntimeError("this scenario has no transit database")
        return self.connections["transit"].transaction()

    def transaction(self):
        """Return a coordinated transaction context across all open connections.

        .. code-block:: python

            with scenario.transaction() as conns:
                conns["project"].execute(...)
        """
        return self.connections.transaction()

    @classmethod
    def create(cls, name: str, base_path) -> "Scenario":
        """Build a complete scenario from disk without creating any files.

        The project database must already exist.  The optional results and
        transit databases are opened only when their files are present on
        disk; they are **never** created as a side effect of opening a
        scenario.

        :Arguments:
            **name** (:obj:`str`): Scenario name (e.g. ``"root"``).

            **base_path** (:obj:`Path` or :obj:`str`): Directory that contains
            ``project_database.sqlite``.

        :Returns:
            **scenario** (:obj:`Scenario`): Fully initialised scenario.

        :Raises:
            **FileNotFoundError**: When ``project_database.sqlite`` does not exist.
        """
        base_path = Path(base_path)
        project_path = base_path / "project_database.sqlite"
        if not project_path.is_file():
            raise FileNotFoundError(f"Project database does not exist: {project_path}")

        openers = {"project": lambda: connect_spatialite(project_path)}

        results_path = base_path / "results_database.sqlite"
        if results_path.is_file():
            openers["results"] = lambda: safe_connect(results_path)

        transit_path = base_path / "public_transport.sqlite"
        if transit_path.is_file():
            openers["transit"] = lambda: connect_spatialite(transit_path)

        closure = ConnectionClosure.open(openers)
        log_handler = None
        try:
            log_handler = logging.FileHandler(base_path / "aequilibrae.log")
            return cls(name, base_path, closure, log_handler)
        except BaseException:
            closure.close()
            if log_handler is not None:
                log_handler.close()
            raise

    @property
    def results(self):
        """The results gateway.

        :Raises:
            **RuntimeError**: When this scenario has no results database.
        """
        if self._results is None:
            raise RuntimeError("this scenario has no results database")
        return self._results

    @property
    def transit(self):
        """The transit gateway.

        :Raises:
            **RuntimeError**: When this scenario has no transit database.
        """
        if self._transit is None:
            raise RuntimeError("this scenario has no transit database")
        return self._transit

    def ensure_idle(self):
        """Assert that no transaction is active on any owned connection."""
        self.connections.ensure_idle()

    def destroy(self):
        """Close all connections and the log handler.  Safe to call more than once."""
        if self._destroyed:
            return
        self.connections.close()
        self.log_handler.close()
        self._destroyed = True
