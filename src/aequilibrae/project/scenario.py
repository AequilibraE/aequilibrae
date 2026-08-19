import logging
from pathlib import Path

from aequilibrae.project.about import About
from aequilibrae.project.data import Matrices, Results
from aequilibrae.project.network import Network
from aequilibrae.project.zoning import Zoning
from aequilibrae.transit import Transit
from aequilibrae.utils.db_utils import ConnectionClosure, safe_connect
from aequilibrae.utils.spatialite_utils import connect_spatialite


class Scenario:
    """Own all persistent database resources for one project scenario."""

    def __init__(
        self,
        name: str,
        base_path: Path,
        connections: ConnectionClosure,
        log_handler: logging.Handler,
        project,
    ):
        self.name = name
        self.base_path = Path(base_path)
        self.path_to_file = self.base_path / "project_database.sqlite"
        self.log_handler = log_handler
        self.connections = connections
        self._destroyed = False
        self._results = None
        self._transit = None

        project_manager = self.connections["project"]
        self.about = About(project_manager)
        self.matrices = Matrices(project_manager, self.base_path / "matrices")
        self.network = Network(project, project_manager)
        self.zoning = Zoning(self.network)
        if "results" in self.connections:
            self._results = Results(project_manager, self.connections["results"])
        if "transit" in self.connections:
            self._transit = Transit(project, project_manager, self.connections["transit"], self.network.periods)

    @classmethod
    def create(cls, name: str, base_path, project) -> "Scenario":
        """Build a complete scenario from disk without creating any files.

        The project database must already exist. The optional results and
        transit databases are opened only when their files are present; they
        are never created as a side effect of opening a scenario.
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
        return cls(name, base_path, closure, logging.FileHandler(base_path / "aequilibrae.log"), project)

    @property
    def results(self):
        if self._results is None:
            raise RuntimeError("this scenario has no results database")
        return self._results

    @property
    def transit(self):
        if self._transit is None:
            raise RuntimeError("this scenario has no transit database")
        return self._transit

    def ensure_idle(self):
        self.connections.ensure_idle()

    def destroy(self):
        if self._destroyed:
            return
        self.connections.close()
        self.log_handler.close()
        self._destroyed = True
