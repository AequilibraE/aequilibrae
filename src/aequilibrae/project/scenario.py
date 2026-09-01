import logging
import pathlib
from contextlib import AbstractContextManager

from scipy.special import fresnel

from aequilibrae.project.about import About
from aequilibrae.project.data import Matrices, Results
from aequilibrae.project.network import Network
from aequilibrae.transit.transit import Transit
from aequilibrae.utils.db_utils import ConnectionClosure, NestedTransactionManager


class Scenario:
    """
    Represents a modelling scenario within an AequilibraE project.

    Each scenario operates independently with its own database and file
    structure while sharing the overall project configuration.

    Scenarios are typically managed through the Project class rather than
    instantiated directly by users.

    The root scenario is special-cased and represents the original project
    configuration. All other scenarios are stored in subdirectories and
    reference their own database files.
    """

    name: str
    base_path: pathlib.Path
    path_to_file: pathlib.Path
    log_handler: logging.StreamHandler

    def __init__(
        self, name: str, base_path: pathlib.Path, path_to_file: pathlib.Path, log_handler: logging.StreamHandler
    ):
        self.name = name
        self.base_path = base_path
        self.path_to_file = path_to_file
        self.log_handler = log_handler

        results_path = self.base_path / "results_database.sqlite"
        results_path = results_path if results_path.is_file() else None

        transit_path = self.base_path / "public_transport.sqlite"
        transit_path = transit_path if transit_path.is_file() else None

        self.connections = ConnectionClosure(self.path_to_file, results_path=results_path, transit_path=transit_path)

        (self.base_path / "matrices").mkdir(parents=True, exist_ok=True)
        self.about = About(self)
        self.network = Network(self)
        self.matrices = Matrices(self.connections.db_connection, self.base_path / "matrices")

        if results_path is not None:
            self._results = Results(self.connections.db_connection, self.connections.results_connection)
        else:
            self._results = None

        if transit_path is not None:
            self._transit = Transit(self.connections.transit_connection)
        else:
            self._transit = None

    @property
    def project_base_path(self) -> pathlib.Path:
        """Alias used by project-owned table consumers."""
        return self.base_path

    @property
    def _results_database_path(self) -> pathlib.Path:
        return self.base_path / "results_database.sqlite"

    @property
    def _transit_database_path(self) -> pathlib.Path:
        return self.base_path / "public_transport.sqlite"

    @property
    def db_connection(self) -> NestedTransactionManager:
        return self.connections.db_connection

    @property
    def transit_connection(self) -> NestedTransactionManager:
        """The transit connection manager.

        Raises:
            RuntimeError: If no transit database has been created for this scenario.
        """
        return self.connections.transit_connection

    @property
    def results_connection(self) -> NestedTransactionManager:
        """The results connection manager, creating its database on first use."""
        try:
            return self.connections.results_connection
        except RuntimeError:
            return self.connections.create_results_connection(self._results_database_path)

    def transaction(self) -> AbstractContextManager[None]:
        return self.connections.transaction()

    def create_transit_database(self) -> Transit:
        """Create and return the transit gateway when explicitly requested."""
        if self._transit is None:
            self.connections.create_transit_connection(self._transit_database_path)
            self._transit = Transit(self)
        return self._transit

    @property
    def results(self) -> Results:
        """The results table gateway, creating its database on first access."""
        if self._results is None:
            self._results = Results(self.connections.db_connection, self.results_connection)
        return self._results

    @property
    def transit(self) -> Transit:
        """The transit table gateway.

        Raises:
            RuntimeError: If no transit database has been created for this scenario.
        """
        if self._transit is None:
            raise RuntimeError("this scenario has no transit database; call create_transit_database() first")
        return self._transit

    def close(self) -> None:
        """Close all database connections owned by this scenario."""
        self.connections.close()
        self.log_handler.close()
