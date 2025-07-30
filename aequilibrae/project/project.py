import functools
import logging
import os
import shutil
import sqlite3
from collections import namedtuple
from contextlib import contextmanager
from pathlib import Path

from aequilibrae import global_logger
from aequilibrae.context import activate_project, get_active_project
from aequilibrae.log import Log
from aequilibrae.log import get_log_handler
from aequilibrae.parameters import Parameters
from aequilibrae.project.about import About
from aequilibrae.project.data import Matrices, Results
from aequilibrae.project.network import Network
from aequilibrae.project.project_cleaning import clean
from aequilibrae.project.project_creation import initialize_tables
from aequilibrae.project.tools import MigrationManager
from aequilibrae.project.zoning import Zoning
from aequilibrae.reference_files import spatialite_database, demo_init_py
from aequilibrae.transit import Transit
from aequilibrae.utils.db_utils import commit_and_close, safe_connect
from aequilibrae.utils.model_run_utils import import_file_as_module
from aequilibrae.utils.spatialite_utils import connect_spatialite, load_spatialite_extension


class Project:
    """AequilibraE project class

    .. code-block:: python
        :caption: Create Project

        >>> new_project = Project()
        >>> new_project.new(project_path)

        # Safely closes the project
        >>> new_project.close()

    .. code-block:: python
        :caption: Open Project

        >>> existing_project = Project()
        >>> existing_project.open(project_path)

        >>> existing_project.close()
    """

    def __init__(self):
        pass

    @classmethod
    def from_path(cls, project_folder):
        project = cls()
        project.open(project_folder)
        return project

    def open(self, project_path: str) -> None:
        """
        Loads project from disk

        :Arguments:
            **project_path** (:obj:`str`): Full path to the project data folder. If the project inside does
            not exist, it will fail.
        """

        base_path = Path(project_path)
        file_name = base_path / "project_database.sqlite"

        if not file_name.is_file() or not file_name.exists():
            raise FileNotFoundError("Model does not exist. Check your path and try again")

        self.__base_path = base_path
        self.__logger = self.__setup_logger()

        self.activate()

        self.__load_objects()
        global_logger.info(f"Opened project on {self.project_base_path}")
        clean(self)

    @property
    def project_base_path(self):
        return self.__base_path

    @property
    def path_to_file(self):
        return self._project_database_path

    @property
    def about(self) -> About:
        return self.__about

    @property
    def logger(self) -> logging.Logger:
        return self.__logger

    @property
    def network(self) -> Network:
        return self.__network

    @property
    def transit(self) -> Transit:
        return self.__transit

    @property
    def matrices(self) -> Matrices:
        return self.__matrices

    @property
    def results(self) -> Results:
        return self.__results

    @property
    def _project_database_path(self) -> Path:
        return self.project_base_path / "project_database.sqlite"

    @property
    def _results_database_path(self) -> Path:
        return self.project_base_path / "results_database.sqlite"

    @property
    def _transit_database_path(self) -> Path:
        return self.project_base_path / "public_transport.sqlite"

    @property
    @contextmanager
    def db_connection(self):
        with commit_and_close(self._project_database_path, spatial=False) as conn:
            yield conn

    @property
    @contextmanager
    def db_connection_spatial(self):
        with commit_and_close(self._project_database_path, spatial=True) as conn:
            yield conn

    @property
    @contextmanager
    def results_connection(self):
        with commit_and_close(self._results_database_path, spatial=False, missing_ok=True) as conn:
            yield conn

    @property
    @contextmanager
    def transit_connection(self):
        with commit_and_close(self._transit_database_path, spatial=True, missing_ok=True) as conn:
            yield conn

    def new(self, project_path: str) -> None:
        """Creates a new project

        :Arguments:
            **project_path** (:obj:`str`): Full path to the project data folder. If folder exists, it will fail
        """

        base_path = Path(project_path)

        if os.path.isdir(project_path):
            raise FileExistsError("Location already exists. Choose a different name or remove the existing directory")

        # We create the project folder and create the base file
        base_path.mkdir(parents=True, exist_ok=True)

        self.__base_path = base_path
        self.__logger = self.__setup_logger()
        self.activate()

        self.__create_empty_network()
        self.__load_objects()
        self.about.create()
        global_logger.info(f"Created project on {base_path}")

    def close(self) -> None:
        """Safely closes the project"""
        if not self.project_base_path:
            global_logger.warning("This Aequilibrae project is not opened")
            return

        try:
            with self.project.db_connection as conn:
                conn.commit()
            clean(self)
            for obj in [self.parameters, self.network]:
                del obj

            del self.network.link_types
            del self.network.modes

            global_logger.info(f"Closed project on {self.project_base_path}")

        except (sqlite3.ProgrammingError, AttributeError):
            global_logger.warning(f"This project at {self.project_base_path} is already closed")

        finally:
            handlers = global_logger.handlers[:]  # Make a copy of the handlers list
            for handler in handlers:
                handler.close()  # Explicitly close each handler to release file handles
                global_logger.removeHandler(handler)  # Remove the handler from the logger

            self.deactivate()

    def activate(self):
        activate_project(self)

    def deactivate(self):
        if get_active_project(must_exist=False) is self:
            activate_project(None)

    def log(self) -> Log:
        """Returns a log object

        allows the user to read the log or clear it"""
        return Log(self.project_base_path)

    def upgrade(self):
        """
        Find and apply all applicable migrations.

        All database upgrades are applied within a single transaction.

        If skipping a specific migration is required, use the ``aequilbrae.project.tools.MigrationManager`` object
        directly. Consult it's documentation page for details. Take care when skipping migrations.
        """
        global_logger.info("Starting database upgrades")
        connections = {
            "project_conn": connect_spatialite(self._project_database_path),
            "transit_conn": None,
            "results_conn": None,
        }
        targets = [
            (MigrationManager(MigrationManager.network_migration_file), "project_conn"),
        ]

        if (self.project_base_path / "public_transport.sqlite").exists():
            targets.append((MigrationManager(MigrationManager.transit_migration_file), "transit_conn"))
            connections["transit_conn"] = connect_spatialite(self._transit_database_path)

        if (self.project_base_path / "results_database.sqlite").exists():
            connections["results_conn"] = safe_connect(self._results_database_path)

        try:
            for mm, main_conn in targets:
                with connections[main_conn] as conn:
                    mm.mark_all_as_seen(conn)

            for mm, main_conn in targets:
                mm.upgrade(main_conn, connections=connections)
            global_logger.info("Completed database upgrades")
        finally:
            for _, main_conn in targets:
                connections[main_conn].close()

    def __load_objects(self):
        matrix_folder = self.project_base_path / "matrices"
        matrix_folder.mkdir(parents=True, exist_ok=True)

        self.__network = Network(self)
        self.__about = About(self)
        self.__matrices = Matrices(self)
        self.__results = Results(self)

    @property
    def project_parameters(self) -> Parameters:
        return Parameters(path=self.project_base_path)

    @property
    def parameters(self) -> dict:
        return self.project_parameters.parameters

    @property
    def run(self):
        """
        Load and return the AequilibraE run module with the default arguments from
        ``parameters.yml`` partially applied.

        Refer to ``run/__init__.py`` file within the project folder for documentation.
        """
        entry_points = self.parameters["run"]
        module = import_file_as_module(self.project_base_path / "run" / "__init__.py", "aequilibrae.run", force=True)

        res = []
        sentinal = object()
        for name, kwargs in entry_points.items():
            attr = getattr(module, name)
            if attr is sentinal:
                raise RuntimeError(f"expected to find callable '{name}' in the run module but didn't")
            elif not callable(attr):
                raise RuntimeError(f"found symbol '{name}' in the run module but it is not callable")

            func = functools.partial(attr, **(kwargs if kwargs is not None else {}))
            res.append((name, func))

        Run = namedtuple("Run", [k for k, _ in res])
        return Run._make([v for _, v in res])

    def check_file_indices(self) -> None:
        """Makes results_database.sqlite and the matrices folder compatible with project database"""
        raise NotImplementedError

    @property
    def zoning(self):
        return Zoning(self.network)

    def __create_empty_network(self):
        shutil.copyfile(spatialite_database, self.path_to_file)
        pth = self.project_base_path / "run"
        pth.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(demo_init_py, pth / "__init__.py")

        # Write parameters to the project folder
        p = self.project_parameters
        p.parameters["system"]["logging_directory"] = str(self.project_base_path)
        p.write_back()

        # Create actual tables
        with self.db_connection_spatial as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            initialize_tables(self.logger, "network", conn=conn)

    def __setup_logger(self):
        logger = logging.getLogger(f"aequilibrae.{self.project_base_path}")
        logger.propagate = False
        logger.setLevel(logging.DEBUG)

        par = self.parameters or self.project_parameters._default
        do_log = par["system"]["logging"]

        if do_log:
            logger.addHandler(get_log_handler(self.project_base_path / "aequilibrae.log"))

        return logger
