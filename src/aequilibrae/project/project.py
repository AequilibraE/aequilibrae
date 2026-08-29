import functools
import logging
import os
import shutil
import sqlite3
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NoReturn

import pandas as pd

from aequilibrae.context import activate_project, get_active_project
from aequilibrae.log import Log
from aequilibrae.parameters import Parameters
from aequilibrae.project.about import About
from aequilibrae.project.data import Matrices, Results
from aequilibrae.project.network import Network
from aequilibrae.project.project_cleaning import clean
from aequilibrae.project.project_creation import initialize_tables
from aequilibrae.project.scenario import Scenario
from aequilibrae.project.tools import MigrationManager
from aequilibrae.project.zoning import Zoning
from aequilibrae.reference_files import demo_init_py, spatialite_database
from aequilibrae.transit import Transit
from aequilibrae.utils.db_utils import AequilibraEConnection, commit_and_close, safe_connect
from aequilibrae.utils.logging_utils import default_log_file_config
from aequilibrae.utils.model_run_utils import import_file_as_module
from aequilibrae.utils.spatialite_utils import connect_spatialite

logger = logging.getLogger(__name__)


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
        self.root_scenario: Scenario
        self.scenario: Scenario

    @classmethod
    def from_path(cls, project_folder: str):
        project = cls()
        project.open(project_folder)
        return project

    def open(self, project_path: os.PathLike | str) -> None:
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

        path_to_file = file_name

        self.root_scenario = Scenario(
            name="root",
            base_path=base_path,
            path_to_file=path_to_file,
            log_handler=logging.FileHandler(base_path / "aequilibrae.log"),
        )
        self.scenario = self.root_scenario

        # project_parameters is a cached_property - drop any value cached from a prior
        # new()/open() on this same Project instance before it's reloaded below, or it
        # would keep pointing at the previous project's parameters.yml.
        self.__dict__.pop("project_parameters", None)

        # Log outputs could interleave if two projects were open at once, but only one open project is supported.
        default_log_file_config(self.scenario.log_handler)

        self.activate()

        self.__load_objects()
        logger.info(f"Opened project on {self.project_base_path}")
        clean(self)

    @property
    def project_base_path(self) -> Path:
        return self.scenario.base_path

    @property
    def path_to_file(self) -> Path:
        return self.scenario.path_to_file

    @property
    def about(self) -> About:
        return self.scenario.about

    @property
    def network(self) -> Network:
        return self.scenario.network

    @property
    def transit(self) -> Transit:
        return self.scenario.transit

    @property
    def matrices(self) -> Matrices:
        return self.scenario.matrices

    @property
    def results(self) -> Results:
        return self.scenario.results

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
    def db_connection(self) -> Iterator[sqlite3.Connection | AequilibraEConnection]:
        with commit_and_close(self._project_database_path, spatial=True) as conn:
            yield conn

    @property
    def db_connection_spatial(self) -> Iterator[sqlite3.Connection | AequilibraEConnection]:
        """Deprecated alias for ``db_connection``, which is now a spatial connection."""
        warnings.warn(
            "'db_connection_spatial' is deprecated and will be removed in version 2.1. "
            "Use 'db_connection' instead, which is now a spatial connection.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.db_connection

    @property
    @contextmanager
    def results_connection(self) -> Iterator[sqlite3.Connection | AequilibraEConnection]:
        with commit_and_close(self._results_database_path, spatial=False, missing_ok=True) as conn:
            yield conn

    @property
    @contextmanager
    def transit_connection(self) -> Iterator[sqlite3.Connection | AequilibraEConnection]:
        with commit_and_close(self._transit_database_path, spatial=True, missing_ok=True) as conn:
            yield conn

    def new(self, project_path: os.PathLike | str) -> None:
        """Creates a new project

        :Arguments:
            **project_path** (:obj:`str`): Full path to the project data folder. If folder exists, it will fail
        """

        base_path = Path(project_path)
        path_to_file = base_path / "project_database.sqlite"

        if os.path.isdir(project_path):
            raise FileExistsError("Location already exists. Choose a different name or remove the existing directory")

        # We create the project folder and create the base file
        base_path.mkdir(parents=True, exist_ok=True)

        self.root_scenario = Scenario(
            name="root",
            base_path=base_path,
            path_to_file=path_to_file,
            log_handler=logging.FileHandler(base_path / "aequilibrae.log"),
        )
        self.scenario = self.root_scenario

        # See the matching comment in open(): drop any parameters cached from a prior
        # new()/open() on this same Project instance before it's reloaded below.
        self.__dict__.pop("project_parameters", None)

        default_log_file_config(self.scenario.log_handler)

        self.activate()

        self.__create_empty_network()
        self.__load_objects()
        self.about.create()
        logger.info(f"Created project on {base_path}")

    def close(self) -> None:
        """Safely closes the project"""
        if not self.project_base_path:
            logger.warning("This Aequilibrae project is not opened")
            return

        try:
            with self.db_connection as conn:
                conn.commit()
            clean(self)
            self.__dict__.pop("project_parameters", None)

            logger.info(f"Closed project on {self.project_base_path}")

            logging.getLogger("aequilibrae").removeHandler(self.scenario.log_handler)
            self.scenario.log_handler.close()
        except (sqlite3.ProgrammingError, AttributeError):
            logger.warning(f"This project at {self.project_base_path} is already closed")
            raise  # FIXME something goes wrong above

        finally:
            self.deactivate()

    def activate(self) -> None:
        activate_project(self)

    def deactivate(self) -> None:
        if get_active_project(must_exist=False) is self:
            activate_project(None)

    def log(self) -> Log:
        """Returns a log object

        allows the user to read the log or clear it"""
        return Log(self.project_base_path)

    def upgrade(self, ignore_project: bool = False, ignore_transit: bool = False, ignore_results: bool = False):
        """
        Find and apply all applicable migrations.

        All database upgrades are applied within a single transaction.

        Optionally ignore specific databases. This is useful when a database is known to be incompatible with some
        migrations but you'd still like to upgrade the others. Take care when ignoring a database. For a particular
        version of aequilibrae, it is assumed that all migrations have been applied successfully or the project was
        created with the latest schema, skipping/ignoring migrations will likely lead to issues/broken assumptions.

        If skipping a specific migration is required, use the ``aequilibrae.project.tools.MigrationManager`` object
        directly. Consult it's documentation page for details. Take care when skipping migrations.

        :Arguments:
            **ignore_project** (:obj:`bool`, optional): Ignore the project database. No direct migrations will be
                  applied. Defaults to False.
            **ignore_transit** (:obj:`bool`, optional): Ignore the transit database. No direct migrations will be
                  applied. Defaults to False.
            **ignore_results** (:obj:`bool`, optional): Ignore the results database. No direct migrations will be
                  applied. Defaults to False.
        """
        logger.info("Starting database upgrades")
        if ignore_project or ignore_transit or ignore_results:
            warnings.warn("Take care when ignoring a database during an upgrade.", stacklevel=2)

        connections: dict[str, sqlite3.Connection | AequilibraEConnection] = {}
        targets: list[tuple[MigrationManager, str]] = []

        if not ignore_project:
            targets.append((MigrationManager(MigrationManager.network_migration_file), "project_conn"))
            connections["project_conn"] = connect_spatialite(self._project_database_path)
        else:
            logger.warning("Ignoring project database during upgrade")

        if not ignore_transit and (self.project_base_path / "public_transport.sqlite").exists():
            targets.append((MigrationManager(MigrationManager.transit_migration_file), "transit_conn"))
            connections["transit_conn"] = connect_spatialite(self._transit_database_path)
        else:
            logger.warning("Ignoring transit database during upgrade")

        if not ignore_results and (self.project_base_path / "results_database.sqlite").exists():
            connections["results_conn"] = safe_connect(self._results_database_path)
        else:
            logger.warning("Ignoring results database during upgrade")

        try:
            for mm, main_conn in targets:
                if connections[main_conn] is not None:
                    with connections[main_conn] as conn:
                        mm.mark_all_as_seen(conn)

            for mm, main_conn in targets:
                mm.upgrade(main_conn, connections=connections)
            logger.info("Completed database upgrades")
        finally:
            for _, main_conn in targets:
                connections[main_conn].close()

    def __load_objects(self) -> None:
        matrix_folder = self.project_base_path / "matrices"
        matrix_folder.mkdir(parents=True, exist_ok=True)

        self.scenario.network = Network(self)
        self.scenario.about = About(self)
        self.scenario.matrices = Matrices(self)
        self.scenario.results = Results(self)
        self.scenario.transit = Transit(self)
        self.scenario.zoning = Zoning(self.scenario.network)

    @functools.cached_property
    def project_parameters(self) -> Parameters:
        return Parameters(path=self.project_base_path)

    @property
    def parameters(self) -> dict:
        return self.project_parameters.parameters

    @property
    def run(self) -> dict[str, functools.partial]:
        """
        Load and return the AequilibraE run module with the default arguments from
        ``parameters.yml`` partially applied.

        Refer to ``run/__init__.py`` file within the project folder for documentation.
        """
        entry_points = self.parameters["run"]
        module = import_file_as_module(
            self.root_scenario.base_path / "run" / "__init__.py", "aequilibrae.run", force=True
        )

        res: dict[str, functools.partial] = {}
        sentinal = object()
        for name, kwargs in entry_points.items():
            attr = getattr(module, name)
            if attr is sentinal:
                raise RuntimeError(f"expected to find callable '{name}' in the run module but didn't")
            elif not callable(attr):
                raise RuntimeError(f"found symbol '{name}' in the run module but it is not callable")

            func = functools.partial(attr, **(kwargs if kwargs is not None else {}))
            res[name] = func

        return res

    def check_file_indices(self) -> NoReturn:
        """Makes results_database.sqlite and the matrices folder compatible with project database"""
        raise NotImplementedError

    @property
    def zoning(self) -> Zoning:
        return self.scenario.zoning

    def __create_empty_network(self) -> None:
        shutil.copyfile(spatialite_database, self.path_to_file)
        pth = self.project_base_path / "run"
        pth.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(demo_init_py, pth / "__init__.py")

        # Write parameters to the project folder
        p = self.project_parameters
        p.parameters["system"]["logging_directory"] = str(self.project_base_path)
        p.write_back()

        # Create actual tables
        with self.db_connection as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            initialize_tables("network", conn=conn)

    def list_scenarios(self) -> pd.DataFrame:
        """
        Lists the existing scenarios.

        :Returns:
            **scenarios** (:obj:`pd.DataFrame`): Pandas DataFrame with existing scenarios
        """
        with self.db_connection as conn:
            return pd.read_sql("SELECT * FROM scenarios", conn)

    def use_scenario(self, scenario_name: str) -> None:
        """
        Switch the active scenario.

        :Arguments:
            **scenario_name** (:obj:`str`): name of the scenario to be activated

        """
        with commit_and_close(self.root_scenario.path_to_file, spatial=False) as conn:
            if conn.execute("SELECT 1 FROM scenarios where scenario_name=?", (scenario_name,)).fetchone() is None:
                raise ValueError(f"scenario '{scenario_name}' does not exist")

        logging.getLogger("aequilibrae").removeHandler(self.scenario.log_handler)

        if scenario_name == "root":
            self.scenario = self.root_scenario
        else:
            path = self.root_scenario.base_path / "scenarios" / scenario_name
            self.scenario = Scenario(
                name=scenario_name,
                base_path=path,
                path_to_file=path / "project_database.sqlite",
                log_handler=logging.FileHandler(path / "aequilibrae.log"),
            )

        self.__dict__.pop("project_parameters", None)
        default_log_file_config(self.scenario.log_handler)

        self.__load_objects()

    def create_empty_scenario(self, scenario_name: str, description: str = "") -> None:
        """
        Creates an empty scenario, without any links, nodes, and zones.

        :Arguments:
            **scenario_name** (:obj:`str`): scenario name

            **description** (:obj:`str`): useful scenario description
        """
        scenario_path = self.root_scenario.base_path / "scenarios" / scenario_name

        current_scenario = self.scenario.name
        self.use_scenario("root")
        try:
            with self.db_connection as conn:
                if (
                    conn.execute("SELECT 1 FROM scenarios where scenario_name=?", (scenario_name,)).fetchone()
                    is not None
                ):
                    raise ValueError("a scenario of that name already exists")

            scenario_path.mkdir(parents=True, exist_ok=True)

            db = scenario_path / "project_database.sqlite"
            shutil.copyfile(spatialite_database, db)

            # Write parameters to the project folder
            p = Parameters(path=scenario_path)
            p.parameters["system"]["logging_directory"] = str(scenario_path)
            p.write_back()

            # Create actual tables
            with commit_and_close(db, spatial=True) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                initialize_tables("network", conn=conn)
                conn.execute("DROP TABLE IF EXISTS scenarios")

            with self.db_connection as conn:
                conn.execute(
                    "INSERT INTO scenarios (scenario_name, description) VALUES(?,?)", (scenario_name, description)
                )
        finally:
            self.use_scenario(current_scenario)

    def clone_scenario(self, scenario_name: str, description: str = "") -> None:
        """
        Clones the active scenario.

        :Arguments:
            **scenario_name** (:obj:`str`): scenario name

            **description** (:obj:`str`): useful scenario description
        """
        scenario_path = self.root_scenario.base_path / "scenarios" / scenario_name

        current_scenario = self.scenario.name
        matrices_path = self.matrices.fldr
        project_db_path = self._project_database_path
        transit_db_path = self._transit_database_path
        results_db_path = self._results_database_path
        parameters_path = self.project_parameters.file

        self.use_scenario("root")
        try:
            with self.db_connection as conn:
                if (
                    conn.execute("SELECT 1 FROM scenarios where scenario_name=?", (scenario_name,)).fetchone()
                    is not None
                ):
                    raise ValueError("a scenario of that name already exists")

            shutil.copytree(matrices_path, scenario_path / "matrices")

            db = scenario_path / "project_database.sqlite"
            shutil.copyfile(project_db_path, db)

            try:
                shutil.copyfile(transit_db_path, scenario_path / "public_transport.sqlite")
            except FileNotFoundError:
                pass

            try:
                shutil.copyfile(results_db_path, scenario_path / "results_database.sqlite")
            except FileNotFoundError:
                pass

            if parameters_path is not None:
                shutil.copy(parameters_path, scenario_path)

            with commit_and_close(db, spatial=True) as conn:
                conn.execute("DROP TABLE IF EXISTS scenarios")

            with self.db_connection as conn:
                conn.execute(
                    "INSERT INTO scenarios (scenario_name, description) VALUES(?,?)", (scenario_name, description)
                )
        finally:
            self.use_scenario(current_scenario)
