import functools
import logging
import shutil
from collections import namedtuple
from pathlib import Path

import pandas as pd

from aequilibrae.log import Log
from aequilibrae.parameters import Parameters
from aequilibrae.project.project_cleaning import clean
from aequilibrae.project.project_creation import initialize_tables
from aequilibrae.project.scenario import Scenario
from aequilibrae.project.tools import MigrationManager
from aequilibrae.reference_files import demo_init_py, spatialite_database
from aequilibrae.utils.db_utils import ConnectionClosure, commit_and_close, safe_connect
from aequilibrae.utils.logging_utils import default_log_file_config
from aequilibrae.utils.model_run_utils import import_file_as_module
from aequilibrae.utils.spatialite_utils import connect_spatialite

logger = logging.getLogger(__name__)


class Project:
    """AequilibraE project class

    Projects are created and opened through class-method factories:

    .. code-block:: python
        :caption: Create a new project

        >>> new_project = Project.new(project_path)
        >>> new_project.close()

    .. code-block:: python
        :caption: Open an existing project

        >>> existing_project = Project.from_path(project_path)
        >>> existing_project.close()

    """

    def __init__(self, root_scenario: Scenario):
        """Wrap a fully constructed root scenario.

        Do not call directly; use :meth:`from_path` or :meth:`new`.
        """
        self.root_scenario: Scenario = root_scenario
        self.scenario: Scenario = root_scenario
        self._closed = False

    @classmethod
    def from_path(cls, project_folder) -> "Project":
        """Open an existing project from *project_folder*.

        :Arguments:
            **project_folder** (:obj:`str` or :obj:`Path`): Full path to the
            project data directory.  Must contain ``project_database.sqlite``.

        :Returns:
            **project** (:obj:`Project`): Open, fully initialised project.

        :Raises:
            **FileNotFoundError**: When the project database does not exist.
            **ValueError**: When the database is not an authoritative root scenario.
        """
        base_path = Path(project_folder)
        if not (base_path / "project_database.sqlite").is_file():
            raise FileNotFoundError("Model does not exist. Check your path and try again")

        scenario = Scenario.create("root", base_path)
        try:
            root = (
                scenario.connections["project"].execute("SELECT 1 FROM scenarios WHERE scenario_name='root'").fetchone()
            )
            if root is None:
                raise ValueError("project database is not an authoritative root scenario")
        except BaseException:
            scenario.destroy()
            raise

        project = cls(scenario)
        default_log_file_config(scenario.log_handler)
        logger.info(f"Opened project on {base_path}")
        clean(project)
        return project

    @classmethod
    def new(cls, project_path: str) -> "Project":
        """Create a new project at *project_path*.

        :Arguments:
            **project_path** (:obj:`str` or :obj:`Path`): Full path for the new
            project data directory.  The directory must not already exist.

        :Returns:
            **project** (:obj:`Project`): Open, fully initialised project.

        :Raises:
            **FileExistsError**: When *project_path* already exists.
        """
        base_path = Path(project_path)
        if base_path.exists():
            raise FileExistsError("Location already exists. Choose a different name or remove the existing directory")

        base_path.mkdir(parents=True, exist_ok=True)
        _create_project_files(base_path)

        scenario = None
        try:
            scenario = Scenario.create("root", base_path)
            initialize_tables(scenario.connections)
            scenario.about.create()
        except BaseException:
            if scenario is not None:
                scenario.destroy()
            shutil.rmtree(base_path, ignore_errors=True)
            raise

        project = cls(scenario)
        default_log_file_config(scenario.log_handler)
        logger.info(f"Created project on {base_path}")
        return project

    @property
    def project_base_path(self) -> Path:
        return self.scenario.base_path

    @property
    def path_to_file(self) -> Path:
        return self.scenario.path_to_file

    @property
    def about(self):
        return self.scenario.about

    @property
    def network(self):
        return self.scenario.network

    @property
    def transit(self):
        return self.scenario.transit

    @property
    def matrices(self):
        return self.scenario.matrices

    @property
    def results(self):
        return self.scenario.results

    @property
    def zoning(self):
        return self.scenario.zoning

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
    def db_connection(self):
        """Return a context manager yielding the project SQLite connection."""
        return self.scenario.connections["project"].transaction()

    def transaction(self):
        """Return a coordinated transaction context across all open connections."""
        return self.scenario.connections.transaction()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Close all owned resources.

        Safe to call more than once; subsequent calls are no-ops.  Accessing
        project attributes after shutdown raises errors from the closed SQLite
        connections.
        """
        if self._closed:
            return

        selected = self.scenario
        root = self.root_scenario
        selected.ensure_idle()
        if root is not selected:
            root.ensure_idle()
        clean(self)
        logging.getLogger("aequilibrae").removeHandler(selected.log_handler)
        if selected is not root:
            selected.destroy()
        root.destroy()
        self._closed = True

    #: Alias for :meth:`shutdown` — kept for backward compatibility.
    close = shutdown

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return False

    def log(self) -> Log:
        """Returns a log object that allows reading or clearing the log."""
        return Log(self.project_base_path)

    @staticmethod
    def upgrade(project_path):
        """Upgrade the databases of a *closed* project path.

        :Arguments:
            **project_path** (:obj:`str` or :obj:`Path`): Path to the project
            folder containing ``project_database.sqlite``.
        """
        base_path = Path(project_path).resolve()
        project_db = base_path / "project_database.sqlite"
        if not project_db.is_file():
            raise ValueError(f"project database does not exist: {project_db}")

        openers = {"project": lambda: connect_spatialite(project_db)}
        has_transit = False

        results_db = base_path / "results_database.sqlite"
        if results_db.is_file():
            openers["results"] = lambda: safe_connect(results_db)

        transit_db = base_path / "public_transport.sqlite"
        if transit_db.is_file():
            openers["transit"] = lambda: connect_spatialite(transit_db)
            has_transit = True

        closure = ConnectionClosure.open(openers)
        try:
            root = closure["project"].execute("SELECT 1 FROM scenarios WHERE scenario_name='root'").fetchone()
            if root is None:
                raise ValueError("upgrade path is not an authoritative root project")
            network = MigrationManager(MigrationManager.network_migration_file)
            network.upgrade(closure)
            if has_transit:
                transit_mm = MigrationManager(MigrationManager.transit_migration_file)
                transit_mm.upgrade(closure)
        finally:
            closure.close()

    @property
    def project_parameters(self) -> Parameters:
        return self.scenario.project_parameters

    @property
    def parameters(self) -> dict:
        return self.scenario.parameters

    @property
    def run(self):
        """Load and return the AequilibraE run module with default arguments from ``parameters.yml``."""
        entry_points = self.parameters["run"]
        module = import_file_as_module(
            self.root_scenario.base_path / "run" / "__init__.py", "aequilibrae.run", force=True
        )

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

    def list_scenarios(self) -> pd.DataFrame:
        """List existing scenarios.

        :Returns:
            **scenarios** (:obj:`pd.DataFrame`): DataFrame with existing scenarios.
        """
        return pd.read_sql("SELECT * FROM scenarios", self.root_scenario.connections["project"])

    def use_scenario(self, scenario_name: str):
        """Switch the active scenario.

        :Arguments:
            **scenario_name** (:obj:`str`): Name of the scenario to activate.
        """
        current = self.scenario
        current.ensure_idle()
        self.root_scenario.ensure_idle()
        root_manager = self.root_scenario.connections["project"]
        if root_manager.execute("SELECT 1 FROM scenarios WHERE scenario_name=?", (scenario_name,)).fetchone() is None:
            raise ValueError(f"scenario '{scenario_name}' does not exist")
        if scenario_name == current.name:
            return

        if scenario_name == "root":
            candidate = self.root_scenario
        else:
            candidate = Scenario.create(scenario_name, self.root_scenario.base_path / "scenarios" / scenario_name)
        self.scenario = candidate

        aequilibrae_logger = logging.getLogger("aequilibrae")
        aequilibrae_logger.removeHandler(current.log_handler)
        default_log_file_config(candidate.log_handler)
        if current is not self.root_scenario:
            current.destroy()

    def create_empty_scenario(self, scenario_name: str, description: str = ""):
        """Create an empty scenario with no links, nodes, or zones.

        :Arguments:
            **scenario_name** (:obj:`str`): Scenario name.

            **description** (:obj:`str`, *Optional*): Human-readable description.
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
                initialize_tables(ConnectionClosure({"project": conn}), databases=("network",))
                conn.execute("DROP TABLE IF EXISTS scenarios")

            with self.db_connection as conn:
                conn.execute(
                    "INSERT INTO scenarios (scenario_name, description) VALUES(?,?)",
                    (scenario_name, description),
                )
        finally:
            self.use_scenario(current_scenario)

    def clone_scenario(self, scenario_name: str, description: str = ""):
        """Clone the active scenario.

        :Arguments:
            **scenario_name** (:obj:`str`): Scenario name.

            **description** (:obj:`str`, *Optional*): Human-readable description.
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

            shutil.copy(parameters_path, scenario_path)

            with commit_and_close(db, spatial=True) as conn:
                conn.execute("DROP TABLE IF EXISTS scenarios")

            with self.db_connection as conn:
                conn.execute(
                    "INSERT INTO scenarios (scenario_name, description) VALUES(?,?)",
                    (scenario_name, description),
                )
        finally:
            self.use_scenario(current_scenario)


def _create_project_files(base_path: Path) -> None:
    """Create the initial filesystem layout for a new project.

    This is the only code path permitted to create ``project_database.sqlite``,
    ``public_transport.sqlite``, and ``results_database.sqlite``; ordinary
    ``open``/``from_path`` operations never create these files.
    """
    shutil.copyfile(spatialite_database, base_path / "project_database.sqlite")
    shutil.copyfile(spatialite_database, base_path / "public_transport.sqlite")
    (base_path / "results_database.sqlite").touch()
    (base_path / "matrices").mkdir(parents=True, exist_ok=True)
    pth = base_path / "run"
    pth.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(demo_init_py, pth / "__init__.py")

    parameters = Parameters(path=base_path)
    parameters.parameters["system"]["logging_directory"] = str(base_path)
    parameters.write_back()
