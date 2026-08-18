import functools
import logging
import os
import shutil
from collections import namedtuple
from pathlib import Path

import pandas as pd

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
from aequilibrae.utils.db_utils import ConnectionClosure, commit_and_close, safe_connect
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
        self.root_scenario: Scenario = None
        self.scenario: Scenario = None

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

        candidate, created = Scenario.open_candidate("root", base_path)
        try:
            if candidate.connections["project"].execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='scenarios'"
            ).fetchone() is None or candidate.connections["project"].execute(
                "SELECT 1 FROM scenarios WHERE scenario_name='root'"
            ).fetchone() is None:
                raise ValueError("project database is not an authoritative root scenario")
            if base_path / "public_transport.sqlite" in created:
                initialize_tables(candidate.connections, databases=("transit",))
            self.root_scenario = candidate
            self.scenario = candidate
            self.__load_objects()
        except BaseException:
            candidate.destroy()
            for path in created:
                path.unlink(missing_ok=True)
            raise

        default_log_file_config(self.scenario.log_handler)
        logger.info(f"Opened project on {self.project_base_path}")
        clean(self)

    @property
    def project_base_path(self):
        return self._require_scenario().base_path

    @property
    def path_to_file(self):
        return self._require_scenario().path_to_file

    @property
    def about(self) -> About:
        return self._require_scenario().about

    @property
    def network(self) -> Network:
        return self._require_scenario().network

    @property
    def transit(self) -> Transit:
        return self._require_scenario().transit

    @property
    def matrices(self) -> Matrices:
        return self._require_scenario().matrices

    @property
    def results(self) -> Results:
        return self._require_scenario().results

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
        return self._require_scenario().connections["project"]

    def transaction(self):
        return self._require_scenario().connections.transaction()

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

        self.__create_empty_network(base_path)
        candidate, _ = Scenario.open_candidate("root", base_path, create_auxiliary=False)
        try:
            self.root_scenario = candidate
            self.scenario = candidate
            initialize_tables(candidate.connections)
            self.__load_objects()
            self.about.create()
        except BaseException:
            candidate.destroy()
            self.root_scenario = None
            self.scenario = None
            shutil.rmtree(base_path, ignore_errors=True)
            raise

        default_log_file_config(self.scenario.log_handler)
        logger.info(f"Created project on {base_path}")

    def shutdown(self) -> None:
        """Destroy scenario resources and make repeated shutdown harmless."""
        if self.scenario is None:
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
        self.scenario = None
        self.root_scenario = None

    def __enter__(self):
        self._require_scenario()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return False

    def log(self) -> Log:
        """Returns a log object

        allows the user to read the log or clear it"""
        return Log(self.project_base_path)

    @staticmethod
    def upgrade(project_path):
        """Upgrade all three databases of a closed project path."""
        base_path = Path(project_path).resolve()
        paths = {
            "project": base_path / "project_database.sqlite",
            "results": base_path / "results_database.sqlite",
            "transit": base_path / "public_transport.sqlite",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise ValueError(f"upgrade requires all three scenario databases; missing: {', '.join(missing)}")
        raw = []
        try:
            project = connect_spatialite(paths["project"])
            raw.append(project)
            results = safe_connect(paths["results"])
            raw.append(results)
            transit = connect_spatialite(paths["transit"])
            raw.append(transit)
            closure = ConnectionClosure({"project": project, "results": results, "transit": transit})
            raw.clear()
            try:
                root = closure["project"].execute(
                    "SELECT 1 FROM scenarios WHERE scenario_name='root'"
                ).fetchone()
                if root is None:
                    raise ValueError("upgrade path is not an authoritative root project")
                managers = [
                    MigrationManager(MigrationManager.network_migration_file, database="project"),
                    MigrationManager(MigrationManager.transit_migration_file, database="transit"),
                ]
                with closure.transaction():
                    for manager in managers:
                        manager._initialize_status(closure)
                        for migration in manager._find_applicable(closure[manager.database]):
                            migration.apply(closure)
            finally:
                closure.close()
        finally:
            for connection in raw:
                connection.close()

    def __load_objects(self):
        matrix_folder = self.project_base_path / "matrices"
        matrix_folder.mkdir(parents=True, exist_ok=True)

        project_manager = self.scenario.connections["project"]
        self.scenario.network = Network(self)
        self.scenario.about = About(project_manager)
        self.scenario.matrices = Matrices(project_manager, matrix_folder)
        self.scenario.results = Results(project_manager, self.scenario.connections["results"])
        self.scenario.transit = Transit(self)

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

    @property
    def zoning(self):
        if self.scenario.zoning is None:
            self.scenario.zoning = Zoning(self.network)
        return self.scenario.zoning

    def _require_scenario(self):
        if self.scenario is None:
            raise RuntimeError("Project is not open")
        return self.scenario

    def __create_empty_network(self, base_path):
        shutil.copyfile(spatialite_database, base_path / "project_database.sqlite")
        shutil.copyfile(spatialite_database, base_path / "public_transport.sqlite")
        (base_path / "results_database.sqlite").touch()
        pth = base_path / "run"
        pth.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(demo_init_py, pth / "__init__.py")

        parameters = Parameters(path=base_path)
        parameters.parameters["system"]["logging_directory"] = str(base_path)
        parameters.write_back()

    def list_scenarios(self):
        """
        Lists the existing scenarios.

        :Returns:
            **scenarios** (:obj:`pd.DataFrame`): Pandas DataFrame with existing scenarios
        """
        return pd.read_sql("SELECT * FROM scenarios", self.root_scenario.connections["project"])

    def use_scenario(self, scenario_name: str):
        """
        Switch the active scenario.

        :Arguments:
            **scenario_name** (:obj:`str`): name of the scenario to be activated

        """
        current = self._require_scenario()
        current.ensure_idle()
        self.root_scenario.ensure_idle()
        root_manager = self.root_scenario.connections["project"]
        if root_manager.execute("SELECT 1 FROM scenarios WHERE scenario_name=?", (scenario_name,)).fetchone() is None:
            raise ValueError(f"scenario '{scenario_name}' does not exist")
        if scenario_name == current.name:
            return

        candidate = self.root_scenario if scenario_name == "root" else Scenario.open_candidate(
            scenario_name, self.root_scenario.base_path / "scenarios" / scenario_name, create_auxiliary=False
        )[0]
        self.scenario = candidate
        try:
            self.__load_objects()
        except BaseException:
            self.scenario = current
            if candidate is not self.root_scenario:
                candidate.destroy()
            raise

        aequilibrae_logger = logging.getLogger("aequilibrae")
        aequilibrae_logger.removeHandler(current.log_handler)
        default_log_file_config(candidate.log_handler)
        if current is not self.root_scenario:
            current.destroy()

    def create_empty_scenario(self, scenario_name: str, description: str = ""):
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

    def clone_scenario(self, scenario_name: str, description: str = ""):
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

            shutil.copy(parameters_path, scenario_path)

            with commit_and_close(db, spatial=True) as conn:
                conn.execute("DROP TABLE IF EXISTS scenarios")

            with self.db_connection as conn:
                conn.execute(
                    "INSERT INTO scenarios (scenario_name, description) VALUES(?,?)", (scenario_name, description)
                )
        finally:
            self.use_scenario(current_scenario)
