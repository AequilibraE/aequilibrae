import logging
from pathlib import Path

from aequilibrae.reference_files import spatialite_database
from aequilibrae.utils.db_utils import ConnectionClosure, safe_connect
from aequilibrae.utils.spatialite_utils import connect_spatialite


class Scenario:
    """Own all persistent database resources for one project scenario."""

    def __init__(self, name: str, base_path: Path, log_handler: logging.Handler, connections: ConnectionClosure):
        self.name = name
        self.base_path = Path(base_path)
        self.path_to_file = self.base_path / "project_database.sqlite"
        self.log_handler = log_handler
        self._connections = connections
        self.network = None
        self.about = None
        self.matrices = None
        self.results = None
        self.transit = None
        self.zoning = None
        self._destroyed = False

    @classmethod
    def open_candidate(cls, name: str, base_path, *, create_auxiliary: bool = True):
        """Build a complete connection owner without mutating a Project."""
        base_path = Path(base_path)
        project_path = base_path / "project_database.sqlite"
        if not project_path.is_file():
            raise FileNotFoundError(f"Project database does not exist: {project_path}")
        results_path = base_path / "results_database.sqlite"
        transit_path = base_path / "public_transport.sqlite"
        created = []
        raw = []
        try:
            if not results_path.exists():
                if not create_auxiliary:
                    raise FileNotFoundError(f"Results database does not exist: {results_path}")
                results_path.touch()
                created.append(results_path)
            if not transit_path.exists():
                if not create_auxiliary:
                    raise FileNotFoundError(f"Transit database does not exist: {transit_path}")
                transit_path.write_bytes(Path(spatialite_database).read_bytes())
                created.append(transit_path)

            project = connect_spatialite(project_path)
            raw.append(project)
            results = safe_connect(results_path)
            raw.append(results)
            transit = connect_spatialite(transit_path)
            raw.append(transit)
            closure = ConnectionClosure({"project": project, "results": results, "transit": transit})
            raw.clear()  # closure now owns every connection
            return cls(name, base_path, logging.FileHandler(base_path / "aequilibrae.log"), closure), created
        except BaseException:
            for connection in raw:
                connection.close()
            for path in reversed(created):
                path.unlink(missing_ok=True)
            raise

    @property
    def connections(self):
        return self._connections

    def ensure_idle(self):
        self._connections.ensure_idle()

    def destroy(self):
        if self._destroyed:
            return
        self._connections.close()
        self.log_handler.close()
        self._destroyed = True
