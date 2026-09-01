import logging
import uuid
from pathlib import Path

from aequilibrae.project.project_creation import run_queries_from_sql_file
from aequilibrae.project.project_table import NonSpatialProjectTable
from aequilibrae.utils.db_utils import NestedTransactionManager, has_table

logger = logging.getLogger(__name__)


class About(NonSpatialProjectTable):
    """Project-table gateway for the project's key/value ``about`` metadata."""

    name = "about"
    key = "infoname"
    record_name = "AboutRecord"

    def __init__(self, connection: NestedTransactionManager) -> None:
        super().__init__(connection)

    def create(self) -> None:
        """Create and initialise the ``about`` table when it is absent."""
        if self.has_about:
            logger.warning("About table already exists. Nothing was done.")
            return

        schema = Path(__file__).parent / "database_specification" / "network" / "tables" / "about.sql"
        with self._connection.transaction() as conn:
            run_queries_from_sql_file(conn, schema)
            conn.execute("UPDATE about SET infovalue=? WHERE infoname='project_id'", (uuid.uuid4().hex,))
            conn.execute("UPDATE about SET infovalue='right' WHERE infoname='driving_side'")
        self._refresh_record_type()

    @property
    def has_about(self) -> bool:
        """Whether the project database contains an ``about`` table."""
        return has_table(self._connection._connection, self.name)
