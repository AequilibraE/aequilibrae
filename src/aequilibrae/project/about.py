import logging
import string
import uuid
from pathlib import Path

from aequilibrae.project.project_creation import run_queries_from_sql_file
from aequilibrae.utils.db_utils import NestedTransactionManager

logger = logging.getLogger(__name__)


class About:
    """Read and edit the project ``about`` table through its persistent manager."""

    def __init__(self, connection: NestedTransactionManager) -> None:
        self._transaction_manager = connection
        self.__characteristics = []
        self.__original = {}
        if self.__has_about():
            self.__load()

    def create(self) -> None:
        with self._transaction_manager.transaction() as conn:
            if not self.__has_about():
                schema = Path(__file__).parent / "database_specification" / "tables" / "about.sql"
                run_queries_from_sql_file(self._transaction_manager.connection, schema)
            count = conn.execute("SELECT count(*) FROM about").fetchone()[0]
            if count == 0:
                conn.execute("UPDATE about SET infovalue=? WHERE infoname='project_ID'", (uuid.uuid4().hex,))
                conn.execute("UPDATE about SET infovalue='right' WHERE infoname='driving_side'")
                self.__load()
            else:
                logger.warning("About table already exists. Nothing was done.")

    def list_fields(self) -> list[str]:
        return list(self.__characteristics)

    def add_info_field(self, info_field: str) -> None:
        allowed = string.ascii_lowercase + "_"
        if any(character not in allowed for character in info_field):
            raise ValueError(f"{info_field} is not valid as a metadata field. Should be a lower case ascii letter or _")
        with self._transaction_manager.transaction() as conn:
            conn.execute("INSERT INTO about (infoname) VALUES(?)", (info_field,))
        self.__characteristics.append(info_field)
        self.__original[info_field] = None

    def write_back(self) -> None:
        with self._transaction_manager.transaction() as conn:
            for key in self.__characteristics:
                value = self.__dict__[key]
                if value != self.__original[key]:
                    conn.execute("UPDATE about SET infovalue=? WHERE infoname=?", (value, key))
                    self.__original[key] = value
                    logger.info("Updated %s on About_Table", key)

    def __has_about(self) -> bool:
        return (
            self._transaction_manager.connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='about'"
            ).fetchone()
            is not None
        )

    def __load(self) -> None:
        self.__characteristics = []
        for name, value in self._transaction_manager.connection.execute(
            "SELECT infoname, infovalue FROM about"
        ).fetchall():
            self.__characteristics.append(name)
            self.__dict__[name] = value
            self.__original[name] = value
