"""Internal SQLite helpers independent of project storage."""

import logging
from contextlib import contextmanager
from sqlite3 import Connection
from typing import Iterator

logger = logging.getLogger(__name__)


@contextmanager
def without_triggers(conn: Connection) -> Iterator[None]:
    """Temporarily remove installed triggers, then restore their exact SQL definitions.

    The caller is responsible for running this context within a managed transaction.
    """
    triggers = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='trigger' AND sql IS NOT NULL ORDER BY name"
    ).fetchall()
    try:
        for name, _ in triggers:
            logger.debug("Temporarily removing trigger %s", name)
            quoted_name = name.replace('"', '""')
            conn.execute(f'DROP TRIGGER "{quoted_name}"')
        yield
    finally:
        for name, sql in triggers:
            exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='trigger' AND name=?", (name,)).fetchone()
            if exists is None:
                logger.debug("Restoring trigger %s", name)
                conn.execute(sql)
