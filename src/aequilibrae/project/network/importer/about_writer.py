"""Whole-import provenance writer that targets the ``about`` table.

Per plan §10, the importer issues **no** ``ALTER TABLE`` statements on
``links`` or ``nodes``. Whole-import metadata lives in the existing
``about`` key/value table, using ``About.add_info_field()`` (idempotent —
only adds the field if it does not already exist).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Sequence

from aequilibrae.utils.db_utils import commit_and_close

if TYPE_CHECKING:
    from aequilibrae.project import Project

logger = logging.getLogger(__name__)


_FIELDS = (
    "network_source",
    "network_source_backend",
    "network_source_url",
    "network_source_release",
    "network_source_fetched_at",
    "network_source_modes",
    "network_source_simplify",
    "network_source_consolidate_tolerance",
    "network_source_download_cache",
    "network_source_aequilibrae_version",
)


def _aequilibrae_version() -> str:
    try:
        from aequilibrae import __version__

        return str(__version__)
    except Exception:  # pragma: no cover
        return "unknown"


class AboutWriter:
    """Writes ``network_source_*`` entries to the project's ``about`` table."""

    def __init__(self, project: "Project"):
        self.project = project

    def write(
        self,
        *,
        source_meta: dict,
        modes: Sequence[str],
        simplify: str,
        consolidate_tolerance: float | None,
        download_cache_relpath: str | None,
    ) -> None:
        values = {
            "network_source": str(source_meta.get("source", "")),
            "network_source_backend": str(source_meta.get("backend", "")),
            "network_source_url": str(source_meta.get("source_url", "")),
            "network_source_release": str(source_meta.get("release", "") or ""),
            "network_source_fetched_at": str(
                source_meta.get("fetched_at", "")
                or datetime.now(timezone.utc).isoformat()
            ),
            "network_source_modes": ",".join(modes),
            "network_source_simplify": str(simplify),
            "network_source_consolidate_tolerance": "" if consolidate_tolerance is None else f"{consolidate_tolerance}",
            "network_source_download_cache": "" if download_cache_relpath is None else download_cache_relpath,
            "network_source_aequilibrae_version": _aequilibrae_version(),
        }

        about = self.project.about
        existing = set(about.list_fields())
        for field_name in _FIELDS:
            if field_name not in existing:
                about.add_info_field(field_name)

        # ``about`` exposes characteristics as plain attributes; assign and write_back.
        for field_name, value in values.items():
            setattr(about, field_name, value)
        about.write_back()
        logger.info("Wrote network-import provenance to about table")
