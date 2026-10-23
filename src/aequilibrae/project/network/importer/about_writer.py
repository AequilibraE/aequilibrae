"""Write network-import provenance into the project's ``about`` table."""

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Sequence

from aequilibrae import version as _aequilibrae_version

if TYPE_CHECKING:
    from aequilibrae.project import Project

logger = logging.getLogger(__name__)


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
        consolidate_tolerance,
        download_cache_relpath,
    ) -> None:
        values = {
            "network_source": source_meta["source"],
            "network_source_backend": source_meta["backend"],
            "network_source_url": source_meta["source_url"],
            "network_source_release": source_meta["release"],
            "network_source_fetched_at": source_meta["fetched_at"] or datetime.now(timezone.utc).isoformat(),
            "network_source_modes": ",".join(modes),
            "network_source_simplify": simplify,
            "network_source_consolidate_tolerance": "" if consolidate_tolerance is None else consolidate_tolerance,
            "network_source_download_cache": download_cache_relpath or "",
            "network_source_aequilibrae_version": _aequilibrae_version,
        }

        about = self.project.about
        for field_name, value in values.items():
            if field_name in about:
                about.update(field_name, infovalue=str(value))
            else:
                about.insert(infoname=field_name, infovalue=str(value))
        logger.info("Wrote network-import provenance to about table")
