"""Write network-import provenance into the project's ``about`` table."""

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Sequence

from aequilibrae import version as _aequilibrae_version

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
            "network_source": str(source_meta["source"]),
            "network_source_backend": str(source_meta["backend"]),
            "network_source_url": str(source_meta["source_url"]),
            "network_source_release": str(source_meta["release"]),
            "network_source_fetched_at": str(source_meta["fetched_at"] or datetime.now(timezone.utc).isoformat()),
            "network_source_modes": ",".join(modes),
            "network_source_simplify": str(simplify),
            "network_source_consolidate_tolerance": (
                "" if consolidate_tolerance is None else f"{consolidate_tolerance}"
            ),
            "network_source_download_cache": ("" if download_cache_relpath is None else download_cache_relpath),
            "network_source_aequilibrae_version": str(_aequilibrae_version),
        }

        about = self.project.about
        existing = set(about.list_fields())
        for field_name in _FIELDS:
            if field_name not in existing:
                about.add_info_field(field_name)

        for field_name, value in values.items():
            setattr(about, field_name, value)
        about.write_back()
        logger.info("Wrote network-import provenance to about table")
