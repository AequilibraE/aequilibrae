import logging
from typing import TYPE_CHECKING, Sequence

from aequilibrae.project.network.importer.download_cache import DownloadCache
from aequilibrae.project.network.importer.exceptions import ImporterError
from aequilibrae.project.network.importer.schema.modes import DEFAULT_MODES
from aequilibrae.project.network.importer.simplifiers.base import resolve_simplifier
from aequilibrae.project.network.importer.sources.base import resolve_source
from aequilibrae.project.network.importer.staged_network import StagedNetwork

if TYPE_CHECKING:
    from aequilibrae.project import Project

logger = logging.getLogger(__name__)

_LINK_DEFAULTS = {
    "name": None,
    "speed_ab": None,
    "speed_ba": None,
    "lanes_ab": None,
    "lanes_ba": None,
    "source_id": None,
}
_NODE_DEFAULTS = {"source_id": None}

# Provenance keys persisted for every import. ``release`` is optional because
# not every source is versioned (e.g. a raw OSM PBF has no release tag).
REQUIRED_SOURCE_META_KEYS = ("source", "backend", "source_url", "fetched_at")
OPTIONAL_SOURCE_META_KEYS = ("release",)
_SOURCE_META_KEYS = REQUIRED_SOURCE_META_KEYS + OPTIONAL_SOURCE_META_KEYS


class NetworkImporter:
    def __init__(self, project: "Project"):
        self.project = project

    def run(
        self,
        source,
        *,
        modes: Sequence[str] = DEFAULT_MODES,
        simplify=False,
        consolidate_tolerance=10.0,
        cache_tag: str = "",
        **source_kwargs,
    ) -> None:
        from aequilibrae.project.network.importer.about_writer import AboutWriter
        from aequilibrae.project.network.importer.db_writer import SpatialiteWriter

        modes_tuple = tuple(modes)
        source_name, acquire = resolve_source(source, **source_kwargs)
        simplifier_name, simplify_fn = resolve_simplifier(simplify)

        download_cache = DownloadCache(
            project_base_path=self.project.project_base_path,
            source_name=source_name,
            tag=cache_tag or source_name,
        )

        logger.info(f"Acquiring network from source '{source_name}' (modes={modes_tuple})")
        net: StagedNetwork = acquire(modes=modes_tuple, download_cache=download_cache)
        _normalize_importer_columns(net)
        _normalize_source_meta(net)
        net.validate()
        logger.info(f"Acquired {len(net.nodes)} nodes and {len(net.links)} links")

        if simplify_fn is not None:
            logger.info(f"Simplifying with '{simplifier_name}'")
            net = simplify_fn(net, consolidate_tolerance=consolidate_tolerance)
            net.validate()
            logger.info(f"After simplification: {len(net.nodes)} nodes, {len(net.links)} links")

        logger.info("Saving network to the project database")
        SpatialiteWriter(self.project).write(net)

        AboutWriter(self.project).write(
            source_meta=net.source_meta,
            modes=modes_tuple,
            simplify=simplifier_name if simplify_fn is not None else "false",
            consolidate_tolerance=consolidate_tolerance if simplify_fn is not None else None,
            download_cache_relpath=download_cache.relative_path,
        )
        logger.info("Network build complete")


def _normalize_importer_columns(net: StagedNetwork) -> None:
    for column, default in _NODE_DEFAULTS.items():
        if column not in net.nodes.columns:
            net.nodes[column] = default
    for column, default in _LINK_DEFAULTS.items():
        if column not in net.links.columns:
            net.links[column] = default


def _normalize_source_meta(net: StagedNetwork) -> None:
    if not isinstance(net.source_meta, dict):
        raise ImporterError("StagedNetwork.source_meta must be a dict")

    missing = [key for key in REQUIRED_SOURCE_META_KEYS if key not in net.source_meta]
    if missing:
        raise ImporterError(f"StagedNetwork.source_meta missing required keys: {missing}")

    net.source_meta = {key: net.source_meta.get(key, "") for key in _SOURCE_META_KEYS}
