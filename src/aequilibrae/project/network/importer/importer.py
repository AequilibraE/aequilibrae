import logging
from typing import TYPE_CHECKING, Sequence

from aequilibrae.project.network.importer.download_cache import DownloadCache
from aequilibrae.project.network.importer.exceptions import ImporterError
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

# Provenance keys persisted for every import. ``release`` is OPTIONAL because not
# every source is versioned (e.g. a raw OSM PBF or Overpass query has no release
# tag), whereas Overture imports do. Splitting the two avoids rejecting valid
# unversioned imports while still recording the release when a source provides
# one. Keep these lists explicit so the contract is not silently mutated.
REQUIRED_SOURCE_META_KEYS = ("source", "backend", "source_url", "fetched_at")
OPTIONAL_SOURCE_META_KEYS = ("release",)
_SOURCE_META_KEYS = REQUIRED_SOURCE_META_KEYS + OPTIONAL_SOURCE_META_KEYS


def _default_modes():
    return ("car", "transit", "bicycle", "walk")


class NetworkImporter:
    def __init__(self, project: "Project"):
        self.project = project

    def run(
        self,
        source,
        *,
        modes: Sequence[str] = _default_modes(),
        simplify="osmnx",
        consolidate_tolerance=10.0,
        cache_tag: str = "",
        metrics: dict | None = None,
        **source_kwargs,
    ) -> None:
        from aequilibrae.project.network.importer.about_writer import AboutWriter
        from aequilibrae.project.network.importer.db_writer import SpatialiteWriter

        modes_tuple = tuple(modes)
        source_obj = resolve_source(source, **source_kwargs)
        simplifier_obj = resolve_simplifier(simplify)

        download_cache = DownloadCache(
            project_base_path=self.project.project_base_path,
            source_name=source_obj.name,
            tag=cache_tag or source_obj.name,
        )

        logger.info(f"Acquiring network from source '{source_obj.name}' (modes={modes_tuple})")
        logger.info("Data download started")
        net: StagedNetwork = source_obj.acquire(modes=modes_tuple, download_cache=download_cache)
        logger.info("Data download finished")
        _normalize_importer_columns(net)
        _normalize_source_meta(net)
        net.validate()
        logger.info(f"Acquired {len(net.nodes)} nodes and {len(net.links)} links")

        if simplifier_obj is not None:
            logger.info("Simplification started")
            logger.info(f"Simplifying with '{simplifier_obj.name}'")
            simplify_kwargs = {}
            if consolidate_tolerance is not None and simplifier_obj.name == "osmnx":
                simplify_kwargs["consolidate_tolerance"] = consolidate_tolerance
            if simplifier_obj.name == "neatnet":
                from aequilibrae.project.network.importer.buildings import fetch_building_footprints

                buildings = fetch_building_footprints(net, download_cache)
                net.source_meta.update(buildings.as_meta())
                if buildings.gdf is not None:
                    simplify_kwargs["exclusion_mask"] = buildings.gdf
            if metrics is not None:
                simplify_kwargs["metrics"] = metrics
            net = simplifier_obj.simplify(net, **simplify_kwargs)
            net.validate()
            logger.info(f"After simplification: {len(net.nodes)} nodes, {len(net.links)} links")
            logger.info("Simplification finished")

        AboutWriter(self.project).write(
            source_meta=net.source_meta,
            modes=modes_tuple,
            simplify=simplifier_obj.name if simplifier_obj is not None else "false",
            consolidate_tolerance=consolidate_tolerance,
            download_cache_relpath=download_cache.relative_path,
        )

        logger.info("Saving to the database started")
        SpatialiteWriter(self.project).write(net)
        logger.info("Saving to the database finished")
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

