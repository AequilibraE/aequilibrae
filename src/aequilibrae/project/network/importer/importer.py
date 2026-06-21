import logging
from typing import TYPE_CHECKING, Sequence

from aequilibrae.project.network.importer.download_cache import DownloadCache
from aequilibrae.project.network.importer.simplifiers.base import resolve_simplifier
from aequilibrae.project.network.importer.sources.base import resolve_source
from aequilibrae.project.network.importer.staged_network import StagedNetwork

if TYPE_CHECKING:
    from aequilibrae.project import Project

logger = logging.getLogger(__name__)


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
        net.validate()
        logger.info(f"Acquired {len(net.nodes)} nodes and {len(net.links)} links")

        if simplifier_obj is not None:
            logger.info("Simplification started")
            logger.info(f"Simplifying with '{simplifier_obj.name}'")
            simplify_kwargs = {}
            if consolidate_tolerance is not None and simplifier_obj.name == "osmnx":
                simplify_kwargs["consolidate_tolerance"] = consolidate_tolerance
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
