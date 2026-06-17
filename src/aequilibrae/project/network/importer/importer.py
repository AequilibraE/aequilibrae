"""``NetworkImporter`` orchestrator.

Composes a ``Source`` → ``Simplifier`` → ``SpatialiteWriter`` →
``AboutWriter`` pipeline. The import is atomic: success writes the project,
failure leaves it unchanged (apart from any raw payload written to
``<project>/downloaded data/`` before the failure point).

See plan §4.1, §5.1.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

from .download_cache import DownloadCache
from .ir import RoutableNetwork
from .simplifiers.base import Simplifier, resolve_simplifier
from .sources.base import Source, resolve_source

if TYPE_CHECKING:
    from aequilibrae.project import Project

logger = logging.getLogger(__name__)


def _default_modes() -> tuple[str, ...]:
    return ("car", "transit", "bicycle", "walk")


class NetworkImporter:
    """Single-shot orchestrator for one network import."""

    def __init__(self, project: "Project"):
        self.project = project

    def run(
        self,
        source: "Source | str",
        *,
        modes: Sequence[str] = _default_modes(),
        simplify: "Simplifier | str | bool" = "osmnx",
        consolidate_tolerance: float | None = 10.0,
        cache_tag: str = "",
        **source_kwargs,
    ) -> None:
        """Run the full import pipeline atomically.

        :Arguments:
            **source**: ``Source`` instance or registered name (e.g.
            ``"osm-overpass"``, ``"osm-pbf"``, ``"overture-cloud"``,
            ``"geodataframe"``, ``"file"``, ``"gmns"``).

            **modes**: Sequence of AequilibraE mode names to retain. This is
            the only filter the importer applies.

            **simplify**: ``"osmnx"`` (default), ``"neatnet"``, or ``False``
            to skip simplification.

            **consolidate_tolerance**: Tolerance (m, in auto-UTM) for
            ``osmnx.consolidate_intersections``. Ignored by ``neatnet`` and
            when ``simplify=False``.

            **cache_tag**: Short human-readable label for the
            ``<project>/downloaded data/`` subfolder (e.g. place name or
            ``bbox_xmin_ymin_xmax_ymax``).
        """
        from .db_writer import SpatialiteWriter
        from .about_writer import AboutWriter

        modes_tuple = tuple(modes)
        source_obj = resolve_source(source, **source_kwargs)
        simplifier_obj = resolve_simplifier(simplify)

        download_cache = DownloadCache(
            project_base_path=self.project.project_base_path,
            source_name=source_obj.name,
            tag=cache_tag or source_obj.name,
        )

        logger.info(f"Acquiring network from source '{source_obj.name}' (modes={modes_tuple})")
        net: RoutableNetwork = source_obj.acquire(
            modes=modes_tuple,
            download_cache=download_cache,
        )
        net.validate()
        logger.info(f"Acquired {len(net.nodes)} nodes and {len(net.links)} links")

        if simplifier_obj is not None:
            logger.info(f"Simplifying with '{simplifier_obj.name}'")
            simplify_kwargs = {}
            if consolidate_tolerance is not None and simplifier_obj.name == "osmnx":
                simplify_kwargs["consolidate_tolerance"] = consolidate_tolerance
            net = simplifier_obj.simplify(net, **simplify_kwargs)
            net.validate()
            logger.info(f"After simplification: {len(net.nodes)} nodes, {len(net.links)} links")

        # Persist source provenance to the about table FIRST so it's always
        # recorded even if the spatialite write fails halfway.
        AboutWriter(self.project).write(
            source_meta=net.source_meta,
            modes=modes_tuple,
            simplify=simplifier_obj.name if simplifier_obj is not None else "false",
            consolidate_tolerance=consolidate_tolerance,
            download_cache_relpath=download_cache.relative_path,
        )

        SpatialiteWriter(self.project).write(net)
        logger.info("Network import complete")
