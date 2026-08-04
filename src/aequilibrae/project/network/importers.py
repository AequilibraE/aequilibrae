"""``project.network.importer`` -- every way of getting a network *into* a project."""

import logging
from typing import Optional, TYPE_CHECKING

from shapely.geometry import Polygon

from aequilibrae.project.network.importer.schema.modes import DEFAULT_MODES

if TYPE_CHECKING:
    from aequilibrae.project.network.network import Network

logger = logging.getLogger(__name__)


class Importer:
    """Network import entry points, reached as ``project.network.importer``.

    .. code-block:: python

        >>> project.network.importer.overture(model_area=area)      # doctest: +SKIP
        >>> project.network.importer.osm(place_name="Nauru")        # doctest: +SKIP
        >>> project.network.importer.gmns(link_file_path=links_csv, node_file_path=nodes_csv)  # doctest: +SKIP
    """

    def __init__(self, network: "Network"):
        self._network = network
        self._project = network.project

    def source(
        self,
        source,
        *,
        modes=DEFAULT_MODES,
        simplify=False,
        consolidate_tolerance: Optional[float] = 10.0,
        cache_tag: str = "",
        **source_kwargs,
    ) -> None:
        """Import by explicit source name: ``osm-overpass``, ``osm-pbf`` or ``overture-cloud``.

        Prefer :meth:`osm` / :meth:`overture` unless you need to name the backend
        directly or pass source-specific keywords.

        :Arguments:
            **source** (:obj:`str`): Source name, or an object exposing ``name`` and ``acquire``

            **modes** (:obj:`tuple`, *Optional*): AequilibraE mode names to keep. Defaults to all

            **simplify** (:obj:`str`/:obj:`bool`, *Optional*): ``False`` (default, no simplification),
            ``"osmnx"`` or ``"neatnet"``

            **consolidate_tolerance** (:obj:`float`, *Optional*): Intersection consolidation radius in
            metres. ``None`` skips the consolidation pass for ``"osmnx"``; ``"neatnet"``, where
            consolidation is integral, falls back to its default

            **cache_tag** (:obj:`str`, *Optional*): Label for the raw-download cache folder
        """
        from aequilibrae.project.network.importer.importer import NetworkImporter

        NetworkImporter(self._project).run(
            source,
            modes=modes,
            simplify=simplify,
            consolidate_tolerance=consolidate_tolerance,
            cache_tag=cache_tag,
            **source_kwargs,
        )

    def osm(
        self,
        *,
        model_area: Optional[Polygon] = None,
        place_name: Optional[str] = None,
        pbf_path=None,
        modes=DEFAULT_MODES,
        custom_filter: Optional[str] = None,
        simplify=False,
        consolidate_tolerance: Optional[float] = 10.0,
    ) -> None:
        """Import a network from OpenStreetMap.

        Exactly one of ``model_area``, ``place_name`` or ``pbf_path`` must be given.
        XML (``.osm``, ``.osm.bz2``) is not supported; convert it first with
        ``osmium cat in.osm -o out.osm.pbf``.

        :Arguments:
            **model_area** (:obj:`Polygon`, *Optional*): EPSG:4326 polygon fetched through Overpass

            **place_name** (:obj:`str`, *Optional*): Place looked up through Nominatim/Overpass

            **pbf_path** (:obj:`str`, *Optional*): Path to a local ``.osm.pbf`` extract

            **modes** (:obj:`tuple`, *Optional*): AequilibraE mode names to keep. Defaults to all

            **custom_filter** (:obj:`str`, *Optional*): Raw Overpass way filter

            **simplify** (:obj:`str`/:obj:`bool`, *Optional*): ``False`` (default), ``"osmnx"`` or ``"neatnet"``

            **consolidate_tolerance** (:obj:`float`, *Optional*): Intersection consolidation radius in metres
        """
        provided = sum(x is not None for x in (model_area, place_name, pbf_path))
        if provided != 1:
            raise ValueError("network.importer.osm requires exactly one of: model_area, place_name, pbf_path")

        if pbf_path is not None:
            self.source(
                "osm-pbf",
                modes=modes,
                simplify=simplify,
                consolidate_tolerance=consolidate_tolerance,
                cache_tag=str(pbf_path),
                pbf_path=pbf_path,
            )
        else:
            self.source(
                "osm-overpass",
                modes=modes,
                simplify=simplify,
                consolidate_tolerance=consolidate_tolerance,
                cache_tag=place_name or "bbox",
                model_area=model_area,
                place_name=place_name,
                custom_filter=custom_filter,
            )

    def overture(
        self,
        *,
        model_area: Polygon,
        modes=DEFAULT_MODES,
        simplify=False,
        consolidate_tolerance: Optional[float] = 10.0,
    ) -> None:
        """Import a network from Overture Maps for an EPSG:4326 polygon.

        The latest release advertised by Overture's STAC catalog is always used, and
        recorded in the project's ``about`` table.

        :Arguments:
            **model_area** (:obj:`Polygon`): EPSG:4326 polygon to import

            **modes** (:obj:`tuple`, *Optional*): AequilibraE mode names to keep. Defaults to all

            **simplify** (:obj:`str`/:obj:`bool`, *Optional*): ``False`` (default), ``"osmnx"`` or ``"neatnet"``

            **consolidate_tolerance** (:obj:`float`, *Optional*): Intersection consolidation radius in metres
        """
        if model_area is None:
            raise ValueError("network.importer.overture requires a `model_area` Polygon")
        bounds = model_area.bounds
        self.source(
            "overture-cloud",
            modes=modes,
            simplify=simplify,
            consolidate_tolerance=consolidate_tolerance,
            cache_tag=f"bbox_{bounds[0]:.4f}_{bounds[1]:.4f}_{bounds[2]:.4f}_{bounds[3]:.4f}",
            model_area=model_area,
        )

    def gmns(
        self,
        link_file_path: str,
        node_file_path: str,
        use_group_path: str = "",
        geometry_path: str = "",
        srid: int = 4326,
    ) -> None:
        """Create an AequilibraE network from links and nodes in GMNS format.

        :Arguments:
            **link_file_path** (:obj:`str`): Path to a links csv file in GMNS format

            **node_file_path** (:obj:`str`): Path to a nodes csv file in GMNS format

            **use_group_path** (:obj:`str`, *Optional*): Path to a csv table containing groupings of uses.
            This helps AequilibraE know when a GMNS use is actually a group of other GMNS uses

            **geometry_path** (:obj:`str`, *Optional*): Path to a csv file containing geometry information
            for a line object, if not specified in the link table

            **srid** (:obj:`int`, *Optional*): Spatial Reference ID in which the GMNS geometries were created
        """
        from aequilibrae.project.network.gmns_builder import GMNSBuilder

        GMNSBuilder(self._network, link_file_path, node_file_path, use_group_path, geometry_path, srid).doWork()
        logger.info("Network built successfully")
