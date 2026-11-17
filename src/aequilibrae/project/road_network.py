from collections.abc import Iterable
from typing import TYPE_CHECKING

import pandas as pd
from numpy.typing import ArrayLike
from shapely import Polygon

from aequilibrae.project.tables import Tables

if TYPE_CHECKING:
    from aequilibrae.routing.graph import Graph
    from aequilibrae.routing.network import Network


class RoadNetwork:
    """Build independent routing networks and graphs from a scenario's tables.

    Access this gateway through ``project.road`` or ``project.scenario.road``.
    """

    def __init__(self, tables: Tables) -> None:
        self._tables = tables

    def build_network(
        self,
        fields: list[str] | None = None,
        modes: list[str] | str | None = None,
        limit_to_area: Polygon | None = None,
    ) -> "Network":
        """Copy project tables into an independent routing Network.

        :Arguments:
            **fields** (:obj:`list`, *Optional*): When working with very large graphs with large number of fields in the
            database, it may be useful to specify which fields to use

            **modes** (:obj:`list`, *Optional*): Network modes to include in the source data

            **limit_to_area** (:obj:`Polygon`, *Optional*): When working with a very large model area, you may want to
            filter your database to a small area for your computation, which you can do by providing a polygon.
            The search is limited to a spatial index search, so it is very fast but NOT PRECISE.

        To use the 'fields' parameter, a minimalistic option is the following

        .. code-block:: python

            >>> project = create_example(project_path)

            >>> fields = ['distance']
            >>> network = project.road.build_network(fields, modes=['c', 'w'])

            >>> project.close()

        """
        from aequilibrae.routing.network import Network

        if isinstance(modes, str):
            modes = [modes]

        available = self._tables.list_modes()
        if modes is not None and (missing := set(modes) - set(available)):
            raise ValueError(f"Modes are not available in this network: {sorted(missing)}")

        modes = available if modes is None else modes
        links, centroids, restrictions, allow_uturns = self._tables._load_graph_data(fields, limit_to_area)

        return Network(
            links,
            modes=modes,
            centroids=centroids,
            turn_restrictions=restrictions,
            allow_uturns=allow_uturns,
            lonlat=self._tables.nodes.lonlat.set_index("node_id"),
        )

    def build_graph(
        self,
        mode: str,
        cost: str,
        *,
        fields: list[str] | None = None,
        centroids: ArrayLike | None = None,
        skimming: str | Iterable[str] | None = None,
        block_centroid_flows: bool = True,
        limit_to_area: Polygon | None = None,
        compact: bool = False,
        remove_dead_ends: bool = False,
        turn_restrictions: pd.DataFrame | None = None,
        allow_uturns: bool | None = None,
    ) -> "Graph":
        """Build a standalone graph without retaining project connections."""
        skimming = (skimming,) if isinstance(skimming, str) else tuple(() if skimming is None else skimming)
        if fields is not None:
            fields = list(dict.fromkeys([*fields, cost, *skimming]))
        return self.build_network(
            fields=fields,
            modes=[mode],
            limit_to_area=limit_to_area,
        ).build_graph(
            mode,
            cost,
            centroids=centroids,
            skimming=skimming,
            compact=compact,
            remove_dead_ends=remove_dead_ends,
            block_centroid_flows=block_centroid_flows,
            turn_restrictions=turn_restrictions,
            allow_uturns=allow_uturns,
        )
