import itertools
import logging
from typing import Union
from collections import defaultdict

import geopandas as gpd
import pandas as pd
from aequilibrae.context import get_active_project
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import RouteChoice
from aequilibrae.paths.graph import Graph


class SubAreaAnalysis:
    def __init__(
        self,
        graph: Graph,
        subarea: gpd.GeoDataFrame,
        demand: Union[pd.DataFrame, AequilibraeMatrix],
        project=None,
    ):
        """
        Construct a sub-area matrix from a provided sub-area GeoDataFrame using route choice.

        This class aims to provide a semi-automated method for constructing the sub-area matrix.
        The user should provide the Graph object, demand matrix, and a GeoDataFrame whose geometry
        union represents the desired sub-area. Perform a route choice assignment, then call the
        ``post_process`` method to obtain a sub-area matrix.

        Check how to run sub-area analysis :ref:`here <example_usage_sub_area_analysis>`.

        :Arguments:
            **graph** (:obj:`Graph`): AequilibraE graph object to use

            **subarea** (:obj:`gpd.GeoDataFrame`): A GeoPandas GeoDataFrame whose geometry union
            represents the sub-area.

            **demand** (:obj:`Union[pandas.DataFrame, AequilibraeMatrix]`): The demand matrix to
            provide to the route choice assignment.
        """
        project = project if project is not None else get_active_project()
        self.logger = project.logger if project else logging.getLogger("aequilibrae")
        self.graph = graph
        self.sub_area_demand = None

        links = gpd.GeoDataFrame(project.network.links.data)
        self.interior_links = links[links.crosses(subarea.union_all().boundary)].sort_index()

        nodes = gpd.GeoDataFrame(project.network.nodes.data).set_index("node_id")
        self.interior_nodes = nodes.sjoin(subarea, how="inner").sort_index()

        self.interior_graph = (
            self.graph.graph.set_index("link_id")
            .loc[self.interior_links.link_id]
            .drop(self.graph.dead_end_links, errors="ignore")
            .reset_index()
            .set_index(["link_id", "direction"])
        )

        self.edge_pairs = {x: (x,) for x in itertools.permutations(self.interior_graph.index, r=2)}
        self.single_edges = {x: ((x,),) for x in self.interior_graph.index}
        self.logger.info(f"Created: {len(self.edge_pairs)} edge pairs from {len(self.single_edges)} edges")

        self.rc = RouteChoice(self.graph)
        self.rc.add_demand(demand)
        self.rc.set_select_links(self.single_edges | self.edge_pairs, link_loading=False)

    def post_process(self, demand_cols=None, keep_original_ods: bool = False):
        """
        Apply the necessary post processing to the route choice assignment select link results.

        :Arguments:
            **demand_cols** (*Optional*: :obj:`[list[str]]`): If provided, only construct the
              sub-area matrix for these demand matrices.

            **keep_original_ods** (*Optional*: :obj:`bool`): If provided, the original origin and destination IDs for
              the demand will be kept. This will create a significantly larger demand matrix but is more flexible.

        :Returns:
            **sub_area_demand** (:obj:`pd.DataFrame`): A DataFrame representing the sub-area
            demand matrix.
        """

        if demand_cols is None:
            demand_cols = self.rc.demand.df.columns

        sl_od = self.rc.get_select_link_od_matrix_results()

        sub_area_demand = []
        for col in demand_cols:
            edge_totals = {k: sl_od[k][col].to_scipy() for k in self.single_edges}
            edge_pair_values = {k: sl_od[k][col].to_scipy() for k in self.edge_pairs}

            entered = defaultdict(float)
            exited = defaultdict(float)

            for (link_id, dir), v in edge_totals.items():
                link = self.interior_graph.loc[link_id, dir]
                for (o, d), load in v.todok().items():
                    o = self.graph.all_nodes[o]
                    d = self.graph.all_nodes[d]

                    o_inside = o in self.interior_nodes.index
                    d_inside = d in self.interior_nodes.index

                    original_ods = (o, d) if keep_original_ods else ()
                    if o_inside and not d_inside:
                        exited[(o, self.graph.all_nodes[link.b_node]) + original_ods] += load
                    elif not o_inside and d_inside:
                        entered[(self.graph.all_nodes[link.a_node], d) + original_ods] += load
                    elif not o_inside and not d_inside:
                        pass

            through = defaultdict(float)
            for (l1, l2), v in edge_pair_values.items():
                link1 = self.interior_graph.loc[l1]
                link2 = self.interior_graph.loc[l2]

                for (o, d), load in v.todok().items():
                    o_inside = o in self.interior_nodes.index
                    d_inside = d in self.interior_nodes.index

                    original_ods = (o, d) if keep_original_ods else ()
                    if not o_inside and not d_inside:
                        through[
                            (self.graph.all_nodes[link1.a_node], self.graph.all_nodes[link2.b_node]) + original_ods
                        ] += load

            sub_area_demand.append(
                pd.DataFrame(
                    list(entered.values()) + list(exited.values()) + list(through.values()),
                    index=pd.MultiIndex.from_tuples(
                        list(entered.keys()) + list(exited.keys()) + list(through.keys()),
                        names=["origin id", "destination id"]
                        + (["original origin id", "original destination id"] if keep_original_ods else []),
                    ),
                    columns=[col],
                )
            )

        interior = []
        for o, d in self.rc.demand.df.index:
            if o in self.interior_nodes.index and d in self.interior_nodes.index:
                interior.append((o, d))

        self.sub_area_demand = pd.concat(sub_area_demand, axis=1).fillna(0.0)

        interior = self.rc.demand.df.loc[interior]
        if keep_original_ods:
            # We need to duplicate the interior ODs if we're "keeping the originals"
            interior.index = pd.MultiIndex.from_tuples(
                (x + x for x in interior.index), names=self.sub_area_demand.index.names
            )

        self.sub_area_demand = pd.concat([self.sub_area_demand, interior]).sort_index()
        return self.sub_area_demand
