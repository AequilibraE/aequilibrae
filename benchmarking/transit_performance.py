from os.path import join
import os

from aequilibrae.transit import Transit
from aequilibrae.project import Project

import numpy as np
import pandas as pd
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import TransitAssignment, TransitClass

import time


def run_example(from_model, zones):

    examples_dir = "temp_examples"
    time_results = []

    for n_zones in zones:

        path = join(examples_dir, f"{from_model}_{n_zones}")

        if os.path.isdir(path):

            project = Project.from_path(path)

        else:
            print("example not created.")
            continue

        print(path)

        data = Transit(project)

        graph = data.create_graph(
            with_outer_stop_transfers=False,
            with_walking_edges=False,
            blocking_centroid_flows=False,
            connector_method="overlapping_regions",
        )

        project.network.build_graphs()

        # c for All motorized vehicles
        graph.create_line_geometry(method="direct", graph="c")

        try:
            data.save_graphs()
        except Exception as e:
            print(f"graphs data already saved. {e.args}")

        data.load()

        data = Transit(project)

        # pt_con = database_connection("transit")
        # graph_db = TransitGraphBuilder.from_db(pt_con, periods.default_period.period_id)
        # graph_db.vertices.drop(columns="geometry")

        transit_graph = graph.to_transit_graph()

        zones_in_the_model = len(transit_graph.centroids)

        names_list = ["pt"]

        mat = AequilibraeMatrix()
        mat.create_empty(zones=zones_in_the_model, matrix_names=names_list, memory_only=True)
        mat.index = transit_graph.centroids[:]
        mat.matrices[:, :, 0] = np.full((zones_in_the_model, zones_in_the_model), 1.0)
        mat.computational_view()
        # mat.get_matrix('pt')

        transit_graph.graph["boardings"] = transit_graph.graph["link_type"].apply(lambda x: 1 if x == "boarding" else 0)
        transit_graph.graph["in_vehicle_trav_time"] = np.where(
            transit_graph.graph["link_type"].isin(["on-board", "dwell"]), 0, transit_graph.graph["trav_time"]
        )
        transit_graph.graph["egress_trav_time"] = np.where(
            transit_graph.graph["link_type"] != "egress_connector", 0, transit_graph.graph["trav_time"]
        )
        transit_graph.graph["access_trav_time"] = np.where(
            transit_graph.graph["link_type"] != "access_connector", 0, transit_graph.graph["trav_time"]
        )

        skim_cols = ["trav_time", "boardings", "in_vehicle_trav_time", "egress_trav_time", "access_trav_time"]

        assigclass = TransitClass(name="pt", graph=transit_graph, matrix=mat)

        print(f"centroids: {transit_graph.centroids.shape[0]}")
        for i in range(0, len(skim_cols)):

            assig = TransitAssignment()
            assig.add_class(assigclass)
            assig.set_time_field("trav_time")
            assig.set_frequency_field("freq")

            assig.set_skimming_fields(skim_cols[:i])

            assig.set_algorithm("os")
            assigclass.set_demand_matrix_core("pt")

            start_time = time.perf_counter()
            # start_time = time.process_time()
            assig.execute()
            end_time = time.perf_counter()
            # end_time = time.process_time()

            elapsed_time = end_time - start_time

            time_results.append([n_zones, i, elapsed_time])

            print(f"{i} Skimming Cols Elapsed time: {elapsed_time} seconds")

            if i != 0:
                assert len(assig.get_skim_results()[0].matrix) == i

            print(f"{len(assig.results())}")

            del assig

        project.close()

    return {from_model: pd.DataFrame(time_results, columns=["zones", "skim_cols", "elapsed_time"])}


from_model = "lyon"
results = run_example(from_model, [2000])
results = pd.read_csv(f"{from_model}_performance_results.csv", index_col=0)
