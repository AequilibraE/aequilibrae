import argparse
import os
import time
from os.path import join

import numpy as np
import pandas as pd

from aequilibrae import Project
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.transit import Transit, TransitAssignment, TransitClass


def run_example(from_model: str, zones: list[int]) -> dict[str, pd.DataFrame]:
    examples_dir = "temp_examples"
    time_results = []

    for n_zones in zones:
        path = join(examples_dir, f"{from_model}_{n_zones}")

        if os.path.isdir(path):
            project = Project.open(path)

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

        # c for all motorised vehicles; direct geometry does not require a road graph.
        graph.create_line_geometry(method="direct", graph="c")

        try:
            data.save_graphs()
        except Exception as e:
            print(f"graphs data already saved. {e.args}")

        data.load()

        data = Transit(project)

        # Graphs can also be reloaded with TransitGraphBuilder.from_db(project, period_id).
        # graph_db = TransitGraphBuilder.from_db(project, project.tables.periods.default_period.period_id)
        # graph_db.vertices.drop(columns="geometry")

        transit_graph = graph.to_transit_graph()

        zones_in_the_model = len(transit_graph.centroids)

        mat = AequilibraeMatrix(index=transit_graph.centroids)
        mat["pt"] = np.full((zones_in_the_model, zones_in_the_model), 1.0)
        # mat["pt"] retrieves the demand array.

        transit_graph.graph["boardings"] = transit_graph.graph["link_type"].apply(lambda x: 1 if x == "boarding" else 0)
        transit_graph.graph["in_vehicle_trav_time"] = np.where(
            transit_graph.graph["link_type"].isin(["on-board", "dwell"]), transit_graph.graph["trav_time"], 0
        )
        transit_graph.graph["egress_trav_time"] = np.where(
            transit_graph.graph["link_type"] != "egress_connector", 0, transit_graph.graph["trav_time"]
        )
        transit_graph.graph["access_trav_time"] = np.where(
            transit_graph.graph["link_type"] != "access_connector", 0, transit_graph.graph["trav_time"]
        )

        skim_cols = ["trav_time", "boardings", "in_vehicle_trav_time", "egress_trav_time", "access_trav_time"]

        print(f"centroids: {transit_graph.centroids.shape[0]}")
        for i in range(len(skim_cols) + 1):
            transit_graph.set_skimming_fields(skim_cols[:i])
            assigclass = TransitClass(name="pt", graph=transit_graph, matrix=mat)
            assig = TransitAssignment(project=project)
            assig.add_class(assigclass)
            assig.set_time_field("trav_time")
            assig.set_frequency_field("freq")

            assig.set_skimming_fields(skim_cols[:i])

            assig.set_algorithm("os")

            start_time = time.perf_counter()
            # start_time = time.process_time()
            assig.execute()
            end_time = time.perf_counter()
            # end_time = time.process_time()

            elapsed_time = end_time - start_time

            time_results.append([n_zones, i, elapsed_time])

            print(f"{i} Skimming Cols Elapsed time: {elapsed_time} seconds")

            skims = assig.get_skim_results()["pt"]
            if i:
                assert set(skims) == set(skim_cols[:i])
            else:
                assert skims is None

            print(f"{len(assig.results())}")

            del assig

        project.close()

    return {from_model: pd.DataFrame(time_results, columns=["zones", "skim_cols", "elapsed_time"])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark transit assignment with varying numbers of skim fields")
    parser.add_argument("--model", default="lyon")
    parser.add_argument("--zones", nargs="+", type=int, default=[2000])
    args = parser.parse_args()
    results = run_example(args.model, args.zones)
    results[args.model].to_csv(f"{args.model}_performance_results.csv", index=False)


if __name__ == "__main__":
    main()
