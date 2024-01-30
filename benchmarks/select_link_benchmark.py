#!/usr/bin/env python3
import sqlite3
from copy import deepcopy
from os.path import join
from pathlib import Path
from argparse import ArgumentParser
import sys
import timeit
import pandas as pd
import warnings

sys.path.append(str(Path(__file__).resolve().parent.parent))

from aequilibrae import Project, TrafficAssignment, TrafficClass  # noqa: E402


def aequilibrae_init(proj_path: str, cost: str):
    """
    Prepare the graph for skimming the network for `cost`
    """
    proj = Project()
    proj.open(proj_path)
    proj.network.build_graphs([cost, "capacity_ab", "capacity_ba"], ["c"])
    graph = proj.network.graphs["c"]
    matrix = proj.matrices.get_matrix("demand_omx")
    matrix.computational_view()
    assignment = TrafficAssignment()
    car = TrafficClass("car", graph, matrix)
    assignment.set_classes([car])
    assignment.set_vdf("BPR")
    assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
    assignment.set_capacity_field("capacity")
    assignment.set_time_field("distance")
    assignment.max_iter = 1
    assignment.set_algorithm("msa")
    # And we will allow paths to be compute going through other centroids/centroid connectors
    # required for the Sioux Falls network, as all nodes are centroids
    # BE CAREFUL WITH THIS SETTING
    graph.set_blocked_centroid_flows(False)
    return graph, matrix, assignment, car


def arkansas(path: str):
    from aequilibrae import Project
    from aequilibrae.paths import TrafficAssignment, TrafficClass
    from aequilibrae import logger

    proj = Project()
    proj.open(path)
    net = proj.network
    curr = proj.conn.cursor()
    nodes = net.nodes  # These centroids are not in the matrices, so we turn them off
    for n in [4701, 4702, 4703]:
        nd = nodes.get(n)
        nd.is_centroid = 0
        nd.save()
    net.build_graphs(modes=["c"])
    car_graph = net.graphs["c"]
    exclude_from_passenger = [x[0] for x in curr.fetchall()]
    curr.execute("select link_id from links where exclusionset IN ('PassengerOnly', 'HOV2', 'HOV3')")
    graph = car_graph
    set1 = graph.network[(graph.network.builtyear > 2010) | (graph.network.removedyear < 2010)].link_id.to_list()
    set2 = graph.network[(graph.network.mode_code < 10) | (graph.network.mode_code > 11)].link_id.to_list()
    exclude_from_passenger.extend(set1 + set2)
    car_graph.exclude_links(exclude_from_passenger)
    # And turn them back into centroids to not alter the model
    for n in [4701, 4702, 4703]:
        nd = nodes.get(n)
        nd.is_centroid = 1
        nd.save()
    car_graph.graph.alpha.fillna(0.15, inplace=True)
    car_graph.graph.beta.fillna(4.0, inplace=True)
    car_graph.graph.hov1tollcost.fillna(0, inplace=True)
    car_graph.graph.mttollcost.fillna(0, inplace=True)
    car_graph.graph.httollcost.fillna(0, inplace=True)
    car_graph.graph.loc[car_graph.graph.a_node == car_graph.graph.b_node, "am_assncap_10"] = 1.0
    car_graph.graph.loc[car_graph.graph.a_node == car_graph.graph.b_node, "tt_am_10"] = 0.001  # Assigns all periods
    period = "am"
    logger.info(f"\n\n Assigning {period.upper()}")
    proj_matrices = proj.matrices
    car_demand = proj_matrices.get_matrix(f"{period.upper()}_omx")
    car_demand.computational_view()  # 'AUTO')
    assig = TrafficAssignment()
    assig.procedure_id = f"{period}_baseline"
    car_class = TrafficClass("car", car_graph, car_demand)
    car_class.set_pce(1)
    car_class.set_vot(0.2)
    car_class.set_fixed_cost("hov1tollcost")
    assig.set_classes([car_class])
    assig.set_vdf("BPR")  # This is not case-sensitive # Then we set the volume delay function
    assig.set_vdf_parameters({"alpha": "alpha", "beta": "beta"})  # And its parameters
    assig.set_time_field(f"tt_{period}_10")
    assig.set_capacity_field(f"{period}_assncap_10")
    assig.max_iter = 1
    assig.set_algorithm("msa")
    assig.rgap_target = 0.00001
    return assig, car_class


def main():
    projects = ["Arkansas"]
    libraries = ["aequilibrae"]
    parser = ArgumentParser()
    parser.add_argument("-m", "--model-path", dest="path", default="../models", help="path to models", metavar="FILE")
    parser.add_argument(
        "-o",
        "--output-path",
        dest="output",
        default="./Images",
        help="where to place output data and images",
        metavar="FILE",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        dest="iters",
        default=2,
        type=int,
        help="number of times to run each library per sample",
        metavar="X",
    )
    parser.add_argument("-r", "--repeats", dest="repeats", default=5, type=int, help="number of samples", metavar="Y")
    parser.add_argument(
        "-c",
        "--cores",
        nargs="+",
        dest="cores",
        default=[0],
        help="number of cores to use. Use 0 for all cores.",
        type=int,
        metavar="N",
    )
    parser.add_argument(
        "-p", "--projects", nargs="+", dest="projects", default=projects, help="projects to benchmark using"
    )
    parser.add_argument("--cost", dest="cost", default="distance", help="cost column to skim for")
    parser.set_defaults(feature=True)
    args = vars(parser.parse_args())
    print(f"Now benchmarking {libraries} on the {args['projects']} model(s).")
    with warnings.catch_warnings():
        # pandas future warnings are really annoying FIXME
        warnings.simplefilter(action="ignore", category=FutureWarning)
        # Benchmark time
        results = []
        conn = sqlite3.connect(join(args["path"], "r.sqlite"))
        for project_name in args["projects"]:
            if project_name in ["chicago_sketch"]:
                graph, matrix, assignment, car = aequilibrae_init(f"{args['path']}/{project_name}", args["cost"])
                dt = {}
                data = []
                select_links = [None]
                for sla in range(1, 21):
                    data.append((9000 + sla, 1))
                    dt[f"qry_{sla}"] = [(9000 + sla, 1)]
                    select_links.append(deepcopy(dt))

            elif project_name in "Arkansas":
                assignment, car = arkansas(f"{args['path']}/{project_name}")
                select_links = [None, {"test": [(24, 1), (79146, 1)], "test 2": [(61, 1), (68, 1)]}]
            else:
                raise Exception("Model Doesn't Exist Fool")
            assignment.set_cores(args["cores"][0])
            for link in select_links:
                if link is not None:
                    car.set_select_links(link)
                print("BENCHING links: ", link.keys() if link else link)
                qries = [] if link is None else list(link.values())

                t = timeit.Timer(lambda: assignment.execute())  # noqa: B023
                times = t.repeat(repeat=3, number=args["iters"])
                df = pd.DataFrame(
                    {
                        "Project": project_name,
                        "Select_Link": False if link is None else True,
                        "queries": 0 if link is None else len(link),
                        "queries_size": sum([len(x) for x in qries]) / max(1, len(qries)),
                        "Minimum_Runtime": [min(times)],
                        "cores": args["cores"][0],
                    }
                )
                results.append(df)
                df.to_sql("benchmarking", conn, if_exists="append")
        ratios = []
        final = pd.concat(results)
        for project_name in args["projects"]:
            no_sl = final.query(f"Project == '{project_name}' and Select_Link == False")["Minimum_Runtime"]
            sl = final.query(f"Project == '{project_name}' and Select_Link == True")["Minimum_Runtime"]
            df = pd.DataFrame({"Project": [project_name], "Runtime_%_Increase": [((sl - no_sl) / no_sl * 100)]})
            ratios.append(df)
        final_ratio = pd.concat(ratios) if len(ratios) != 1 else ratios[0]

        print(final)
        print("\n")
        print(final_ratio)


if __name__ == "__main__":
    main()
