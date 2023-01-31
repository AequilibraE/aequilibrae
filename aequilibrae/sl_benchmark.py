#!/usr/bin/env python3
import os
from pathlib import Path
from socket import gethostname
from argparse import ArgumentParser
from datetime import datetime
import sys
import timeit
import pandas as pd
import warnings
from aequilibrae import Project, TrafficAssignment, TrafficClass, AequilibraeMatrix
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))


def aequilibrae_init(
    proj_path: str,
    cost: str,
    cores: int = 0,
):
    """
    Prepare the graph for skimming the network for `cost`
    """
    proj = Project()
    proj.open(proj_path)
    # curr.execute("select st_x(geometry), st_y(geometry) from nodes")
    # geo = np.array(curr.fetchall())
    proj.network.build_graphs([cost, "capacity_ab", "capacity_ba"], ["c"])
    graph = proj.network.graphs["c"]
    matrix = proj.matrices.get_matrix("demand_omx")
    matrix.computational_view()
    # matrix.matrix_view = np.zeros((1790, 1790, 1))
    # matrix = AequilibraeMatrix()
    # matrix.create_empty(zones=graph.num_zones, matrix_names=["dummy"])
    #
    assignment = TrafficAssignment()
    car = TrafficClass("car", graph, matrix)
    assignment.set_classes([car])
    assignment.set_vdf("BPR")
    assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
    # assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})
    assignment.set_capacity_field("capacity")
    assignment.set_time_field("distance")
    assignment.max_iter = 1
    assignment.set_algorithm("msa")
    # assignment.set_cores(1)
    algorithms = ["msa", "cfw", "bfw", "frank-wolfe"]

    # And we will allow paths to be compute going through other centroids/centroid connectors
    # required for the Sioux Falls network, as all nodes are centroids
    # BE CAREFUL WITH THIS SETTING
    graph.set_blocked_centroid_flows(False)
    return graph, matrix, assignment, car


def arkansas():
    # from os.path import joinfrom
    from aequilibrae import Project
    from aequilibrae.paths import TrafficAssignment, TrafficClass

    # import logging import sys
    import numpy as np
    from aequilibrae import logger

    print("ark")
    pth = r"C:\Users\61435\Desktop\aequilibrae_performance_tests\models\Arkansas"
    proj = Project()
    proj.open(pth)
    net = proj.network
    curr = proj.conn.cursor()
    proj_matrices = proj.matrices
    modes = net.modes
    nodes = net.nodes  # These centroids are not in the matrices, so we turn them off
    for n in [4701, 4702, 4703]:
        nd = nodes.get(n)
        nd.is_centroid = 0
        nd.save()
    net.build_graphs(modes=["T"])
    truckGraph = net.graphs["T"]
    net.build_graphs(modes=["c"])
    carGraph = net.graphs["c"]
    # Link exclusions (equivalent to selections when creating TransCAD .net
    curr.execute("select link_id from links where exclusionset IN ('TruckOnly', 'HOV2', 'HOV3')")
    exclude_from_passenger = [x[0] for x in curr.fetchall()]
    curr.execute("select link_id from links where exclusionset IN ('PassengerOnly', 'HOV2', 'HOV3')")
    exclude_from_truck = [x[0] for x in curr.fetchall()]
    graph = truckGraph
    set1 = graph.network[(graph.network.builtyear > 2010) | (graph.network.removedyear < 2010)].link_id.to_list()
    set2 = graph.network[(graph.network.mode_code < 10) | (graph.network.mode_code > 11)].link_id.to_list()
    exclude_from_passenger.extend(set1 + set2)
    exclude_from_truck.extend(set1 + set2)
    truckGraph.exclude_links(exclude_from_truck)
    carGraph.exclude_links(exclude_from_passenger)
    # And turn them back into centroids to not alter the model
    for n in [4701, 4702, 4703]:
        nd = nodes.get(n)
        nd.is_centroid = 1
        nd.save()
    # Default parameters for BPR and tolls
    for graph in [carGraph, truckGraph]:
        graph.graph.alpha.fillna(0.15, inplace=True)
        graph.graph.beta.fillna(4.0, inplace=True)
        graph.graph.hov1tollcost.fillna(0, inplace=True)
        graph.graph.mttollcost.fillna(0, inplace=True)
        graph.graph.httollcost.fillna(0, inplace=True)
        # Sets capacities and travel times for links without any
        for period in ["am"]:  #'pm', 'md', 'nt']:
            graph.graph.loc[graph.graph.a_node == graph.graph.b_node, f"{period}_assncap_10"] = 1.0
            graph.graph.loc[graph.graph.a_node == graph.graph.b_node, f"tt_{period}_10"] = 0.001  # Assigns all periods
    for period in ["am"]:  # , 'md', 'pm', 'nt']:
        logger.info(f"\n\n Assigning {period.upper()}")
        proj_matrices = proj.matrices
        carDemand = proj_matrices.get_matrix(f"{period.upper()}_omx")
        carDemand.computational_view()  #'AUTO')
        lightTruckDemand = proj_matrices.get_matrix(f"{period.upper()}_omx")
        lightTruckDemand.computational_view("COMMTRK")
        lightTruckDemand.matrix_view = np.array(lightTruckDemand.matrix_view, np.float64)
        heavyTruckDemand = proj_matrices.get_matrix(f"{period.upper()}_omx")
        heavyTruckDemand.computational_view("HTRK")
        heavyTruckDemand.matrix_view = np.array(heavyTruckDemand.matrix_view, np.float64)
        assig = TrafficAssignment()
        assig.procedure_id = f"{period}_baseline"
        carClass = TrafficClass("car", carGraph, carDemand)
        carClass.set_pce(1)
        carClass.set_vot(0.2)
        carClass.set_fixed_cost("hov1tollcost")
        # The link exclusions for commercial trucks are actually the same as the ones for passenger cars, and not heavy trucks
        lightTruckClass = TrafficClass("light_truck", carGraph, lightTruckDemand)
        lightTruckClass.set_pce(1.5)
        lightTruckClass.set_vot(0.5)
        lightTruckClass.set_fixed_cost("mttollcost")

        heavyTruckClass = TrafficClass("heavy_truck", truckGraph, heavyTruckDemand)
        heavyTruckClass.set_pce(2.5)
        heavyTruckClass.set_vot(1.0)
        heavyTruckClass.set_fixed_cost("httollcost")
        # The first thing to do is to add at list of traffic classes to be assigned
        assig.set_classes([heavyTruckClass, carClass, lightTruckClass])
        assig.set_vdf("BPR")  # This is not case-sensitive # Then we set the volume delay function
        assig.set_vdf_parameters({"alpha": "alpha", "beta": "beta"})  # And its parameters
        assig.set_time_field(f"tt_{period}_10")
        assig.set_capacity_field(
            f"{period}_assncap_10"
        )  # The capacity and free flow travel times as they exist in the graph    # And the algorithm we want to use to assign    # NT is not converging properly (it is too loose, so even MSA converges incredibly fast)
        if period == "nt":
            assig.set_algorithm("msa")
            assig.max_iter = 50
            assig.rgap_target = 0.000001
        else:
            assig.set_algorithm("bfw")
            assig.max_iter = 50
            assig.rgap_target = 0.00001
    return assig, carClass
    assig.execute()  # we then execute the assignment
    assig.save_results(f"assign_{period}_gen_cost_path_file")
    proj.close()


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
    # parser.add_argument("-l", "--libraries", nargs='+', dest="libraries",
    #                     choices=libraries, default=libraries,
    #                     help="libraries to benchmark")
    parser.add_argument(
        "-p", "--projects", nargs="+", dest="projects", default=projects, help="projects to benchmark using"
    )
    parser.add_argument("--cost", dest="cost", default="distance", help="cost column to skim for")
    # parser.add_argument('--details', dest='details')
    parser.set_defaults(feature=True)

    args = vars(parser.parse_args())

    # libraries = args['libraries']
    output_path = args["output"]
    cores = args["cores"]
    print(f"Now benchmarking {libraries} on the {args['projects']} model(s).")
    # print(f"Running with {args['iters']} iterations, {args['repeats']}",
    #       f"times, for a total of {args['iters'] * args['repeats']} samples.")
    # Arkansas links
    # select_links = [None, {"test": [(24, 1), (79146, 1)], "test 2": [(61, 1), (68, 1)]}]
    # Chicago links
    select_links = [None, {"test": [(2, 1), (7, 1), (1, 1), (6, 1)], "set 2": [(1, 1), (3, 1)]}]

    with warnings.catch_warnings():
        # pandas future warnings are really annoying FIXME
        warnings.simplefilter(action="ignore", category=FutureWarning)
        # proj_path: str, cost: str, select_links, cores: int = 0,
        # Benchmark time
        results = []

        for project_name in args["projects"]:
            if project_name in "chicago_sketch":
                graph, matrix, assignment, car = aequilibrae_init(
                    f"{args['path']}/{project_name}", args["cost"], args["cores"]
                )
            elif project_name in "Arkansas":
                assignment, car = arkansas()
            else:
                raise Exception("Model Doesn't Exist Fool")
            assignment.set_cores(args["cores"][0])
            for link in select_links:
                if link is not None:
                    car.set_select_links(link)
                print("BENCHING links: ", link)

                t = timeit.Timer(lambda: assignment.execute())
                times = t.repeat(repeat=3, number=args["iters"])
                df = pd.DataFrame(
                    {
                        "Project": project_name,
                        "Select_Link": False if link is None else True,
                        "Minimum_Runtime": [min(times)],
                    }
                )
                results.append(df)
        ratios = []
        final = pd.concat(results)
        for project_name in args["projects"]:
            no_sl = final.loc[(final["Project"] == project_name) & (final["Select_Link"] == False)][
                "Minimum_Runtime"
            ].values[0]
            sl = final.loc[(final["Project"] == project_name) & (final["Select_Link"] == True)][
                "Minimum_Runtime"
            ].values[0]
            df = pd.DataFrame({"Project": project_name, "Runtime_%_Increase": [(sl - no_sl) / no_sl * 100]})
            ratios.append(df)
        final_ratio = pd.concat(ratios)
        print(final)
        print("\n")
        print(final_ratio)


if __name__ == "__main__":
    main()
