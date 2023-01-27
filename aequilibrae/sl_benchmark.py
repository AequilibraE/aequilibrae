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

def aequilibrae_init(proj_path: str, cost: str, cores: int = 0, ):
    """
    Prepare the graph for skimming the network for `cost`
    """
    proj = Project()
    proj.open(proj_path)
    # curr.execute("select st_x(geometry), st_y(geometry) from nodes")
    # geo = np.array(curr.fetchall())
    print(proj.matrices.list())
    #ARKANSAS SPECIFIC
    proj.network.build_graphs([cost, "capacity_ab", "capacity_ba"], ["c"])
    graph = proj.network.graphs["c"]
    # graph.prepare_graph(graph.centroids)
    # let's say we want to minimize the cost
    # graph.set_graph(cost)
    # print(graph.graph.head())
    # print(graph.graph.columns)
    # print(graph.compact_graph.head())
    # And will skim the cost while we are at it
    # graph.set_skimming(cost)
    # raise Exception()
    print('laoding')
    matrix = proj.matrices.get_matrix("demand_omx")
    print(matrix)
    print("loaded")
    matrix.computational_view()
    print(matrix.matrix_view)
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
    print(graph.graph.columns)
    assignment.set_capacity_field("capacity")
    assignment.set_time_field("distance")
    assignment.max_iter = 1
    assignment.set_algorithm("msa")
    assignment.set_cores(1)
    algorithms = ["msa", "cfw", "bfw", "frank-wolfe"]

    # And we will allow paths to be compute going through other centroids/centroid connectors
    # required for the Sioux Falls network, as all nodes are centroids
    # BE CAREFUL WITH THIS SETTING
    graph.set_blocked_centroid_flows(False)
    return graph, matrix, assignment, car

def main():
    projects = ["Arkansas"]
    libraries = ["aequilibrae"]

    parser = ArgumentParser()
    parser.add_argument("-m", "--model-path", dest="path", default='../models',
                        help="path to models", metavar="FILE")
    parser.add_argument("-o", "--output-path", dest="output", default='./Images',
                        help="where to place output data and images", metavar="FILE")
    parser.add_argument("-i", "--iterations", dest="iters", default=2, type=int,
                        help="number of times to run each library per sample", metavar="X")
    parser.add_argument("-r", "--repeats", dest="repeats", default=5, type=int,
                        help="number of samples", metavar="Y")
    parser.add_argument("-c", "--cores", nargs="+", dest="cores", default=[0],
                        help="number of cores to use. Use 0 for all cores.",
                        type=int, metavar="N")
    # parser.add_argument("-l", "--libraries", nargs='+', dest="libraries",
    #                     choices=libraries, default=libraries,
    #                     help="libraries to benchmark")
    parser.add_argument("-p", "--projects", nargs='+', dest="projects",
                        default=projects, help="projects to benchmark using")
    parser.add_argument("--cost", dest="cost", default='distance',
                        help="cost column to skim for")
    # parser.add_argument('--details', dest='details')
    parser.set_defaults(feature=True)

    args = vars(parser.parse_args())

    # libraries = args['libraries']
    output_path = args["output"]
    cores = args["cores"]
    print(f"Now benchmarking {libraries} on the {args['projects']} model(s).")
    print(f"Running with {args['iters']} iterations, {args['repeats']}",
          f"times, for a total of {args['iters'] * args['repeats']} samples.")
    # Arkansas links
    # select_links = [None, {"test": [(24, 1), (79146, 1), (61, 1), (68, 1)]}]
    # Chicago links
    select_links = [None,
                    {"test": [(2, 1), (7, 1), (1, 1), (6, 1)]}]


    with warnings.catch_warnings():
        # pandas future warnings are really annoying FIXME
        warnings.simplefilter(action="ignore", category=FutureWarning)
        # proj_path: str, cost: str, select_links, cores: int = 0,
        # Benchmark time
        results = []
        proj_series = []
        project_name = args["projects"][0]
        graph, matrix, assignment, car = aequilibrae_init(f"{args['path']}/{project_name}",
                                                                  args["cost"], args["cores"])
        for link in select_links:
            for project_name in args["projects"]:
                print("select links is ", link)
                if link is not None:
                    car.set_select_links(link)
                print("BENCHING ")
            # args["graph"], args["nodes"], args["geo"] = project_init(f"{args['path']}/{project_name}", args["cost"])
            # proj_series.append(pd.DataFrame({
            #     "num_links": [args["graph"].compact_num_links],
            #     "num_nodes": [args["graph"].compact_num_nodes],
            #     "num_zones": [args["graph"].num_zones],
            #     "num_centroids": [len(args["graph"].centroids)]
            # }, index=[project_name]))

            # for core_count in (range(cores[0], cores[1] + 1) if len(cores) == 2 else cores):
            #     args["cores"] = core_count
                t = timeit.Timer(lambda: assignment.execute())
                times = t.repeat(repeat=3, number=args["iters"])
                results.append(("SL" if link is not None else "BASE", min(times)))


            # print("-" * 30)
        print(results)
        print("BASE/SL", results[0][1]/results[1][1])
        #
        # results = pd.concat(results)
        # summary = results.groupby(["project_name", "library", "cores"]).agg(
        #     average=("runtime", "mean"), min=("runtime", "min"), max=("runtime", "max")
        # )
        # time = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        # print(time)
        # print(summary)
        # results.to_csv(os.path.join(output_path, f"{time}_table.csv"))

        # proj_summary = pd.concat(proj_series)
        # proj_summary.to_csv(os.path.join(output_path, f"{time}_project_summary.csv"))





if __name__ == "__main__":
    main()
