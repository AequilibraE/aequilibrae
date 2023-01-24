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
from aequilibrae import Project

sys.path.append(str(Path(__file__).resolve().parent))

def aequilibrae_init(proj_path: str, cost: str, cores: int = 0):
    """
    Prepare the graph for skimming the network for `cost`
    """
    proj = Project()
    proj.open(proj_path)
    # curr.execute("select st_x(geometry), st_y(geometry) from nodes")
    # geo = np.array(curr.fetchall())

    proj.network.build_graphs([cost])
    graph = proj.network.graphs["c"]
    graph.prepare_graph(graph.centroids)
    # let's say we want to minimize the cost
    graph.set_graph(cost)

    # And will skim the cost while we are at it
    graph.set_skimming(cost)

    # And we will allow paths to be compute going through other centroids/centroid connectors
    # required for the Sioux Falls network, as all nodes are centroids
    # BE CAREFUL WITH THIS SETTING
    graph.set_blocked_centroid_flows(False)
    return graph

def aequilibrae_init(data):
    """
    Prepare the graph for skimming the network for `cost`
    """
    graph = data["graph"]
    cost = data["cost"]
    cores = data["cores"]
    graph.prepare_graph(graph.centroids)
    # let's say we want to minimize the cost
    graph.set_graph(cost)

    # And will skim the cost while we are at it
    graph.set_skimming(cost)

    # And we will allow paths to be compute going through other centroids/centroid connectors
    # required for the Sioux Falls network, as all nodes are centroids
    # BE CAREFUL WITH THIS SETTING
    graph.set_blocked_centroid_flows(False)
    return (graph, cores)


def aequilibrae_testing(graph, cost: str, iters: int = 2, repeats: int = 5):
    graph = aequilibrae_init(graph, cost)
    t = timeit.Timer(lambda: aequilibrae_compute_skim(graph))
    times = t.repeat(repeat=repeats, number=iters)
    return times





def run_bench(lib, project_name, init, func, data):
    print(f"Running {lib} on {project_name} with {data['cores']} core(s)...")
    stuff = init(data)
    t = timeit.Timer(lambda: func(*stuff))
    df = pd.DataFrame({"runtime": [x / data["iters"] for x in t.repeat(repeat=data["repeats"], number=data["iters"])]})
    df["library"] = lib
    df["project_name"] = project_name
    df["cores"] = data['cores']
    df["computer"] = gethostname()
    if data["details"]:
        df["details"] = data["details"]
    return df


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
    parser.add_argument("--cost", dest="cost", default='free_flow_time',
                        help="cost column to skim for")
    # parser.add_argument('--details', dest='details')
    parser.set_defaults(feature=True)

    args = vars(parser.parse_args())

    libraries = args['libraries']
    output_path = args["output"]
    cores = args["cores"]
    print(f"Now benchmarking {libraries} on the {args['projects']} model(s).")
    print(f"Running with {args['iters']} iterations, {args['repeats']}",
          f"times, for a total of {args['iters'] * args['repeats']} samples.")
    with warnings.catch_warnings():
        # pandas future warnings are really annoying FIXME
        warnings.simplefilter(action="ignore", category=FutureWarning)

        # Benchmark time
        results = []
        proj_series = []
        for project_name in args["projects"]:
            args["graph"], args["nodes"], args["geo"] = project_init(f"{args['path']}/{project_name}", args["cost"])
            proj_series.append(pd.DataFrame({
                "num_links": [args["graph"].compact_num_links],
                "num_nodes": [args["graph"].compact_num_nodes],
                "num_zones": [args["graph"].num_zones],
                "num_centroids": [len(args["graph"].centroids)]
            }, index=[project_name]))

            for core_count in (range(cores[0], cores[1] + 1) if len(cores) == 2 else cores):
                args["cores"] = core_count

                if "aequilibrae" in libraries:
                    from aeq_testing import aequilibrae_init, aequilibrae_compute_skim
                    results.append(run_bench("aequilibrae", project_name, aequilibrae_init,
                                             aequilibrae_compute_skim, args))


                print("-" * 30)

        results = pd.concat(results)
        summary = results.groupby(["project_name", "library", "cores"]).agg(
            average=("runtime", "mean"), min=("runtime", "min"), max=("runtime", "max")
        )
        time = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        print(time)
        print(summary)
        results.to_csv(os.path.join(output_path, f"{time}_table.csv"))

        # proj_summary = pd.concat(proj_series)
        # proj_summary.to_csv(os.path.join(output_path, f"{time}_project_summary.csv"))





if __name__ == "__main__":
    main()
