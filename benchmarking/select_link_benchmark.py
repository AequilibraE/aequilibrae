#!/usr/bin/env python3
import sqlite3
import sys
import timeit
import warnings
from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import replace
from os.path import join
from pathlib import Path

import pandas as pd

from aequilibrae import Project, TrafficAssignment, TrafficClass
from aequilibrae.paths.vdf import bpr

sys.path.append(str(Path(__file__).resolve().parent.parent))


def aequilibrae_init(proj_path: str, cost: str):
    """
    Prepare the graph for skimming the network for `cost`
    """
    proj = Project.from_path(proj_path)
    network = proj.network.build(fields=[cost, "capacity"], modes=["c"])
    matrix = proj.matrices.get_matrix("demand_omx")
    matrix.computational_view()
    car = TrafficClass(name="car", mode="c", cost=cost, demand=matrix, block_centroid_flows=False)
    configuration = {
        "network": network,
        "vdf": bpr,
        "time_field": cost,
        "capacity_field": "capacity",
        "name_mapping": {"alpha": 0.15, "beta": 4.0},
        "algorithm": "msa",
        "project": proj,
    }
    return configuration, car


def arkansas(path: str):
    import logging

    from aequilibrae import Project
    from aequilibrae.paths import TrafficClass

    logger = logging.getLogger("aequilibrae")

    proj = Project.from_path(path)
    net = proj.network
    nodes = net.nodes
    for node in [4701, 4702, 4703]:
        nodes.update(node, is_centroid=0)
    network = net.build(modes=["c"])
    for node in [4701, 4702, 4703]:
        nodes.update(node, is_centroid=1)
    with proj.db_connection as conn:
        excluded = {
            row[0]
            for row in conn.execute(
                "select link_id from links where exclusionset IN ('PassengerOnly', 'HOV2', 'HOV3')"
            ).fetchall()
        }
    links = network.links
    unavailable = (links.builtyear > 2010) | (links.removedyear < 2010) | ~links.mode_code.between(10, 11)
    links.loc[unavailable | links.link_id.isin(excluded), "modes"] = ""
    defaults = {
        "alpha": 0.15,
        "beta": 4.0,
        "hov1tollcost": 0.0,
        "mttollcost": 0.0,
        "httollcost": 0.0,
        "am_assncap_10": 1.0,
        "tt_am_10": 0.001,
    }
    for field, default in defaults.items():
        for column in (field, field + "_ab", field + "_ba"):
            if column in links:
                links[column] = links[column].fillna(default)
    logger.info("Assigning AM")
    demand = proj.matrices.get_matrix("AM_omx")
    demand.computational_view()
    car = TrafficClass(name="car", mode="c", cost="tt_am_10", demand=demand, vot=0.2, fixed_cost_field="hov1tollcost")
    configuration = {
        "network": network,
        "vdf": bpr,
        "time_field": "tt_am_10",
        "capacity_field": "am_assncap_10",
        "name_mapping": {"alpha": "alpha", "beta": "beta"},
        "algorithm": "msa",
        "project": proj,
    }
    return configuration, car


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
                configuration, car = aequilibrae_init(f"{args['path']}/{project_name}", args["cost"])
                dt = {}
                data = []
                select_links = [None]
                for sla in range(1, 21):
                    data.append((9000 + sla, 1))
                    dt[f"qry_{sla}"] = [(9000 + sla, 1)]
                    select_links.append(deepcopy(dt))

            elif project_name in "Arkansas":
                configuration, car = arkansas(f"{args['path']}/{project_name}")
                select_links = [None, {"test": [(24, 1), (79146, 1)], "test 2": [(61, 1), (68, 1)]}]
            else:
                raise Exception("Model Doesn't Exist Fool")
            for link in select_links:
                traffic = replace(car, select_link_sets=link or {})
                assignment = TrafficAssignment(**configuration, classes=[traffic])
                print("BENCHING links: ", link.keys() if link else link)
                qries = [] if link is None else list(link.values())

                t = timeit.Timer(lambda: assignment.execute(max_iterations=1, rgap=1e-5, cores=args["cores"][0]))  # noqa: B023
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
            df = pd.DataFrame({"Project": [project_name], "Runtime_%_Increase": [(sl - no_sl) / no_sl * 100]})
            ratios.append(df)
        final_ratio = pd.concat(ratios) if len(ratios) != 1 else ratios[0]

        print(final)
        print("\n")
        print(final_ratio)


if __name__ == "__main__":
    main()
