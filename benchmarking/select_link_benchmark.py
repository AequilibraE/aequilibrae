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

import numpy as np
import pandas as pd

from aequilibrae import Project
from aequilibrae.traffic import TrafficAssignment, TrafficClass
from aequilibrae.traffic.vdf import bpr

sys.path.append(str(Path(__file__).resolve().parent.parent))


def aequilibrae_init(proj_path: str, cost: str) -> tuple[dict[str, object], TrafficClass]:
    """
    Prepare an independent network and assignment configuration for `cost`.
    """
    proj = Project.open(proj_path)
    try:
        network = proj.road.build_network(fields=[cost, "distance", "capacity"], modes=["c"])
        with proj.matrices.get_matrix("demand.omx") as store:
            matrix = store.subset(list(store))
        # Keep centroid connector turns allowed for Sioux Falls, since all nodes are centroids
        # BE CAREFUL WITH THIS SETTING
        car = TrafficClass(
            name="car", mode="c", cost=cost, demand=matrix, graph_options={"block_centroid_flows": False}
        )
        return {
            "network": network,
            "project": proj,
            "time_field": "distance",
            "capacity_field": "capacity",
            "vdf": bpr,
            "name_mapping": {"alpha": 0.15, "beta": 4.0},
            "algorithm": "msa",
        }, car
    except BaseException:
        proj.close()
        raise


def arkansas(path: str) -> tuple[dict[str, object], TrafficClass]:
    proj = Project.open(path)
    try:
        source = proj.road.build_network(modes=["c"])
        # These centroids are not in the matrices, so omit them from the independent snapshot.
        centroids = source.centroids[~np.isin(source.centroids, [4701, 4702, 4703])]
        links = source.links
        excluded = (
            links.exclusionset.isin(["PassengerOnly", "HOV2", "HOV3"])
            | (links.builtyear > 2010)
            | (links.removedyear < 2010)
            | (links.mode_code < 10)
            | (links.mode_code > 11)
        )
        links = links.loc[~excluded].copy()
        # No centroid changes need reverting: the project tables were not altered.
        for field, value in {"alpha": 0.15, "beta": 4.0, "hov1tollcost": 0, "mttollcost": 0, "httollcost": 0}.items():
            for column in (field, field + "_ab", field + "_ba"):
                if column in links:
                    links[column] = links[column].fillna(value)
        for field, value in {"am_assncap_10": 1.0, "tt_am_10": 0.001}.items():  # Assigns all periods
            for column in (field, field + "_ab", field + "_ba"):
                if column in links:
                    links.loc[links.a_node == links.b_node, column] = value
        turns = source.turn_restrictions
        if turns is not None:
            forward = links.loc[links.direction >= 0, ["a_node", "b_node"]]
            backward = links.loc[links.direction <= 0, ["b_node", "a_node"]].set_axis(["a_node", "b_node"], axis=1)
            edges = pd.MultiIndex.from_frame(pd.concat([forward, backward]))
            incoming = pd.MultiIndex.from_frame(turns[["from_node", "via_node"]])
            outgoing = pd.MultiIndex.from_frame(turns[["via_node", "to_node"]])
            turns = turns.loc[incoming.isin(edges) & outgoing.isin(edges)]
        network = replace(source, links=links, centroids=centroids, turn_restrictions=turns)
        period = "am"
        with proj.matrices.get_matrix(f"{period.upper()}.omx") as store:
            car_demand = store.subset(list(store))
        car_class = TrafficClass(
            name="car", mode="c", cost=f"tt_{period}_10", demand=car_demand, vot=0.2, fixed_cost_field="hov1tollcost"
        )
        return {
            "network": network,
            "project": proj,
            "time_field": f"tt_{period}_10",
            "capacity_field": f"{period}_assncap_10",
            "vdf": bpr,
            "name_mapping": {"alpha": "alpha", "beta": "beta"},
            "algorithm": "msa",
        }, car_class
    except BaseException:
        proj.close()
        raise


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
                assignment_options, car = aequilibrae_init(f"{args['path']}/{project_name}", args["cost"])
                dt = {}
                data = []
                select_links = [None]
                for sla in range(1, 21):
                    data.append((9000 + sla, 1))
                    dt[f"qry_{sla}"] = [(9000 + sla, 1)]
                    select_links.append(deepcopy(dt))

            elif project_name == "Arkansas":
                assignment_options, car = arkansas(f"{args['path']}/{project_name}")
                select_links = [None, {"test": [(24, 1), (79146, 1)], "test 2": [(61, 1), (68, 1)]}]
            else:
                raise ValueError(f"Unsupported benchmark model: {project_name}")
            for link in select_links:
                configuration = replace(car, select_link_sets=link or {})
                assignment = TrafficAssignment(classes=[configuration], **assignment_options)
                print("BENCHING links: ", link.keys() if link else link)
                qries = [] if link is None else list(link.values())

                t = timeit.Timer(
                    lambda: assignment.execute(max_iterations=1, rgap=0.00001, cores=args["cores"][0])  # noqa: B023
                )
                times = t.repeat(repeat=args["repeats"], number=args["iters"])
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
            assignment_options["project"].close()
        conn.close()
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
