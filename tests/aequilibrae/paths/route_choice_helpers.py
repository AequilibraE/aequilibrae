"""Shared route-choice setup; synthetic networks isolate turns, compression and identity."""

import numpy as np
import pandas as pd

from aequilibrae.matrix import from_df
from aequilibrae.paths import Network, RouteChoice


def junction(penalty, *, lonlat=None):
    links = pd.DataFrame(
        {
            "link_id": [11, 12, 13, 14],
            "a_node": [10, 10, 30, 20],
            "b_node": [20, 30, 20, 40],
            "direction": [1] * 4,
            "time": [1.0, 2.0, 1.0, 1.0],
        }
    )
    turns = (
        None
        if penalty is None
        else pd.DataFrame({"from_node": [10], "via_node": [20], "to_node": [40], "penalty": [penalty]})
    )
    return Network(links, centroids=np.array([10, 20, 30, 40]), turn_restrictions=turns, lonlat=lonlat).build_graph(
        "c", cost="time", compact=True, block_centroid_flows=False
    )


def diamond(penalty=0.5, *, cost="time", lonlat=None):
    links = pd.DataFrame(
        {
            "link_id": [71, 12, 55, 24],
            "a_node": [10, 20, 10, 30],
            "b_node": [20, 40, 30, 40],
            "direction": [1, 1, 1, 1],
            "time": [1.0, 1.0, 2.0, 2.0],
            "distance": [30.0, 40.0, 50.0, 60.0],
        }
    )
    turns = (
        None
        if penalty is None
        else pd.DataFrame({"from_node": [10], "via_node": [20], "to_node": [40], "penalty": [penalty]})
    )
    return Network(links, centroids=np.array([10, 20, 30, 40]), turn_restrictions=turns, lonlat=lonlat).build_graph(
        "c", cost=cost, compact=True, block_centroid_flows=False
    )


def compressed_graph(turn):
    links = pd.DataFrame(
        {
            "link_id": [71, 72, 12, 55, 56, 24],
            "a_node": [10, 20, 30, 10, 40, 50],
            "b_node": [20, 30, 60, 40, 50, 60],
            "direction": [1] * 6,
            "time": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        }
    )
    turns = pd.DataFrame({"from_node": [20], "via_node": [30], "to_node": [60], "penalty": [5.0]}) if turn else None
    lonlat = pd.DataFrame({"lon": np.arange(6) * 0.01, "lat": np.zeros(6)}, index=[10, 20, 30, 40, 50, 60])
    graph = Network(links, centroids=np.array([10, 60]), turn_restrictions=turns, lonlat=lonlat).build_graph(
        "c", cost="time", compact=True, block_centroid_flows=False
    )
    assert graph.num_nodes < graph.mapping.network.node_count
    return graph


def chain_graph():
    return Network(
        pd.DataFrame(
            {
                "link_id": [11, 12, 13],
                "a_node": [10, 20, 30],
                "b_node": [20, 30, 40],
                "direction": 1,
                "time": [1.0, 2.0, 3.0],
            }
        ),
        centroids=[10, 40],
    ).build_graph("c", "time", compact=True)


def mapped_network():
    # Physical rows are deliberately unsorted, with link 0 oriented against the forward route.
    links = pd.DataFrame(
        {
            "link_id": [0, 900, 81, 66, 75],
            "a_node": [200, 100, 200, 200, 100],
            "b_node": [150, 150, 300, 500, 700],
            "direction": [0, 0, 0, 0, 1],
            "modes": ["c", "c", "c", "c", "w"],
            "time_ab": [2.0, 1.0, 3.0, 9.0, 5.0],
            "time_ba": [4.0, 2.0, 6.0, 9.0, 5.0],
        },
        index=[50, 9, 2, 80, 1],
    )
    return Network(links, centroids=[300, 100])


def single_od_demand(origin, destination, flow=1.0):
    return from_df(
        pd.DataFrame({"origin id": [origin], "destination id": [destination], "flow": [flow]}),
        "origin id",
        "destination id",
        sparse=True,
    )


def configured_choice(graph):
    choice = RouteChoice(graph)
    choice.set_cores(1)
    choice.set_choice_set_generation("bfsle", max_routes=1)
    return choice


def sorted_results(choice):
    result = choice.get_results().copy()
    result["route set"] = result["route set"].map(tuple)
    result["route directions"] = result["route directions"].map(tuple)
    return result.sort_values(["origin id", "destination id", "route set"]).reset_index(drop=True)


def coordinates_for(graph, heuristic="euclidean"):
    x, y = ("x", "y") if heuristic == "euclidean" else ("lon", "lat")
    coordinates = pd.DataFrame(
        {x: np.arange(graph.num_nodes, dtype=float), y: np.zeros(graph.num_nodes)},
        index=graph.all_nodes,
    )
    # External IDs, not row order, determine the compact coordinate mapping.
    return coordinates.iloc[::-1]


def batched_junction_routes(graph, bfsle, max_routes):
    ods = [(10, 40), (20, 40)]
    frame = pd.DataFrame(
        {"flow": [1.0, 1.0]}, index=pd.MultiIndex.from_tuples(ods, names=["origin id", "destination id"])
    )
    demand = from_df(frame.reset_index(), "origin id", "destination id", index=graph.centroids, sparse=True)
    choice = configured_choice(graph)
    choice.set_choice_set_generation(
        "bfsle" if bfsle else "link-penalisation", max_routes=max_routes, max_depth=6, penalty=4.0
    )
    choice.set_demand(demand)
    choice.execute()
    return {
        (row["origin id"], row["destination id"], tuple(row["route set"])): row
        for _, row in choice.get_results().iterrows()
    }


def recompute_imported_routes(graph, routes, *, destination=40):
    choice = configured_choice(graph)
    choice.set_demand(single_od_demand(10, destination))
    supplied = pd.DataFrame(
        {
            "origin id": [10] * len(routes),
            "destination id": [destination] * len(routes),
            "route set": routes,
            "route directions": [[1] * len(route) for route in routes],
        }
    )
    choice.execute_from_pandas(supplied, recompute_psl=True)
    return choice.get_results()
