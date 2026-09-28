"""Path-search benchmarks on the TNTP networks."""

import json

import numpy as np
import pandas as pd
import pytest
from aequilibrae.paths.cython.a_star import EuclideanContext, a_star
from aequilibrae.paths.cython.context import GraphContext
from aequilibrae.paths.cython.dijkstra import dijkstra
from aequilibrae.paths.cython.queries import SearchQuery
from aequilibrae.paths.cython.search_results import SearchResults

from aequilibrae.paths.routing_context import make_routing_context
from aequilibrae.utils.list_all_turns import list_left_turns

TURN_PENALTY_MODELS = {"Anaheim", "ChicagoRegional", "SiouxFalls"}
OD_SAMPLE_SIZE = 20
OD_RANDOM_SEED = 20260928


def load_node_coordinates(folder, model_stub):
    """Load coordinates available alongside the TNTP networks."""
    if model_stub == "Anaheim":
        with open(folder / "anaheim_nodes.geojson") as source:
            features = json.load(source)["features"]
        return pd.DataFrame(
            [
                {
                    "node_id": feature["properties"]["id"],
                    "longitude": feature["geometry"]["coordinates"][0],
                    "latitude": feature["geometry"]["coordinates"][1],
                }
                for feature in features
            ]
        )

    node_files = list(folder.glob("*node.tntp"))
    if not node_files:
        return None

    nodes = pd.read_csv(node_files[0], sep=r"\s+", engine="python", usecols=[0, 1, 2])
    nodes.columns = ["node_id", "longitude", "latitude"]
    return nodes


def make_coordinates(graph, node_table):
    """Return coordinates in graph node order, or a zero heuristic if absent."""
    if node_table is None:
        values = np.zeros((len(graph.all_nodes), 2), dtype=np.float64)
        return values

    indexed = node_table.drop_duplicates("node_id").set_index("node_id")
    if not np.isin(graph.all_nodes, indexed.index).all():
        return np.zeros((len(graph.all_nodes), 2), dtype=np.float64)

    return indexed.loc[graph.all_nodes, ["longitude", "latitude"]].to_numpy(dtype=np.float64)


def real_nodes(graph):
    """Return road nodes, excluding blocked centroid nodes."""
    first_real_node = graph.num_zones if graph.block_centroid_flows else 0
    return graph.all_nodes[first_real_node:]


def left_turn_restrictions(graph, nodes):
    """Convert left-turn link pairs on real nodes to prohibited movements."""
    real_node_ids = real_nodes(graph)
    links = graph.network[["link_id", "a_node", "b_node", "direction"]].copy()
    if graph.block_centroid_flows:
        links = links.loc[links["a_node"].isin(real_node_ids) & links["b_node"].isin(real_node_ids)]
        nodes = nodes.loc[nodes["node_id"].isin(real_node_ids)]
    links["modes"] = "c"
    turns = list_left_turns(links, nodes, mode="c")
    turns = turns.loc[turns["turn_type"] == "left"]
    link_data = links.set_index("link_id")

    restrictions = []
    for turn in turns.itertuples(index=False):
        incoming = link_data.loc[turn.from_link_id]
        outgoing = link_data.loc[turn.to_link_id]
        via_node = int(turn.node_id)
        from_node = int(incoming.a_node if turn.from_dir == 1 else incoming.b_node)
        to_node = int(outgoing.b_node if turn.to_dir == 1 else outgoing.a_node)

        restrictions.append((from_node, via_node, to_node, np.inf))

    return pd.DataFrame(restrictions, columns=["from_node", "via_node", "to_node", "penalty"])


def make_od_sample(graph):
    rng = np.random.default_rng(OD_RANDOM_SEED)
    nodes = real_nodes(graph)
    pairs = []
    seen = set()

    while len(pairs) < OD_SAMPLE_SIZE:
        origin, destination = rng.choice(nodes, size=2, replace=False)
        pair = (int(origin), int(destination))
        if pair not in seen:
            pairs.append(pair)
            seen.add(pair)

    return pairs


def estimate_real_node_scale(context, graph, coordinates):
    """Estimate a safe A* scale from real-link cost/distance ratios."""
    offsets = np.asarray(context.fs, dtype=np.int64)
    tails = np.repeat(np.arange(context.node_count), np.diff(offsets))
    heads = context.heads
    costs = context.costs
    if graph.block_centroid_flows:
        real_links = (tails >= graph.num_zones) & (heads >= graph.num_zones)
    else:
        real_links = np.ones(context.link_count, dtype=bool)

    delta = coordinates[heads] - coordinates[tails]
    distances = np.hypot(delta[:, 0], delta[:, 1])
    valid = real_links & np.isfinite(costs) & (distances > 0)
    ratios = costs[valid] / distances[valid]
    return float(np.min(ratios)) if ratios.size else 0.0


def run_path_finding_search(benchmark, graph, model_stub, model_folder, algorithm, turn_penalties=False):
    graph.clear_turn_restrictions()
    node_table = load_node_coordinates(model_folder, model_stub)
    if turn_penalties:
        if model_stub not in TURN_PENALTY_MODELS:
            pytest.skip(f"No node coordinates available to classify left turns for {model_stub}")

        restrictions = left_turn_restrictions(graph, node_table)
        assert not restrictions.empty, f"No left turns found for {model_stub}"
        graph.set_turn_restrictions(restrictions)

    context: GraphContext = make_routing_context(graph)
    coordinates = make_coordinates(graph, node_table)

    heuristic = EuclideanContext(coordinates[:, 0], coordinates[:, 1], 1.0)
    scale = estimate_real_node_scale(context, graph, coordinates) if node_table is not None else 0.0
    heuristic.update_scale(scale)

    ods = make_od_sample(graph)

    node_indices = {int(node): index for index, node in enumerate(graph.all_nodes)}
    results = SearchResults(context.node_count, context.state_count, context.link_count)

    queries = []
    for origin, destination in ods:
        destination_index = node_indices[destination]
        mask = np.zeros(context.node_count, dtype=np.bool_)
        mask[destination_index] = True

        query = SearchQuery(context.node_count, node_indices[origin], mask)
        queries.append((query, destination_index))

    def run_searches():
        for query, destination in queries:
            if algorithm == "dijkstra":
                dijkstra(context, query, results)
            else:
                a_star(context, query, destination, heuristic, results)
        return results

    benchmark(run_searches)
