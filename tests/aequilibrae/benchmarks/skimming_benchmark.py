"""Network skimming benchmarks on TNTP networks with centroid blocking."""

import numpy as np
import pandas as pd

from aequilibrae.paths.cython.context import NodeBasedContext, TurnBasedContext
from aequilibrae.paths.network_skimming import NetworkSkimming
from aequilibrae.paths.routing_context import make_routing_context

from .path_finding_benchmark import left_turn_restrictions, load_node_coordinates

SKIMMING_CASES = ("node_based", "centroid_turns")
SKIMMING_CASES_WITH_LEFT_TURNS = (*SKIMMING_CASES, "left_turns")
# Loose minimums for the TNTP networks used by these benchmarks.
MIN_CENTROID_BANS = {"Anaheim": 90, "Barcelona": 770, "ChicagoRegional": 1_600, "Winnipeg": 580}
MIN_LEFT_TURNS = {"Anaheim": 400, "ChicagoRegional": 28_000}


def zero_cost_road_turn(graph):
    """Pick a compact-graph road turn to enable turn-based routing without changing its cost."""
    arcs = graph.compact_graph
    ids = arcs["id"].to_numpy(np.int64, copy=False)
    tails = np.empty(graph.compact_num_links, dtype=np.int64)
    heads = np.empty(graph.compact_num_links, dtype=np.int64)
    tails[ids] = arcs["a_node"].to_numpy(np.int64, copy=False)
    heads[ids] = arcs["b_node"].to_numpy(np.int64, copy=False)

    for incoming in range(graph.compact_num_links):
        tail, via = int(tails[incoming]), int(heads[incoming])
        if tail < graph.num_zones or via < graph.num_zones:
            continue
        for outgoing in range(int(graph.compact_fs[via]), int(graph.compact_fs[via + 1])):
            destination = int(heads[outgoing])
            if destination >= graph.num_zones and destination != tail:
                nodes = graph.compact_all_nodes
                return pd.DataFrame(
                    {
                        "from_node": [int(nodes[tail])],
                        "via_node": [int(nodes[via])],
                        "to_node": [int(nodes[destination])],
                        "penalty": [0.0],
                    }
                )
    raise ValueError("No road turn is available to enable turn-based skimming")


def run_skimming_benchmark(benchmark, graph, model_stub, model_folder, routing_case):
    """Time skimming alone; prepare turn restrictions outside the timed call."""
    original_blocking = graph.block_centroid_flows
    graph.clear_turn_restrictions()
    graph.set_blocked_centroid_flows(True)
    try:
        if routing_case == "centroid_turns":
            graph.set_turn_restrictions(zero_cost_road_turn(graph))
        elif routing_case == "left_turns":
            nodes = load_node_coordinates(model_folder, model_stub)
            restrictions = left_turn_restrictions(graph, nodes)
            assert len(restrictions) >= MIN_LEFT_TURNS[model_stub], f"Too few left turns for {model_stub}"
            graph.set_turn_restrictions(restrictions)
        elif routing_case != "node_based":
            raise ValueError(f"Unknown skimming case: {routing_case}")

        # Skimming uses the compact context. Check the setup before timing it.
        context = make_routing_context(graph, compact=True)
        penalties = graph.compact_turn_penalties
        if routing_case == "node_based":
            assert isinstance(context, NodeBasedContext)
            assert context.blocked_centroid_count == graph.num_zones
            assert penalties.size == 0
        else:
            assert isinstance(context, TurnBasedContext)
            assert context.blocked_centroid_count == 0
            if routing_case == "centroid_turns":
                assert np.count_nonzero(penalties == 0) == 1
                assert np.count_nonzero(np.isinf(penalties)) >= MIN_CENTROID_BANS[model_stub]
                assert penalties.size == np.count_nonzero(np.isinf(penalties)) + 1
            else:
                assert np.isinf(penalties).all()
                assert penalties.size >= MIN_LEFT_TURNS[model_stub] + MIN_CENTROID_BANS[model_stub]

        skim = NetworkSkimming(graph)
        benchmark(skim.execute)
    finally:
        graph.clear_turn_restrictions()
        graph.set_blocked_centroid_flows(original_blocking)
