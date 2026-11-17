"""Network skimming benchmarks on TNTP networks with centroid blocking."""

import numpy as np
import pandas as pd
from aequilibrae.paths.cython.context import NodeBasedContext, TurnBasedContext

from aequilibrae.paths.network_skimming import NetworkSkimming

from .path_finding_benchmark import left_turn_restrictions, load_node_coordinates

SKIMMING_CASES = ("node_based", "centroid_turns")
SKIMMING_CASES_WITH_LEFT_TURNS = (*SKIMMING_CASES, "left_turns")
# Loose minimums for the TNTP networks used by these benchmarks.
MIN_CENTROID_BANS = {"Anaheim": 90, "Barcelona": 770, "ChicagoRegional": 1_600, "Winnipeg": 580}
MIN_LEFT_TURNS = {"Anaheim": 400, "ChicagoRegional": 28_000}


def zero_cost_road_turn(links, centroids):
    """Pick a real road turn to enable turn-based routing without changing its cost."""
    forward = links.loc[links.direction != -1, ["a_node", "b_node"]]
    backward = links.loc[links.direction != 1, ["b_node", "a_node"]].set_axis(["a_node", "b_node"], axis=1)
    arcs = pd.concat([forward, backward])
    tails = arcs.a_node.to_numpy()
    heads = arcs.b_node.to_numpy()

    outgoing = {}
    for tail, head in zip(tails, heads, strict=True):
        outgoing.setdefault(int(tail), []).append(int(head))

    centroids = set(map(int, centroids))
    for tail, via in zip(tails, heads, strict=True):
        tail, via = int(tail), int(via)
        if tail in centroids or via in centroids:
            continue
        for destination in outgoing.get(via, ()):
            if destination not in centroids and destination != tail:
                return pd.DataFrame(
                    {
                        "from_node": [tail],
                        "via_node": [via],
                        "to_node": [destination],
                        "penalty": [0.0],
                    }
                )
    raise ValueError("No road turn is available to enable turn-based skimming")


def run_skimming_benchmark(benchmark, network, model_stub, model_folder, routing_case, *, block_centroid_flows):
    """Time skimming alone; prepare turn restrictions outside the timed call."""
    restrictions = None
    if routing_case == "centroid_turns":
        restrictions = zero_cost_road_turn(network.links, network.centroids)
    elif routing_case == "left_turns":
        nodes = load_node_coordinates(model_folder, model_stub)
        restrictions = left_turn_restrictions(network, nodes, block_centroid_flows=block_centroid_flows)
        assert len(restrictions) >= MIN_LEFT_TURNS[model_stub], f"Too few left turns for {model_stub}"
    elif routing_case != "node_based":
        raise ValueError(f"Unknown skimming case: {routing_case}")

    graph = network.build_graph(
        "c",
        "free_flow_time",
        skimming=["free_flow_time"],
        compact=True,
        remove_dead_ends=True,
        block_centroid_flows=block_centroid_flows,
        turn_restrictions=restrictions,
    )

    # Skimming uses the compact context. Check the setup before timing it.
    context = graph.routing
    penalties = context.turn_penalties if isinstance(context, TurnBasedContext) else np.array([])

    if routing_case == "node_based":
        assert isinstance(context, NodeBasedContext)
        assert context.blocked_centroid_count == (graph.num_zones if block_centroid_flows else 0)
        assert penalties.size == 0
    else:
        assert isinstance(context, TurnBasedContext)
        assert context.blocked_centroid_count == 0

        if routing_case == "centroid_turns":
            assert np.count_nonzero(penalties == 0) == 1
            centroid_bans = MIN_CENTROID_BANS[model_stub] if block_centroid_flows else 0
            assert np.count_nonzero(np.isinf(penalties)) >= centroid_bans
            assert penalties.size == np.count_nonzero(np.isinf(penalties)) + 1
        else:
            assert np.isinf(penalties).all()
            centroid_bans = MIN_CENTROID_BANS[model_stub] if block_centroid_flows else 0
            assert penalties.size >= MIN_LEFT_TURNS[model_stub] + centroid_bans

    skim = NetworkSkimming(graph)
    benchmark(skim.execute)
