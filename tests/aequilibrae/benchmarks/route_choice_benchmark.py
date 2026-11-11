"""Route generation and results-table benchmarks on TNTP networks.

Run with ``TNTP_ROOT`` set and ``--benchmark-only -k route_choice``. Both phases
include table construction; the table phase excludes generation and PSL computation.
"""

import hashlib

import numpy as np
import pandas as pd
from aequilibrae.paths.cython.context import NodeBasedContext, TurnBasedContext
from aequilibrae.paths.cython.route_choice_set import RouteChoiceSet
from aequilibrae.utils.cython.bridge import Bridge

from aequilibrae.paths import estimate_heuristic_scale
from aequilibrae.paths.routing_context import make_routing_context

from .path_finding_benchmark import OD_RANDOM_SEED, left_turn_restrictions, load_node_coordinates
from .skimming_benchmark import MIN_CENTROID_BANS, MIN_LEFT_TURNS, zero_cost_road_turn

ROUTE_CHOICE_ROUTING_CASES = ("node_based", "centroid_turns", "left_turns")
OD_COUNT = 100
MAX_ROUTES = 5
CORES = 1


def make_route_choice_ods(graph):
    """Sample distinct centroid pairs so endpoints survive graph compression."""
    rng = np.random.default_rng(OD_RANDOM_SEED)
    pairs = []
    seen = set()
    while len(pairs) < OD_COUNT:
        origin, destination = rng.choice(graph.centroids, size=2, replace=False)
        pair = (int(origin), int(destination))
        if pair not in seen:
            pairs.append(pair)
            seen.add(pair)
    return pairs


def run_route_choice_benchmark(
    benchmark, graph, model_stub, model_folder, algorithm, routing_case, a_star, *, table_only=False
):
    """Time normal batch generation or rebuilding its uncached results table."""
    original_blocking = graph.block_centroid_flows
    graph.clear_turn_restrictions()
    graph.set_blocked_centroid_flows(True)
    try:
        nodes = load_node_coordinates(model_folder, model_stub)
        assert nodes is not None, f"No coordinates available for {model_stub}"
        if routing_case == "centroid_turns":
            graph.set_turn_restrictions(zero_cost_road_turn(graph))
        elif routing_case == "left_turns":
            restrictions = left_turn_restrictions(graph, nodes)
            assert len(restrictions) >= MIN_LEFT_TURNS[model_stub]
            graph.set_turn_restrictions(restrictions)
        elif routing_case != "node_based":
            raise ValueError(f"Unknown route choice case: {routing_case}")

        context = make_routing_context(graph, compact=True)
        if routing_case == "node_based":
            assert isinstance(context, NodeBasedContext)
            assert context.blocked_centroid_count == graph.num_zones
        else:
            assert isinstance(context, TurnBasedContext)
            assert context.blocked_centroid_count == 0
            assert np.count_nonzero(np.isinf(graph.compact_turn_penalties)) >= MIN_CENTROID_BANS[model_stub]

        coordinates = nodes.set_index("node_id")[["longitude", "latitude"]].rename(
            columns={"longitude": "x", "latitude": "y"}
        )

        scale = estimate_heuristic_scale(graph, coordinates) if a_star else None
        if a_star and scale is not None and scale > 0:
            raise ValueError("A* benchmark requires a non-zero heuristic")

        choice = RouteChoiceSet(graph, coordinates=coordinates)
        ods = make_route_choice_ods(graph)

        options = {
            "nodes": ods,
            "bfsle": algorithm == "bfsle",
            "max_routes": MAX_ROUTES,
            "max_depth": 5 if algorithm == "bfsle" else 100,
            "max_misses": 100,
            "penalty": 1.0 if algorithm == "bfsle" else 1.1,
            "seed": OD_RANDOM_SEED,
            "cores": CORES,
            "a_star": a_star,
            "heuristic_scale": scale,
            "store_results": True,
            "path_size_logit": True,
        }
        benchmark.group = "route_choice_table" if table_only else "route_choice_generation"

        benchmark.extra_info.update(
            model=model_stub,
            phase="table" if table_only else "generation",
            algorithm=algorithm,
            routing_case=routing_case,
            search="a_star" if a_star else "dijkstra",
            od_count=OD_COUNT,
            max_routes=MAX_ROUTES,
            max_depth=options["max_depth"],
            max_misses=options["max_misses"],
            penalty=options["penalty"],
            seed=OD_RANDOM_SEED,
            cores=CORES,
            heuristic_scale=scale,
        )

        with Bridge() as bridge:

            def generate():
                choice.batched(bridge=bridge, **options)
                return choice.get_results()

            if table_only:
                reference = generate()
                table = benchmark(choice.get_results, rebuild=True)
                pd.testing.assert_frame_equal(table, reference)
                assert table is not reference
            else:
                table = benchmark(generate)

        assert not table.empty
        counts = table.groupby(["origin id", "destination id"]).size()
        assert set(counts.index) == set(ods), "Some sampled OD pairs have no routes"
        assert counts.between(1, MAX_ROUTES).all()
        assert all(isinstance(route, np.ndarray) and route.dtype == np.int64 for route in table["route set"])
        assert np.isfinite(table[["cost", "path overlap", "probability"]].to_numpy()).all()
        np.testing.assert_allclose(table.groupby(["origin id", "destination id"])["probability"].sum(), 1.0)
        normalised = table.assign(**{"route set": table["route set"].map(tuple)}).sort_values(
            ["origin id", "destination id", "route set"]
        )
        digest = hashlib.sha256(pd.util.hash_pandas_object(normalised, index=False).to_numpy().tobytes()).hexdigest()
        benchmark.extra_info.update(
            result_digest=digest,
            route_count=len(table),
            link_count=sum(map(len, table["route set"])),
            ods_with_five_routes=int((counts == MAX_ROUTES).sum()),
        )
    finally:
        graph.clear_turn_restrictions()
        graph.set_blocked_centroid_flows(original_blocking)
