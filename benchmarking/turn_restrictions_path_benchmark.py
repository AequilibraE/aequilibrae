import argparse
import heapq
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, TypedDict

import networkx as nx
import numpy as np
import pandas as pd
from aequilibrae import Project


class ArcInfo(TypedDict):
    a_node: int
    b_node: int
    link_id: int
    direction: int
    distance: float


ArcMeta = Dict[int, ArcInfo]
DirectedTurn = Tuple[int, int, int]
DirectedLink = Tuple[int, int]


def _sample_od_pairs(centroids: np.ndarray, sample_size: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    if centroids.shape[0] < 2:
        raise ValueError("Need at least two centroids to sample OD pairs")

    origins = rng.choice(centroids, size=sample_size, replace=True)
    destinations = rng.choice(centroids, size=sample_size, replace=True)

    same = origins == destinations
    while np.any(same):
        destinations[same] = rng.choice(centroids, size=int(np.sum(same)), replace=True)
        same = origins == destinations

    return [(int(o), int(d)) for o, d in zip(origins, destinations)]


def _prepare_graph(model_path: Path, mode: str):
    project = Project.from_path(model_path)
    project.network.build_graphs(modes=[mode])
    graph = project.network.graphs[mode]

    centroids = np.asarray(graph.centroids, dtype=np.int64)
    if centroids.size == 0:
        raise ValueError("Model has no centroids in this graph")

    graph.set_graph("distance")
    graph.set_blocked_centroid_flows(False)
    graph.set_skimming(["distance"])
    return project, graph, centroids, "distance"


def _build_arc_index(graph):
    graph_df = graph.graph[["id", "a_node", "b_node", "link_id", "direction", "distance"]].copy()
    graph_df["a_node"] = graph.all_nodes[graph_df["a_node"].to_numpy(dtype=np.int64)]
    graph_df["b_node"] = graph.all_nodes[graph_df["b_node"].to_numpy(dtype=np.int64)]

    arc_lookup: Dict[Tuple[int, int, int], Tuple[int, int, float]] = {}
    arc_meta: ArcMeta = {}
    outgoing: Dict[int, List[int]] = defaultdict(list)
    incoming: Dict[int, List[int]] = defaultdict(list)

    for row in graph_df.itertuples(index=False):
        arc_id = int(row.id)
        a_node = int(row.a_node)
        b_node = int(row.b_node)
        link_id = int(row.link_id)
        direction = int(row.direction)
        distance = float(getattr(row, "distance"))

        arc_lookup[(a_node, b_node, link_id)] = (arc_id, direction, distance)
        arc_meta[arc_id] = {
            "a_node": a_node,
            "b_node": b_node,
            "link_id": link_id,
            "direction": direction,
            "distance": distance,
        }
        outgoing[a_node].append(arc_id)
        incoming[b_node].append(arc_id)

    return graph_df, arc_lookup, arc_meta, outgoing, incoming

def _build_turn_bans(
        graph,
        od_pairs: Sequence[Tuple[int, int]],
        arc_lookup: Dict[Tuple[int, int, int], Tuple[int, int, float]],
        rng: np.random.Generator,
) -> Tuple[pd.DataFrame, int, int]:
    prohibited: set[DirectedTurn] = set()
    computed = 0
    valid_for_bans = 0

    for origin, destination in od_pairs:
        try:
            res = graph.compute_path(origin, destination)
        except Exception:
            continue

        directed, _ = _aeq_directed_path(res.path_nodes, res.path, arc_lookup)
        computed += 1

        if len(res.path_nodes) < 3:
            continue

        valid_for_bans += 1
        idx = int(rng.integers(0, len(res.path_nodes) - 2))
        from_node = int(res.path_nodes[idx])
        via_node = int(res.path_nodes[idx + 1])
        to_node = int(res.path_nodes[idx + 2])
        prohibited.add((from_node, via_node, to_node))

    tr = pd.DataFrame(
        {
            "from_node": [x[0] for x in prohibited],
            "via_node": [x[1] for x in prohibited],
            "to_node": [x[2] for x in prohibited],
            "penalty": [np.nan] * len(prohibited),
        }
    )

    return tr, computed, valid_for_bans


def _build_networkx_equivalent_digraph(graph_df: pd.DataFrame):
    g = nx.MultiDiGraph()
    a_nodes = graph_df["a_node"].to_numpy(dtype=np.int64)
    b_nodes = graph_df["b_node"].to_numpy(dtype=np.int64)
    ids = graph_df["id"].to_numpy(dtype=np.int64)
    link_ids = graph_df["link_id"].to_numpy(dtype=np.int64)
    directions = graph_df["direction"].to_numpy(dtype=np.int64)
    weights = graph_df["distance"].to_numpy(dtype=np.float64)

    for a_node, b_node, arc_id, link_id, direction, weight in zip(
            a_nodes, b_nodes, ids, link_ids, directions, weights
    ):
        g.add_edge(
            a_node,
            b_node,
            key=arc_id,
            weight=float(weight),
            link_id=link_id,
            direction=direction,
        )
    return g


def _build_state_transitions(
        arc_meta: ArcMeta,
        incoming: Dict[int, List[int]],
        outgoing: Dict[int, List[int]],
        penalty_lookup: Dict[DirectedTurn, float],
        prohibited: set[DirectedTurn],
        allow_uturns: bool,
) -> Dict[int, List[Tuple[int, float]]]:
    transitions: Dict[int, List[Tuple[int, float]]] = defaultdict(list)

    nodes = set(incoming.keys()) & set(outgoing.keys())
    for node in nodes:
        for in_arc in incoming[node]:
            in_meta = arc_meta[in_arc]
            for out_arc in outgoing[node]:
                out_meta = arc_meta[out_arc]

                if not allow_uturns:
                    if int(out_meta["b_node"]) == int(in_meta["a_node"]):
                        continue

                transition_key = (int(in_meta["a_node"]), int(in_meta["b_node"]), int(out_meta["b_node"]))

                if transition_key in prohibited:
                    continue

                penalty = penalty_lookup.get(transition_key, 0.0)
                if math.isnan(penalty):
                    continue

                add_cost = float(out_meta["distance"]) + max(0.0, penalty)
                transitions[in_arc].append((out_arc, add_cost))

    return transitions


def _state_shortest_path(
        origin: int,
        destination: int,
        arc_meta: ArcMeta,
        incoming: Dict[int, List[int]],
        outgoing: Dict[int, List[int]],
        transitions: Dict[int, List[Tuple[int, float]]],
) -> Optional[Tuple[List[DirectedLink], float]]:
    start_arcs = outgoing.get(origin, [])
    end_arcs = incoming.get(destination, [])

    if not start_arcs or not end_arcs:
        return None

    target = set(end_arcs)
    max_arc_id = max(arc_meta.keys())
    distances = np.full(max_arc_id + 1, np.inf, dtype=np.float64)
    predecessors = np.full(max_arc_id + 1, -1, dtype=np.int64)

    heap: List[Tuple[float, int]] = []
    for arc_id in start_arcs:
        start_cost = float(arc_meta[arc_id]["distance"])
        if start_cost < distances[arc_id]:
            distances[arc_id] = start_cost
            heapq.heappush(heap, (start_cost, arc_id))

    final_arc = -1
    while heap:
        current_cost, arc_id = heapq.heappop(heap)
        if current_cost > distances[arc_id]:
            continue

        if arc_id in target:
            final_arc = arc_id
            break

        for to_arc, transition_cost in transitions.get(arc_id, []):
            new_cost = current_cost + transition_cost
            if new_cost < distances[to_arc]:
                distances[to_arc] = new_cost
                predecessors[to_arc] = arc_id
                heapq.heappush(heap, (new_cost, to_arc))

    if final_arc < 0:
        return None

    arc_sequence = [final_arc]
    while predecessors[arc_sequence[-1]] >= 0:
        arc_sequence.append(int(predecessors[arc_sequence[-1]]))
    arc_sequence.reverse()

    directed = [(int(arc_meta[a]["link_id"]), int(arc_meta[a]["direction"])) for a in arc_sequence]
    return directed, float(distances[final_arc])


def run_benchmark(
        model_path: Path,
        mode: str,
        turn_ban_sample: int,
        comparison_sample: int,
        seed: int,
        allow_uturns: bool,
        output_csv: Optional[Path],
) -> None:
    rng = np.random.default_rng(seed)
    project, graph, centroids, chosen_cost = _prepare_graph(model_path, mode)

    try:
        graph_df, arc_lookup, arc_meta, outgoing, incoming = _build_arc_index(graph)

        print(f"Loaded model: {model_path}")
        print(f"Mode: {mode}")
        print(f"Centroids: {centroids.shape[0]}")
        print(f"Cost field: {chosen_cost}")

        od_pairs_for_bans = _sample_od_pairs(centroids, turn_ban_sample, rng)
        turn_bans_df, computed_paths, ban_candidates = _build_turn_bans(graph, od_pairs_for_bans, arc_lookup, rng)

        if turn_bans_df.empty:
            raise RuntimeError("Could not build any prohibited turn from sampled paths")

        print(f"Sampled OD for turn bans: {turn_ban_sample}")
        print(f"Computed paths for turn bans: {computed_paths}")
        print(f"Paths eligible for turn-ban selection: {ban_candidates}")
        print(f"Prohibited turns inserted: {len(turn_bans_df)}")

        graph.set_turn_restrictions(turn_bans_df, allow_path_uturns=allow_uturns)

        networkx_equivalent = _build_networkx_equivalent_digraph(graph_df)
        print(
            "networkx-equivalent digraph built "
            f"(nodes={networkx_equivalent.number_of_nodes()}, edges={networkx_equivalent.number_of_edges()})"
        )

        from_nodes = pd.to_numeric(turn_bans_df["from_node"], errors="coerce").to_numpy(dtype=np.int64)
        via_nodes = pd.to_numeric(turn_bans_df["via_node"], errors="coerce").to_numpy(dtype=np.int64)
        to_nodes = pd.to_numeric(turn_bans_df["to_node"], errors="coerce").to_numpy(dtype=np.int64)
        penalties = pd.to_numeric(turn_bans_df["penalty"], errors="coerce").to_numpy(dtype=np.float64)

        penalty_lookup: Dict[DirectedTurn, float] = {}
        for from_node, via_node, to_node, penalty in zip(from_nodes, via_nodes, to_nodes, penalties):
            penalty_lookup[(int(from_node), int(via_node), int(to_node))] = float(penalty)
        prohibited = {key for key, penalty in penalty_lookup.items() if math.isnan(penalty) or penalty < 0}

        transitions = _build_state_transitions(
            arc_meta=arc_meta,
            incoming=incoming,
            outgoing=outgoing,
            penalty_lookup=penalty_lookup,
            prohibited=prohibited,
            allow_path_uturns=allow_uturns,
        )

        od_pairs_for_compare = _sample_od_pairs(centroids, comparison_sample, rng)
        rows: List[Dict[str, object]] = []

        for origin, destination in od_pairs_for_compare:
            status = "ok"
            aeq_path: Optional[List[DirectedLink]] = None
            aeq_cost: Optional[float] = None
            nx_path: Optional[List[DirectedLink]] = None
            nx_cost: Optional[float] = None

            try:
                aeq_res = graph.compute_path(origin, destination)
                aeq_path, aeq_cost = _aeq_directed_path(aeq_res.path_nodes, aeq_res.path, arc_lookup)
            except Exception:
                status = "aeq_failed"

            nx_res = _state_shortest_path(
                origin=origin,
                destination=destination,
                arc_meta=arc_meta,
                incoming=incoming,
                outgoing=outgoing,
                transitions=transitions,
            )
            if nx_res is None:
                if status == "ok":
                    status = "nx_unreachable"
            else:
                nx_path, nx_cost = nx_res

            if aeq_path is None and nx_path is not None:
                status = "aeq_missing"
            if aeq_path is not None and nx_path is None:
                status = "nx_missing"

            path_match = bool(aeq_path == nx_path) if aeq_path is not None and nx_path is not None else False
            cost_match = (
                bool(np.isclose(aeq_cost, nx_cost, rtol=1e-8, atol=1e-8))
                if aeq_cost is not None and nx_cost is not None
                else False
            )

            rows.append(
                {
                    "origin": origin,
                    "destination": destination,
                    "status": status,
                    "aeq_cost": aeq_cost,
                    "nx_cost": nx_cost,
                    "cost_abs_diff": None if aeq_cost is None or nx_cost is None else abs(aeq_cost - nx_cost),
                    "path_match": path_match,
                    "cost_match": cost_match,
                    "aeq_links": None if aeq_path is None else len(aeq_path),
                    "nx_links": None if nx_path is None else len(nx_path),
                }
            )

        df = pd.DataFrame(rows)

        ok_rows = df[df["status"] == "ok"]
        print(f"OD pairs compared: {len(df)}")
        print(f"Rows with successful paths on both engines: {len(ok_rows)}")
        print(f"Exact path matches: {int(ok_rows['path_match'].sum())}/{len(ok_rows)}")
        print(f"Cost matches (1e-8): {int(ok_rows['cost_match'].sum())}/{len(ok_rows)}")

        if len(ok_rows) > 0:
            print(f"Average absolute cost difference: {ok_rows['cost_abs_diff'].mean():.12f}")

        if output_csv is not None:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False)
            print(f"Wrote detailed comparison to: {output_csv}")

    finally:
        project.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark AequilibraE vs networkx-equivalent digraph pathfinding under generated turn restrictions"
    )
    parser.add_argument("--model-path", type=Path, required=True, help="Path to the AequilibraE model/project")
    parser.add_argument("--mode", type=str, default="c", help="Network mode to build graph for")
    parser.add_argument(
        "--turn-ban-sample",
        type=int,
        default=100,
        help="Number of centroid OD pairs used to generate turn bans",
    )
    parser.add_argument(
        "--comparison-sample",
        type=int,
        default=1000,
        help="Number of centroid OD pairs used for AequilibraE vs networkx-equivalent comparison",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--allow-uturns", action="store_true", help="Allow U-turns in both engines")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("benchmarking/results/turn_restrictions_path_benchmark.csv"),
        help="Path to write per-OD comparison results",
    )

    args = parser.parse_args()
    run_benchmark(
        model_path=args.model_path,
        mode=args.mode,
        turn_ban_sample=args.turn_ban_sample,
        comparison_sample=args.comparison_sample,
        seed=args.seed,
        allow_uturns=args.allow_uturns,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()




def _aeq_directed_path(
        path_nodes: Sequence[int], path_links: Sequence[int],
        arc_lookup: Dict[Tuple[int, int, int], Tuple[int, int, float]]
) -> Tuple[List[DirectedLink], float]:
    directed: List[DirectedLink] = []
    total_cost = 0.0

    for a_node, b_node, link_id in zip(path_nodes[:-1], path_nodes[1:], path_links):
        key = (int(a_node), int(b_node), int(link_id))
        _, direction, cost = arc_lookup[key]
        directed.append((int(link_id), int(direction)))
        total_cost += cost

    return directed, total_cost


