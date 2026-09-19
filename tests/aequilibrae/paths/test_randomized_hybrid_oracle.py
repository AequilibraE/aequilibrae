"""Randomized small-network parity between the hybrid kernel, the arc-based kernel and a Python oracle."""

from __future__ import annotations

import heapq
import random
import numpy as np
import pandas as pd
import pytest

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph
from aequilibrae.paths.all_or_nothing import allOrNothing
from aequilibrae.paths.cython.basic_path_finding import path_finding_hybrid
from aequilibrae.paths.network_skimming import NetworkSkimming
from aequilibrae.paths.results import AssignmentResults


class ArcStateDijkstraOracle:
    """Independent pure-Python arc-state Dijkstra oracle."""

    def __init__(self, links_df: pd.DataFrame, turns_df: pd.DataFrame, allow_uturns: bool = False):
        self.allow_uturns = allow_uturns
        self.arcs = []  # arc_id -> (u, v, cost, link_id)
        self.outgoing = {}

        for row in links_df.itertuples(index=False):
            u, v = int(row.a_node), int(row.b_node)
            cost = float(row.free_flow_time)
            lid = int(row.link_id)
            aid = len(self.arcs)
            self.arcs.append((u, v, cost, lid))
            self.outgoing.setdefault(u, []).append(aid)

        self.turn_lookup = {}
        for row in turns_df.itertuples(index=False):
            key = (int(row.from_node), int(row.via_node), int(row.to_node))
            pen = float(row.penalty) if pd.notnull(row.penalty) else np.inf
            self.turn_lookup[key] = pen

    def shortest_path(self, origin: int, destination: int) -> tuple[float, list[int], list[int]] | None:
        """Returns (cost, node_sequence, link_id_sequence) or None."""
        if origin == destination:
            return 0.0, [origin], []

        pq = []
        best_cost = {}

        for aid in self.outgoing.get(origin, []):
            u, v, c, lid = self.arcs[aid]
            best_cost[aid] = c
            heapq.heappush(pq, (c, aid, [u, v], [lid]))

        best_dest_cost = float("inf")
        best_dest_nodes = None
        best_dest_links = None

        while pq:
            cost, cur_arc, nodes, links = heapq.heappop(pq)
            if cost > best_cost.get(cur_arc, float("inf")):
                continue

            u, v, _, _ = self.arcs[cur_arc]
            if v == destination:
                if cost < best_dest_cost:
                    best_dest_cost = cost
                    best_dest_nodes = nodes
                    best_dest_links = links
                continue

            for next_arc in self.outgoing.get(v, []):
                _, w, next_c, next_lid = self.arcs[next_arc]
                if not self.allow_uturns and w == u:
                    continue

                pen = self.turn_lookup.get((u, v, w), 0.0)
                if np.isinf(pen):
                    continue

                new_cost = cost + next_c + pen
                if new_cost < best_cost.get(next_arc, float("inf")):
                    best_cost[next_arc] = new_cost
                    heapq.heappush(pq, (new_cost, next_arc, nodes + [w], links + [next_lid]))

        if best_dest_nodes is None:
            return None
        return best_dest_cost, best_dest_nodes, best_dest_links


def _generate_random_network(seed: int, num_nodes: int = 8, num_links: int = 18):
    rng = random.Random(seed)
    nodes = list(range(1, num_nodes + 1))

    # Ensure a basic chain/cycle for connectivity
    edge_set = set()
    links = []
    link_id = 1

    # Base chain
    for i in range(num_nodes - 1):
        u, v = nodes[i], nodes[i + 1]
        edge_set.add((u, v))
        cost = round(rng.uniform(2.0, 15.0), 1)
        links.append(
            {"link_id": link_id, "a_node": u, "b_node": v, "direction": 1, "distance": cost, "free_flow_time": cost}
        )
        link_id += 1

    # Additional random directed edges
    while len(links) < num_links:
        u = rng.choice(nodes)
        v = rng.choice(nodes)
        if u != v and (u, v) not in edge_set:
            edge_set.add((u, v))
            cost = round(rng.uniform(2.0, 20.0), 1)
            links.append(
                {"link_id": link_id, "a_node": u, "b_node": v, "direction": 1, "distance": cost, "free_flow_time": cost}
            )
            link_id += 1

    links_df = pd.DataFrame(links)
    links_df["modes"] = "c"
    links_df["link_type"] = "road"

    # Find valid 2-hop movements
    two_hops = []
    for u, v in edge_set:
        for v2, w in edge_set:
            if v == v2 and u != w:
                two_hops.append((u, v, w))

    # Sample turn restrictions: some finite, some prohibited
    turns = []
    if two_hops:
        sample_size = min(len(two_hops), rng.randint(2, 6))
        sampled = rng.sample(two_hops, sample_size)
        for i, (u, v, w) in enumerate(sampled):
            if i % 2 == 0:
                pen = round(rng.uniform(1.0, 15.0), 1)
            else:
                pen = np.inf
            turns.append({"from_node": u, "via_node": v, "to_node": w, "penalty": pen})

    turns_df = pd.DataFrame(turns) if turns else pd.DataFrame(columns=["from_node", "via_node", "to_node", "penalty"])
    return links_df, turns_df, nodes



def _prepared_graph(seed: int, centroids=None, num_nodes: int = 8, num_links: int = 18):
    """Builds a prepared, turn-restricted Graph from one randomized network."""
    links_df, turns_df, nodes = _generate_random_network(seed, num_nodes=num_nodes, num_links=num_links)
    graph = Graph()
    graph.network = links_df.copy()
    if not turns_df.empty:
        graph.set_turn_restrictions(turns_df, allow_path_uturns=False)
    graph.prepare_graph(centroids=centroids, remove_dead_ends=False)
    graph.set_graph("free_flow_time")
    return graph, links_df, turns_df, nodes


def _assign(graph, use_hybrid: bool, cores: int = 1):
    """Runs an all-or-nothing assignment on `graph` with the requested kernel."""
    graph.set_hybrid_kernel(use_hybrid)
    mat = AequilibraeMatrix()
    mat.create_empty(file_name=AequilibraeMatrix().random_name(), zones=len(graph.centroids), matrix_names=["matrix"])
    mat.index[:] = graph.centroids[:]
    mat.computational_view(core_list=["matrix"])
    mat.matrix_view[:, :] = 1.0

    res = AssignmentResults()
    res.cores = cores
    res.prepare(graph, mat)
    allOrNothing("car", mat, graph, res).execute()
    return np.array(res.link_loads, copy=True)


@pytest.mark.parametrize("seed", list(range(20)))
def test_randomized_small_network_oracle_parity(seed: int):
    """Property test verifying 100% parity between Python oracle, arc-based kernel, and hybrid kernel."""
    links_df, turns_df, nodes = _generate_random_network(seed, num_nodes=7, num_links=15)
    oracle = ArcStateDijkstraOracle(links_df, turns_df, allow_uturns=False)

    g = Graph()
    g.network = links_df.copy()
    if not turns_df.empty:
        g.set_turn_restrictions(turns_df, allow_path_uturns=False)
    g.prepare_graph(centroids=None, remove_dead_ends=False)
    g.set_graph("free_flow_time")

    # Turn penalty lookup for independent auditing
    turn_lookup = {}
    if not turns_df.empty:
        for row in turns_df.itertuples(index=False):
            key = (int(row.from_node), int(row.via_node), int(row.to_node))
            turn_lookup[key] = float(row.penalty) if pd.notnull(row.penalty) else np.inf

    link_costs = {int(r.link_id): float(r.free_flow_time) for r in links_df.itertuples(index=False)}

    # Test all pairs of nodes
    for orig in nodes:
        for dest in nodes:
            if orig == dest:
                continue

            oracle_res = oracle.shortest_path(orig, dest)

            # Arc-based Dijkstra
            g.set_hybrid_kernel(False)
            arc_res = g.compute_path(orig, dest)

            # Hybrid kernel
            g.set_hybrid_kernel(True)
            hybrid_res = g.compute_path(orig, dest)

            if oracle_res is None:
                # Must be unreachable in both
                assert arc_res.path is None, f"Seed {seed}: {orig} -> {dest} expected unreachable in arc-based"
                assert hybrid_res.path is None, f"Seed {seed}: {orig} -> {dest} expected unreachable in hybrid"
            else:
                expected_cost, _, _ = oracle_res

                assert arc_res.path is not None, f"Seed {seed}: {orig} -> {dest} unreachable in arc-based"
                assert hybrid_res.path is not None, f"Seed {seed}: {orig} -> {dest} unreachable in hybrid"

                # Cost parity
                assert arc_res.milepost[-1] == pytest.approx(expected_cost, abs=1e-5), f"Seed {seed}: {orig} -> {dest}"
                assert hybrid_res.milepost[-1] == pytest.approx(expected_cost, abs=1e-5), (
                    f"Seed {seed}: {orig} -> {dest}"
                )

                # Independent audit of hybrid path
                h_nodes = [int(n) for n in hybrid_res.path_nodes]
                h_links = [int(lid) for lid in hybrid_res.path]
                assert h_nodes[0] == orig
                assert h_nodes[-1] == dest
                assert len(h_nodes) == len(h_links) + 1

                recomputed_cost = 0.0
                for i, lid in enumerate(h_links):
                    recomputed_cost += link_costs[lid]
                    if i > 0:
                        pen = turn_lookup.get((h_nodes[i - 1], h_nodes[i], h_nodes[i + 1]), 0.0)
                        assert not np.isinf(pen), (
                            f"Prohibited turn in path {h_nodes[i - 1]}->{h_nodes[i]}->{h_nodes[i + 1]}"
                        )
                        recomputed_cost += pen

                assert recomputed_cost == pytest.approx(expected_cost, abs=1e-5)


def test_hybrid_settled_label_efficiency_and_early_exit():
    """Verifies the hybrid kernel settles at most one label per plain node, and that early exit settles fewer."""
    # Big enough that the restricted intersections are a small minority of the network,
    # which is the regime the collapse is supposed to pay off in.
    graph, _, _, nodes = _prepared_graph(seed=3, num_nodes=30, num_links=90)
    assert graph.has_turn_restrictions

    num_nodes = graph.num_nodes
    num_arcs = graph.num_links
    csr_indices = graph.graph["b_node"].to_numpy(np.int64, copy=False)
    a_nodes = graph.graph["a_node"].to_numpy(np.int64, copy=False)
    graph_costs = graph.cost.astype(np.float64)
    graph_fs = graph.fs.astype(np.int64)

    origin_idx = graph.nodes_to_indices[nodes[0]]
    dest_idx = graph.nodes_to_indices[nodes[-1]]

    def run(destinations, destination_count, stateful):
        settled = np.zeros(1, dtype=np.int64)
        path_finding_hybrid(
            origin_idx,
            destinations,
            destination_count,
            graph_costs,
            csr_indices,
            graph_fs,
            a_nodes,
            stateful,
            graph.rep_arc,
            np.full(num_nodes, -1, dtype=np.int64),
            np.full(num_nodes, -1, dtype=np.int64),
            np.zeros(num_nodes, dtype=np.int64),
            np.full(num_nodes, np.inf, dtype=np.float64),
            np.zeros(num_nodes, dtype=np.float64),
            np.full(num_arcs, -1, dtype=np.int64),
            np.zeros(num_arcs, dtype=np.float64),
            graph.turn_fs,
            graph.turn_to_arcs,
            graph.turn_penalties,
            False,
            False,
            0,
            csr_indices,
            a_nodes,
            settled,
        )
        return int(settled[0])

    single_destination = np.zeros(num_nodes, dtype=np.uint8)
    single_destination[dest_idx] = 1

    settled_full = run(np.zeros(0, dtype=np.uint8), -1, graph.stateful)
    settled_early = run(single_destination, 1, graph.stateful)
    # Marking every node stateful disables the collapse, which is exactly the arc-based kernel.
    settled_arc = run(np.zeros(0, dtype=np.uint8), -1, np.ones_like(graph.stateful))

    assert settled_full > 0
    assert settled_early > 0
    assert settled_early <= settled_full
    assert settled_full < settled_arc, "the collapse settled no fewer labels than full arc state"

    # The design claim is that only stateful nodes hold more than one live label: every other
    # node collapses onto its representative arc. A plain arc-based kernel would settle up to
    # num_arcs labels, so bound the hybrid by the label space it is supposed to occupy.
    in_degree = np.bincount(csr_indices[:num_arcs], minlength=num_nodes)[:num_nodes]
    label_budget = int(np.where(graph.stateful[:num_nodes].astype(bool), np.maximum(in_degree, 1), 1).sum())
    assert settled_full <= label_budget, (
        f"hybrid settled {settled_full} labels but only {label_budget} are reachable under the state-collapse rule"
    )


@pytest.mark.parametrize("seed", list(range(5)))
def test_skimming_oracle_parity(seed: int):
    """Verifies turn-aware skimming matches the Python oracle, not only the arc-based kernel."""
    centroids = np.array([1, 2, 3, 4], dtype=np.int64)
    graph, links_df, turns_df, _ = _prepared_graph(seed, centroids=centroids)
    # The oracle prices movements only; it has no notion of centroid blocking.
    graph.set_blocked_centroid_flows(False)
    graph.set_skimming("free_flow_time")
    oracle = ArcStateDijkstraOracle(links_df, turns_df, allow_uturns=False)

    graph.set_hybrid_kernel(True)
    skm_hybrid = NetworkSkimming(graph)
    skm_hybrid.execute()
    mat_hybrid = np.array(skm_hybrid.results.skims.free_flow_time[:, :], copy=True)
    index = np.array(skm_hybrid.results.skims.index[:], copy=True)

    graph.set_hybrid_kernel(False)
    skm_arc = NetworkSkimming(graph)
    skm_arc.execute()
    mat_arc = np.array(skm_arc.results.skims.free_flow_time[:, :], copy=True)

    # The two kernels must agree with each other ...
    np.testing.assert_allclose(mat_hybrid, mat_arc, equal_nan=True)

    # ... and with an independent oracle, which is the part that makes this a parity test.
    for i, origin in enumerate(index):
        for j, dest in enumerate(index):
            if origin == dest:
                continue
            expected = oracle.shortest_path(int(origin), int(dest))
            if expected is None:
                assert not np.isfinite(mat_hybrid[i, j]) or mat_hybrid[i, j] == 0.0
                continue
            assert mat_hybrid[i, j] == pytest.approx(expected[0], abs=1e-5), f"seed {seed}: {origin} -> {dest}"


@pytest.mark.parametrize("seed", list(range(5)))
def test_assignment_link_load_parity_between_kernels(seed: int):
    """Verifies all-or-nothing link loads are identical under the hybrid and arc-based kernels."""
    graph, _, _, _ = _prepared_graph(seed, centroids=np.array([1, 2, 3, 4], dtype=np.int64))
    assert graph.has_turn_restrictions

    loads_hybrid = _assign(graph, use_hybrid=True)
    loads_arc = _assign(graph, use_hybrid=False)

    assert loads_hybrid.sum() > 0
    np.testing.assert_allclose(loads_hybrid, loads_arc)


def test_assignment_link_loads_are_thread_count_invariant():
    """Verifies a multi-threaded assignment produces the same link loads as a single-threaded run."""
    graph, _, _, _ = _prepared_graph(seed=1, centroids=np.array([1, 2, 3, 4], dtype=np.int64))

    single = _assign(graph, use_hybrid=True, cores=1)
    pooled = _assign(graph, use_hybrid=True, cores=4)

    assert single.sum() > 0
    np.testing.assert_allclose(single, pooled)
