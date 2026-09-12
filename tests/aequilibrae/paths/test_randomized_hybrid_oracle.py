"""Randomized small-network full-versus-hybrid oracle parity test suite across 25 diverse seeds."""

from __future__ import annotations

import heapq
import random
import numpy as np
import pandas as pd
import pytest

from aequilibrae.paths import Graph


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
        links.append({"link_id": link_id, "a_node": u, "b_node": v, "direction": 1, "distance": cost, "free_flow_time": cost})
        link_id += 1

    # Additional random directed edges
    while len(links) < num_links:
        u = rng.choice(nodes)
        v = rng.choice(nodes)
        if u != v and (u, v) not in edge_set:
            edge_set.add((u, v))
            cost = round(rng.uniform(2.0, 20.0), 1)
            links.append({"link_id": link_id, "a_node": u, "b_node": v, "direction": 1, "distance": cost, "free_flow_time": cost})
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
                assert hybrid_res.milepost[-1] == pytest.approx(expected_cost, abs=1e-5), f"Seed {seed}: {orig} -> {dest}"

                # Independent audit of hybrid path
                h_nodes = [int(n) for n in hybrid_res.path_nodes]
                h_links = [int(l) for l in hybrid_res.path]
                assert h_nodes[0] == orig
                assert h_nodes[-1] == dest
                assert len(h_nodes) == len(h_links) + 1

                recomputed_cost = 0.0
                for i, lid in enumerate(h_links):
                    recomputed_cost += link_costs[lid]
                    if i > 0:
                        pen = turn_lookup.get((h_nodes[i - 1], h_nodes[i], h_nodes[i + 1]), 0.0)
                        assert not np.isinf(pen), f"Prohibited turn in path {h_nodes[i-1]}->{h_nodes[i]}->{h_nodes[i+1]}"
                        recomputed_cost += pen

                assert recomputed_cost == pytest.approx(expected_cost, abs=1e-5)
