"""Test suite verifying original-link loads, select-link results, and turn totals against an independent oracle."""

from __future__ import annotations

import heapq
import numpy as np
import pandas as pd
import pytest

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph, TrafficAssignment, TrafficClass
from aequilibrae.paths.vdf import bpr


class TurnAssignmentOracle:
    """Independent pure-Python arc-state Dijkstra oracle for traffic assignment."""

    def __init__(self, links_df: pd.DataFrame, turns_df: pd.DataFrame | None = None, allow_uturns: bool = False):
        self.links_df = links_df.copy()
        self.turns_df = (
            turns_df.copy()
            if turns_df is not None
            else pd.DataFrame(columns=["from_node", "via_node", "to_node", "penalty"])
        )
        self.allow_uturns = allow_uturns

        # Precompute turn penalty lookup
        self.turn_lookup = {}
        for row in self.turns_df.itertuples(index=False):
            key = (int(row.from_node), int(row.via_node), int(row.to_node))
            pen = float(row.penalty) if pd.notnull(row.penalty) else np.inf
            if key in self.turn_lookup:
                if np.isinf(pen) or np.isinf(self.turn_lookup[key]):
                    self.turn_lookup[key] = np.inf
                else:
                    self.turn_lookup[key] = pen
            else:
                self.turn_lookup[key] = pen

        # Directed arcs: arc_id -> (u, v, cost, original_link_id)
        self.arcs = []
        self.outgoing = {}  # node -> list of arc_ids
        for row in self.links_df.itertuples(index=False):
            u, v = int(row.a_node), int(row.b_node)
            cost = float(row.free_flow_time if hasattr(row, "free_flow_time") else row.distance)
            lid = int(row.link_id)
            direc = int(row.direction) if hasattr(row, "direction") else 1

            if direc in (1, 0):
                aid = len(self.arcs)
                self.arcs.append((u, v, cost, lid))
                self.outgoing.setdefault(u, []).append(aid)
            if direc in (-1, 0):
                aid = len(self.arcs)
                self.arcs.append((v, u, cost, lid))
                self.outgoing.setdefault(v, []).append(aid)

    def shortest_path(self, origin: int, destination: int) -> tuple[list[int], float, float] | None:
        """Finds shortest path returning (link_id_sequence, total_cost, total_turn_penalties)."""
        if origin == destination:
            return [], 0.0, 0.0

        # State: (cost, current_arc_id, turn_penalty_sum, path_arc_ids)
        pq = []
        best_cost = {}

        # Initial arcs from origin
        for arc_id in self.outgoing.get(origin, []):
            u, v, cost, lid = self.arcs[arc_id]
            heapq.heappush(pq, (cost, arc_id, 0.0, [arc_id]))
            best_cost[arc_id] = cost

        best_dest_path = None
        best_dest_cost = float("inf")
        best_dest_turn_pen = 0.0

        while pq:
            cost, cur_arc, turn_pen_sum, path = heapq.heappop(pq)
            if cost > best_cost.get(cur_arc, float("inf")):
                continue

            u, v, _, cur_lid = self.arcs[cur_arc]
            if v == destination:
                if cost < best_dest_cost:
                    best_dest_cost = cost
                    best_dest_path = path
                    best_dest_turn_pen = turn_pen_sum
                continue

            for next_arc in self.outgoing.get(v, []):
                _, w, next_cost, next_lid = self.arcs[next_arc]

                # U-turn check
                if not self.allow_uturns and w == u:
                    continue

                # Turn penalty
                movement = (u, v, w)
                pen = self.turn_lookup.get(movement, 0.0)
                if np.isinf(pen):
                    continue

                new_cost = cost + next_cost + pen
                if new_cost < best_cost.get(next_arc, float("inf")):
                    best_cost[next_arc] = new_cost
                    heapq.heappush(pq, (new_cost, next_arc, turn_pen_sum + pen, path + [next_arc]))

        if best_dest_path is None:
            return None

        link_ids = [self.arcs[a][3] for a in best_dest_path]
        return link_ids, best_dest_cost, best_dest_turn_pen

    def assign_all_or_nothing(
        self,
        centroids: list[int],
        demand_matrix: np.ndarray,
        select_links: list[int] | None = None,
    ) -> dict:
        """Executes all-or-nothing assignment and returns oracle loads, turn penalties, and select-link matrices."""
        num_links = len(self.links_df)
        link_id_to_idx = {int(lid): i for i, lid in enumerate(self.links_df.link_id)}

        link_loads = np.zeros(num_links, dtype=np.float64)
        select_link_loads = np.zeros(num_links, dtype=np.float64)
        select_link_od = np.zeros((len(centroids), len(centroids)), dtype=np.float64)
        total_turn_penalty = 0.0

        select_set = set(select_links) if select_links else set()

        for o_idx, o in enumerate(centroids):
            for d_idx, d in enumerate(centroids):
                dem = demand_matrix[o_idx, d_idx]
                if dem <= 0 or o == d:
                    continue

                res = self.shortest_path(o, d)
                if res is None:
                    continue

                path_links, _, turn_pens = res
                total_turn_penalty += dem * turn_pens

                path_select = bool(select_set.intersection(path_links))
                if path_select:
                    select_link_od[o_idx, d_idx] += dem

                for lid in path_links:
                    idx = link_id_to_idx[lid]
                    link_loads[idx] += dem
                    if path_select:
                        select_link_loads[idx] += dem

        return {
            "link_loads": link_loads,
            "total_turn_penalty": total_turn_penalty,
            "select_link_od": select_link_od,
            "select_link_loads": select_link_loads,
        }


def _make_test_network_and_demand():
    # Triangular diamond network:
    # 1 -> 2 (link 1, cost 10)
    # 2 -> 3 (link 2, cost 10)
    # 1 -> 4 (link 3, cost 12)
    # 4 -> 3 (link 4, cost 12)
    # 2 -> 4 (link 5, cost 3)
    links = [
        {
            "link_id": 1,
            "a_node": 1,
            "b_node": 2,
            "direction": 1,
            "distance": 10.0,
            "free_flow_time": 10.0,
            "capacity": 1000.0,
        },
        {
            "link_id": 2,
            "a_node": 2,
            "b_node": 3,
            "direction": 1,
            "distance": 10.0,
            "free_flow_time": 10.0,
            "capacity": 1000.0,
        },
        {
            "link_id": 3,
            "a_node": 1,
            "b_node": 4,
            "direction": 1,
            "distance": 12.0,
            "free_flow_time": 12.0,
            "capacity": 1000.0,
        },
        {
            "link_id": 4,
            "a_node": 4,
            "b_node": 3,
            "direction": 1,
            "distance": 12.0,
            "free_flow_time": 12.0,
            "capacity": 1000.0,
        },
        {
            "link_id": 5,
            "a_node": 2,
            "b_node": 4,
            "direction": 1,
            "distance": 3.0,
            "free_flow_time": 3.0,
            "capacity": 1000.0,
        },
    ]
    df = pd.DataFrame(links)
    df["modes"] = "c"
    df["link_type"] = "road"
    return df


@pytest.mark.parametrize("compress", [False, True])
def test_turn_assignment_oracle_parity_finite_penalties(compress: bool):
    """Verifies link loads, turn penalty totals, and select-link against pure-Python oracle."""
    net = _make_test_network_and_demand()
    # Path options from 1 to 3:
    # 1 -> 2 -> 3: links 1, 2 (cost 10 + 10 + 10 = 30)
    # 1 -> 4 -> 3: links 3, 4 (cost 12 + 12 + 5 = 29)
    # 1 -> 2 -> 4 -> 3: links 1, 5, 4 (cost 10 + 3 + 12 + 1 = 26)
    turns = pd.DataFrame(
        [
            {"from_node": 1, "via_node": 2, "to_node": 3, "penalty": 10.0},
            {"from_node": 1, "via_node": 4, "to_node": 3, "penalty": 5.0},
            {"from_node": 1, "via_node": 2, "to_node": 4, "penalty": 1.0},
        ]
    )
    centroids = np.array([1, 3], dtype=np.int64)
    demand = np.zeros((2, 2), dtype=np.float64)
    demand[0, 1] = 100.0  # 1 -> 3

    # Run Independent Oracle
    oracle = TurnAssignmentOracle(net, turns, allow_uturns=False)
    oracle_res = oracle.assign_all_or_nothing(list(centroids), demand, select_links=[5])
    # The cheapest route is 1 -> 2 -> 4 -> 3, which pays the 1.0 penalty on 1 -> 2 -> 4 once per
    # unit of demand. Pinning it here keeps the oracle itself honest, not just the two in step.
    assert oracle_res["total_turn_penalty"] == pytest.approx(100.0)

    # Run AequilibraE Assignment
    g = Graph()
    g.network = net.copy()
    g.set_turn_restrictions(turns, allow_path_uturns=False)
    g.prepare_graph(centroids=centroids, remove_dead_ends=compress)
    g.set_graph("free_flow_time")

    mat = AequilibraeMatrix()
    mat.create_empty(memory_only=True, zones=len(centroids), matrix_names=["demand"])
    mat.index[:] = centroids
    mat.computational_view(core_list=["demand"])
    mat.matrix_view[:, :] = demand

    tc = TrafficClass("car", g, mat)
    tc.set_select_links({"sel_link_5": [(5, 1)]})

    assig = TrafficAssignment()
    assig.set_classes([tc])
    assig.set_vdf(bpr, {"alpha": 0.15, "beta": 4.0})
    assig.set_capacity_field("capacity")
    assig.set_time_field("free_flow_time")
    assig.set_algorithm("all-or-nothing")
    assig.execute()

    assigned_tc = assig.classes[0]

    # Compare total turn penalty (100.0)
    assert assigned_tc.results.total_turn_penalty == pytest.approx(oracle_res["total_turn_penalty"])

    # Compare original link loads:
    aeq_loads = assigned_tc.results.link_loads.flatten()
    supernet_ids = g.graph.__supernet_id__.to_numpy(copy=False)
    reconstructed_loads = np.zeros(len(net), dtype=np.float64)
    for row_idx, sup_id in enumerate(supernet_ids):
        lid = g.graph.iloc[row_idx]["link_id"]
        net_idx = int(np.flatnonzero(net.link_id == lid)[0])
        reconstructed_loads[net_idx] = aeq_loads[sup_id]

    np.testing.assert_allclose(reconstructed_loads, oracle_res["link_loads"], rtol=1e-5)

    # Compare select-link OD matrix
    sel_matrix = assigned_tc.results.select_link_od.matrix["sel_link_5"]
    np.testing.assert_allclose(np.squeeze(sel_matrix), oracle_res["select_link_od"], rtol=1e-5)

    # Compare select-link loading, which the OD matrix above does not exercise: it is written by a
    # second backtrack over the same path and can be wrong while the OD matrix is right.
    sel_loading = np.asarray(assigned_tc.results.select_link_loading["sel_link_5"]).reshape(-1)
    reconstructed_sel = np.zeros(len(net), dtype=np.float64)
    for row_idx, sup_id in enumerate(supernet_ids):
        lid = g.graph.iloc[row_idx]["link_id"]
        net_idx = int(np.flatnonzero(net.link_id == lid)[0])
        reconstructed_sel[net_idx] = sel_loading[sup_id]

    np.testing.assert_allclose(reconstructed_sel, oracle_res["select_link_loads"], rtol=1e-5)
