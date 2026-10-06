import numpy as np
import pandas as pd
import pytest

from aequilibrae.paths import Graph


def test_effective_via_node_protected_from_chain_compression():
    graph = Graph()
    graph.network = pd.DataFrame(
        [(1, 1, 2, 1, 1.0), (2, 2, 3, 1, 1.0), (3, 3, 4, 1, 1.0)],
        columns=["link_id", "a_node", "b_node", "direction", "cost"],
    )
    graph.set_turn_restrictions(pd.DataFrame({"from_node": [1], "via_node": [2], "to_node": [3], "penalty": [5.0]}))
    graph.prepare_graph(np.array([1, 4]), remove_dead_ends=False)
    graph.set_graph("cost")
    graph.set_skimming(["cost"])

    assert 2 in graph.compact_all_nodes
    assert graph.compact_num_links < graph.num_links
    assert graph.compute_path(1, 4).milepost[-1] == pytest.approx(8.0)
    assert graph.compute_skims().results.skims.matrix["cost"][0, 1] == pytest.approx(8.0)


def test_cycle_preservation_under_chain_compression():
    graph = Graph()
    graph.network = pd.DataFrame(
        [(1, 1, 2, 1, 1.0), (2, 2, 3, 1, 1.0), (3, 3, 5, 1, 1.0),
         (4, 5, 2, 1, 1.0), (5, 2, 4, 1, 1.0)],
        columns=["link_id", "a_node", "b_node", "direction", "cost"],
    )
    graph.set_turn_restrictions(pd.DataFrame({"from_node": [1], "via_node": [2], "to_node": [4], "penalty": [np.inf]}))
    graph.prepare_graph(np.array([1, 4]), remove_dead_ends=False)
    graph.set_graph("cost")
    graph.set_skimming(["cost"])

    result = graph.compute_path(1, 4)
    np.testing.assert_array_equal(result.path_nodes, [1, 2, 3, 5, 2, 4])
    assert result.milepost[-1] == pytest.approx(5.0)
    assert graph.compute_skims().results.skims.matrix["cost"][0, 1] == pytest.approx(5.0)
