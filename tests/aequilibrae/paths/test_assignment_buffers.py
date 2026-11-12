"""Assignment loads and skims remain meaningful across graph reductions."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from aequilibrae.matrix import from_dict
from aequilibrae.paths import Network, TrafficClass
from aequilibrae.paths.assignment_context import AssignmentInputs, PreparedTrafficClass


@pytest.fixture
def network():
    links = pd.DataFrame(
        {
            "link_id": [9, 2, 7, 5],
            "a_node": [10, 20, 20, 100],
            "b_node": [20, 30, 40, 101],
            "direction": [0, 0, 0, 1],
            "modes": ["c", "c", "c", "w"],
            "time": [1.0, 2.0, 3.0, 4.0],
            "distance": [5.0, 6.0, 7.0, 8.0],
        }
    )
    return Network(links, centroids=[30, 10])


@pytest.fixture
def demand():
    # Both columns contain trips from node 10 to node 30.
    return from_dict(
        {"work": np.array([[0.0, 0.0], [8.0, 0.0]]), "other": np.array([[0.0, 0.0], [12.0, 0.0]])},
        [30, 10],
    )


@pytest.fixture
def make_assignment(network, demand):
    def make(*, compact=True, turn=False, sparse_demand=False, fixed_cost=0.0):
        source = network
        if turn:
            turns = pd.DataFrame({"from_node": [10], "via_node": [20], "to_node": [30], "penalty": [0.5]})
            source = Network(network.links, centroids=network.centroids, turn_restrictions=turns)
        matrix = demand
        if sparse_demand:
            matrix = from_dict({name: sparse.csr_array(values) for name, values in demand.items()}, demand.index)
        configuration = TrafficClass(
            mode="c",
            cost="time",
            demand=matrix,
            graph_options={"compact": compact, "remove_dead_ends": False},
            select_link_sets={"kept": [(9, 1)], "dead": [(7, 0)]},
        )
        fields = source._field_arrays(["time", "distance"])
        prepared = PreparedTrafficClass(configuration, source, fields)
        inputs = AssignmentInputs(
            prepared.graph,
            matrix,
            "time",
            prepared._selected_links,
            1,
            field_values=fields.__getitem__,
            skim_fields=["time", "distance", "assignment_cost"],
            cost_name="assignment_cost",
        )
        prepared.fixed_cost.fill(fixed_cost)
        inputs.update_costs(fields["time"], prepared.fixed_cost)
        return prepared, inputs, inputs.make_state(1, -1)

    return make


@pytest.mark.parametrize(
    "compact,sparse_demand",
    [(False, False), (True, False), (True, True)],
    ids=["uncontracted", "contracted", "sparse-demand"],
)
def test_loads_are_projected_to_network_order(network, make_assignment, compact, sparse_demand):
    _, inputs, state = make_assignment(compact=compact, sparse_demand=sparse_demand)
    inputs.driver.run(state.output)
    state.update_totals()

    expected = np.zeros((network.mapping.link_count, 2))
    expected[network.mapping.link_index([9, 2], 1)] = [8.0, 12.0]
    np.testing.assert_array_equal(inputs.mapping.project_loads(state.output.loading.link_loads), expected)
    np.testing.assert_array_equal(state.total_link_loads, expected.sum(axis=1))


def test_selected_links_match_only_paths_that_use_them(make_assignment):
    _, inputs, state = make_assignment()
    inputs.driver.run(state.output)
    selected = state.output.select_link.loading.loads

    np.testing.assert_array_equal(selected["kept"], state.output.loading.link_loads)
    assert not np.any(selected["dead"])


@pytest.mark.parametrize("turn", [False, True])
def test_skims_separate_travel_time_fixed_costs_and_turn_delay(make_assignment, turn):
    _, inputs, state = make_assignment(turn=turn, fixed_cost=0.25)
    inputs.driver.run(state.output)
    state.update_totals()
    skims = state.skims.matrices

    travel_time = 3.0 + (0.5 if turn else 0.0)
    assert skims["time"][1, 0] == travel_time
    assert skims["distance"][1, 0] == 11.0
    assert skims["assignment_cost"][1, 0] == travel_time + 0.5
    assert state.output.turn_cost_total == (10.0 if turn else 0.0)
