import numpy as np
import pandas as pd
import pytest
from aequilibrae.utils.cython.openmp_helper import omp_get_max_threads

from aequilibrae.matrix import from_dict
from aequilibrae.paths import Network
from aequilibrae.paths.assignment_context import AssignmentInputs
from aequilibrae.paths.results import AssignmentResults


def test_set_cores():
    a = AssignmentResults()
    a.set_cores(10)

    with pytest.raises(ValueError):
        a.set_cores(1.3)

    a.set_cores(-2)
    assert a.cores == max(1, omp_get_max_threads() - 2)


def test_set_save_path_file():
    a = AssignmentResults()

    # Never save by default
    assert a.save_path_file is False


@pytest.fixture
def make_results():
    links = pd.DataFrame(
        {
            "link_id": [9, 2, 7, 5],
            "a_node": [10, 30, 20, 100],
            "b_node": [20, 20, 40, 101],
            "direction": [0, 0, 0, 1],
            "modes": ["c", "c", "c", "w"],
            "time": [1.0, 2.0, 3.0, 4.0],
        }
    )
    network = Network(links, centroids=[30, 10])
    demand = from_dict(
        {"work": np.array([[0.0, 2.0], [8.0, 0.0]]), "other": np.array([[0.0, 3.0], [12.0, 0.0]])},
        [30, 10],
    )

    def make(selections=None):
        graph = network.build_graph("c", cost="time", compact=True, remove_dead_ends=True)
        inputs = AssignmentInputs(
            graph,
            demand,
            "time",
            graph.selected_links(selections or {}),
            1,
            field_values=network.field_values,
        )
        inputs.update_costs(network.field_values("time"), np.zeros(network.mapping.link_count))
        state = inputs.make_state(1, -1)
        inputs.driver.run(state.output)
        state.update_totals()

        results = AssignmentResults()
        results.bind(state, inputs.class_names)
        return results

    return make


def test_load_table_preserves_source_order_and_directions(make_results):
    results = make_results()
    expected = pd.DataFrame(
        [[8, 2, 10, 12, 3, 15], [2, 8, 10, 3, 12, 15], [0] * 6, [0] * 6],
        columns=["work_ab", "work_ba", "work_tot", "other_ab", "other_ba", "other_tot"],
        index=pd.Index([9, 2, 7, 5], dtype=np.uint64),
        dtype=float,
    )
    pd.testing.assert_frame_equal(results.get_load_results(), expected)


def test_network_load_snapshot_is_read_only_and_survives_reset(make_results):
    results = make_results()
    snapshot = results.link_loads
    expected = np.array([[8, 12], [2, 3], [2, 3], [8, 12], [0, 0], [0, 0], [0, 0]], dtype=float)
    np.testing.assert_array_equal(snapshot, expected)
    assert not snapshot.flags.writeable

    results.reset()

    assert not np.any(results.link_loads)
    np.testing.assert_array_equal(snapshot, expected)


def test_selected_load_table_expands_to_network_links(make_results):
    results = make_results({"screen": [(9, 1)]})
    expected = pd.DataFrame(
        [[8, 0, 8, 12, 0, 12], [0, 8, 8, 0, 12, 12], [0] * 6, [0] * 6],
        columns=[
            "screen_work_ab",
            "screen_work_ba",
            "screen_work_tot",
            "screen_other_ab",
            "screen_other_ba",
            "screen_other_tot",
        ],
        index=pd.Index([9, 2, 7, 5], dtype=np.uint64),
        dtype=float,
    )
    pd.testing.assert_frame_equal(results.get_sl_results(), expected)


def test_empty_selected_load_table_keeps_source_order(make_results):
    table = make_results().get_sl_results()
    assert table.columns.empty
    assert table.index.tolist() == [9, 2, 7, 5]
