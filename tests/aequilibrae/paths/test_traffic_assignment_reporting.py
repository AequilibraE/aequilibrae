"""Assignment reporting follows the shared network, not a class's reduced graph."""

import numpy as np
import pandas as pd
import pytest

from aequilibrae.matrix import from_dict
from aequilibrae.paths import Network, TrafficAssignment, TrafficClass
from aequilibrae.paths.vdf import bpr


@pytest.fixture
def make_assignment():
    network = Network(
        pd.DataFrame(
            {
                "link_id": [9, 2, 7, 5],
                "a_node": [10, 30, 20, 100],
                "b_node": [20, 20, 40, 101],
                "direction": [0, 0, 0, 1],
                "modes": ["c", "c", "c", "w"],
                "time_ab": [1.0, 5.0, 3.0, 7.0],
                "time_ba": [4.0, 2.0, 6.0, 8.0],
                "capacity_ab": 100.0,
                "capacity_ba": 200.0,
            }
        )
    )
    cars = from_dict(
        {"work": np.array([[0.0, 2.0], [8.0, 0.0]]), "other": np.array([[0.0, 3.0], [12.0, 0.0]])}, [30, 10]
    )
    walk = from_dict({"walk": np.array([[0.0, 3.0], [0.0, 0.0]])}, [100, 101])
    classes = [
        TrafficClass(mode="c", cost="time", demand=cars, pce=2.0),
        TrafficClass(mode="w", cost="time", demand=walk),
    ]

    def make(preload=None):
        assignment = TrafficAssignment(
            network=network,
            classes=classes,
            time_field="time",
            capacity_field="capacity",
            vdf=bpr,
            name_mapping={"alpha": 0.0, "beta": 1.0},
            algorithm="all-or-nothing",
            preload=preload,
        )
        assignment.execute(cores=1, elementwise_cores=1)
        return assignment

    return make


def test_result_tables_keep_source_row_order(make_assignment):
    assignment = make_assignment()
    assert assignment.results().index.tolist() == [9, 2, 7, 5]
    for results in assignment.class_results.values():
        assert results.get_load_results().index.tolist() == [9, 2, 7, 5]


def test_network_metrics_include_links_outside_the_first_class(make_assignment):
    table = make_assignment().results()
    walk = table.loc[5]
    assert walk["PCE_AB"] == walk["PCE_tot"] == 3.0
    assert walk["Congested_Time_AB"] == walk["Congested_Time_Max"] == 7.0
    assert walk["Delay_factor_AB"] == walk["Delay_factor_Max"] == 1.0
    assert walk["VOC_AB"] == walk["VOC_max"] == pytest.approx(0.03)
    assert not walk[["PCE_BA", "Congested_Time_BA", "Delay_factor_BA", "VOC_BA"]].any()


def test_missing_preload_reports_zero(make_assignment):
    table = make_assignment().results()
    assert not table[["Preload_AB", "Preload_BA", "Preload_tot"]].to_numpy().any()


@pytest.mark.parametrize("as_frame", [False, True], ids=["array", "dataframe"])
def test_preload_and_pce_follow_network_order(make_assignment, as_frame):
    preload = np.arange(1, 8, dtype=float)
    if as_frame:
        preload = pd.DataFrame(
            {"link_id": [9, 9, 2, 2, 7, 7, 5], "direction": [1, -1, 1, -1, 1, -1, 1], "preload": preload}
        ).iloc[::-1]
    table = make_assignment(preload).results()

    np.testing.assert_array_equal(
        table[["Preload_AB", "Preload_BA", "Preload_tot"]], [[1, 2, 3], [3, 4, 7], [5, 6, 11], [7, 0, 7]]
    )
    np.testing.assert_array_equal(
        table[["PCE_AB", "PCE_BA", "PCE_tot"]], [[41, 12, 53], [13, 44, 57], [5, 6, 11], [10, 0, 10]]
    )
    assert table.loc[9, "VOC_AB"] == pytest.approx(0.41)
    assert table.loc[9, "VOC_BA"] == pytest.approx(0.06)
    assert table.loc[2, "Congested_Time_AB"] == 5.0
    assert table.loc[2, "Congested_Time_BA"] == 2.0
