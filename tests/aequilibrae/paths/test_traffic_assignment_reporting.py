"""Assignment reporting follows the shared network, not a class's reduced graph."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from aequilibrae.matrix import from_dict
from aequilibrae.routing import Network
from aequilibrae.traffic import TrafficAssignment, TrafficClass
from aequilibrae.traffic.vdf import bpr


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
    centroids = [30, 10, 100, 101]
    work, other, walk_values = [np.zeros((4, 4)) for _ in range(3)]
    work[0, 1], work[1, 0] = 2.0, 8.0
    other[0, 1], other[1, 0] = 3.0, 12.0
    walk_values[2, 3] = 3.0

    cars = from_dict({"work": work, "other": other}, centroids)
    walk = from_dict({"walk": walk_values}, centroids)
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


@pytest.fixture
def reporting_configuration(sioux_falls_example):
    network = Network(
        pd.DataFrame(
            {"link_id": [1], "a_node": [10], "b_node": [20], "direction": [0], "time": [1.0], "capacity": [100.0]}
        ),
        centroids=[10, 20],
    )
    values = np.array([[0.0, 5.0], [3.0, 0.0]])
    classes = [
        TrafficClass(mode="c", name=name, cost="time", demand=from_dict({core: values}, network.centroids))
        for name, core in (("cars", "commute"), ("trucks", "freight"))
    ]
    return {
        "network": network,
        "classes": classes,
        "time_field": "time",
        "capacity_field": "capacity",
        "vdf": bpr,
        "algorithm": "all-or-nothing",
        "project": sioux_falls_example,
    }


def test_combined_results_keep_demand_names_and_save(reporting_configuration):
    assignment = TrafficAssignment(**reporting_configuration)
    frame = assignment.execute(cores=1, elementwise_cores=1)
    assert frame.columns.is_unique
    assert frame.columns[:6].tolist() == [
        "commute_ab",
        "commute_ba",
        "commute_tot",
        "freight_ab",
        "freight_ba",
        "freight_tot",
    ]
    for results in assignment.class_results.values():
        class_frame = results.get_load_results()
        pd.testing.assert_frame_equal(frame[class_frame.columns], class_frame.rename_axis("link_id"))
    record = assignment.save_results("unique_core_results")
    stored = assignment.project.results.get_results("unique_core_results")
    pd.testing.assert_frame_equal(stored, frame.reset_index(), check_dtype=False)
    assert isinstance(json.loads(record.procedure_report), dict)


@pytest.mark.parametrize("names", [("flow", "flow"), ("flow", "FLOW"), ("PCE", "other"), ("Preload", "other")])
def test_conflicting_result_names_rejected_before_execution(reporting_configuration, names):
    traffic = reporting_configuration["classes"][0]
    classes = [
        replace(traffic, name=class_name, demand=from_dict({core: traffic.demand["commute"]}, traffic.demand.index))
        for class_name, core in zip(("cars", "trucks"), names, strict=True)
    ]
    with pytest.raises(ValueError, match="unique result columns"):
        TrafficAssignment(**(reporting_configuration | {"classes": classes}))


def test_execution_rechecks_result_name_conflicts(reporting_configuration):
    assignment = TrafficAssignment(**reporting_configuration)
    demand = assignment.classes[1].demand
    demand["commute"] = demand["freight"]
    with pytest.raises(ValueError, match="unique result columns"):
        assignment.execute(cores=1)


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
