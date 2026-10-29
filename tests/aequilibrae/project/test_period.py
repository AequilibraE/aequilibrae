from random import randint

import pandas as pd
import pytest


def add_extra_test_periods(project):
    for num in range(2, 6):
        project.network.periods.insert(period_id=num, period_start=num, period_end=num, period_description="test")

    return project


def test_save_and_assignment(sioux_falls_example):
    project = add_extra_test_periods(sioux_falls_example)
    periods = project.network.periods
    nd = randint(2, 5)
    assert "modes" not in periods.columns
    assert "link_types" not in periods.columns
    assert periods.key == "period_id"
    periods.update(nd, period_description="test")

    expected = pd.DataFrame(
        {
            "period_id": [1, 2, 3, 4, 5],
            "period_start": [0, 2, 3, 4, 5],
            "period_end": [86400, 2, 3, 4, 5],
        }
    )
    expected["period_description"] = "test"
    expected.at[0, "period_description"] = "Default time period, whole day"

    pd.testing.assert_frame_equal(periods.data, expected)


def test_data_fields(sioux_falls_example):
    periods = sioux_falls_example.network.periods
    fields = sorted(periods.columns)
    with sioux_falls_example.db_connection as conn:
        dt = conn.execute("pragma table_info(periods)").fetchall()
    actual_fields = sorted([x[1] for x in dt if x[1] != "ogc_fid"])
    assert fields == actual_fields, "Period has unexpected set of fields"


def test_renumber(sioux_falls_example):
    periods = sioux_falls_example.network.periods
    period = periods.get(1)
    with pytest.raises(ValueError, match="You cannot renumber"):
        periods.renumber(period.period_id, 1)
    num = randint(25, 2000)
    with pytest.raises(ValueError, match="You cannot renumber"):
        periods.renumber(period.period_id, num)
    periods.insert(period_id=num, period_start=0, period_end=0, period_description="test")
    periods.renumber(num, num + 1)
    assert periods.get(num + 1).period_id == num + 1
