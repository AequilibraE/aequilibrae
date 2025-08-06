from random import randint

import pandas as pd
import pytest


def add_extra_test_periods(project):
    for num in range(2, 6):
        project.network.periods.new_period(num, num, num, "test")

    return project


def test_save_and_assignment(sioux_falls_example):
    project = add_extra_test_periods(sioux_falls_example)
    periods = project.network.periods
    nd = randint(2, 5)
    period = periods.get(nd)

    with pytest.raises(AttributeError):
        period.modes = "abc"
    with pytest.raises(AttributeError):
        period.link_types = "default"
    with pytest.raises(AttributeError):
        period.period_id = 2

    period.period_description = "test"
    assert period.period_description == "test"

    period.save()

    expected = pd.DataFrame(
        {
            "period_id": [1, nd],
            "period_start": [0, nd],
            "period_end": [86400, nd],
        }
    )
    expected["period_description"] = "test"
    expected.at[0, "period_description"] = "Default time period, whole day"

    pd.testing.assert_frame_equal(periods.data, expected)


def test_data_fields(sioux_falls_example):
    periods = sioux_falls_example.network.periods
    period = periods.get(1)
    fields = sorted(period.data_fields())
    with sioux_falls_example.db_connection as conn:
        dt = conn.execute("pragma table_info(periods)").fetchall()
    actual_fields = sorted([x[1] for x in dt if x[1] != "ogc_fid"])
    assert fields == actual_fields, "Period has unexpected set of fields"


def test_renumber(sioux_falls_example):
    periods = sioux_falls_example.network.periods
    period = periods.get(1)
    with pytest.raises(ValueError):
        period.renumber(1)
    num = randint(25, 2000)
    with pytest.raises(ValueError):
        period.renumber(num)
    new_period = periods.new_period(num, 0, 0, "test")
    new_period.renumber(num + 1)
    assert new_period.period_id == num + 1
