from random import randint
from sqlite3 import IntegrityError

import pandas as pd
import pytest

from aequilibrae.utils.db_utils import read_and_close


def test_get(sioux_falls_example):
    periods = sioux_falls_example.network.periods
    for num in range(2, 6):
        periods.insert(period_id=num, period_start=num, period_end=num, period_description="test")

    nd = randint(2, 5)
    period = periods.get(nd)
    assert period.period_id == nd, "get period returned wrong object"

    periods.renumber(nd, 200)
    with pytest.raises(ValueError, match=rf"periods has no record with period_id={nd}"):
        _ = periods.get(nd)


def test_fields(sioux_falls_example):
    periods = sioux_falls_example.network.periods
    f_editor = periods.fields
    fields = sorted(f_editor.all_fields())

    with read_and_close(sioux_falls_example.path_to_file) as conn:
        dt = conn.execute("pragma table_info(periods)").fetchall()
    actual_fields = sorted({x[1] for x in dt})
    assert fields == actual_fields, "Table editor is weird for table periods"


def test_new_period_and_default_period(sioux_falls_example):
    periods = sioux_falls_example.network.periods
    assert periods.default_period.period_id == 1

    period_id = periods.new_period(2, start=7 * 3600, end=9 * 3600, description="Morning peak")
    period = periods.get(period_id)
    assert period.period_start == 7 * 3600
    assert period.period_end == 9 * 3600
    assert period.period_description == "Morning peak"


def test_update(sioux_falls_example):
    periods = sioux_falls_example.network.periods
    with pytest.raises(IntegrityError, match="Cannot update default period"):
        periods.update(1, period_description="whole day")

    periods.insert(period_id=2, period_start=0, period_end=3600, period_description="morning")
    periods.update(2, period_description="morning peak")
    assert periods.get(2).period_description == "morning peak"


def test_save(sioux_falls_example):
    periods = sioux_falls_example.network.periods
    for num in range(2, 6):
        periods.insert(period_id=num, period_start=num, period_end=num, period_description="test")

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
