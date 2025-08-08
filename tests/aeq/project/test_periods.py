from copy import copy, deepcopy
from random import randint

import pandas as pd
import pytest

from aequilibrae.utils.db_utils import read_and_close


def test_get(sioux_falls_example):
    periods = sioux_falls_example.network.periods
    for num in range(2, 6):
        sioux_falls_example.network.periods.new_period(num, num, num, "test")

    nd = randint(2, 5)
    period = periods.get(nd)
    assert period.period_id == nd, "get period returned wrong object"

    period.renumber(200)
    with pytest.raises(ValueError):
        _ = periods.get(nd)


def test_fields(sioux_falls_example):
    periods = sioux_falls_example.network.periods
    f_editor = periods.fields
    fields = sorted(f_editor.all_fields())

    with read_and_close(sioux_falls_example.path_to_file) as conn:
        dt = conn.execute("pragma table_info(periods)").fetchall()
    actual_fields = sorted({x[1] for x in dt})
    assert fields == actual_fields, "Table editor is weird for table periods"


def test_copy(sioux_falls_example):
    periods = sioux_falls_example.network.periods
    with pytest.raises(Exception):
        _ = copy(periods)
    with pytest.raises(Exception):
        _ = deepcopy(periods)


def test_save(sioux_falls_example):
    periods = sioux_falls_example.network.periods
    for num in range(2, 6):
        sioux_falls_example.network.periods.new_period(num, num, num, "test")

    periods.save()

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
