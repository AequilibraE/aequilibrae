import sqlite3
from dataclasses import FrozenInstanceError

import geopandas as gpd
import pandas as pd
import pytest


@pytest.fixture
def turns(sioux_falls_example):
    return sioux_falls_example.network.turn_restrictions


@pytest.fixture
def movements(sioux_falls_example):
    with sioux_falls_example.db_connection as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT l1.a_node, l1.b_node, l2.b_node
            FROM links l1 JOIN links l2 ON l1.b_node = l2.a_node
            WHERE l1.direction >= 0 AND l2.direction >= 0
            ORDER BY l1.a_node, l1.b_node, l2.b_node
            LIMIT 2
            """
        ).fetchall()
    assert len(rows) == 2
    return [dict(zip(("from_node", "via_node", "to_node"), row, strict=True)) for row in rows]


def test_table_reads_and_scalar_writes(turns, movements):
    key = turns.insert(restriction_id=42, **movements[0], penalty=8, modes="c")
    record = turns.get(key)

    assert key == 42
    assert key in turns
    assert len(turns) == 1
    assert record.modes == "c"
    assert record.penalty == 8
    assert record.geometry.geom_type == "LineString"
    assert list(turns) == [record]
    assert isinstance(turns.data, gpd.GeoDataFrame)
    assert turns.data.iloc[0].geometry.equals(record.geometry)
    with pytest.raises(FrozenInstanceError):
        record.penalty = 5

    turns.update(key, penalty=5)
    assert turns.get(key).penalty == 5
    assert record.penalty == 8

    turns.delete(key)
    assert key not in turns
    assert turns.get(key, default=None) is None
    for operation in (lambda: turns.get(key), lambda: turns.update(key, penalty=1), lambda: turns.delete(key)):
        with pytest.raises(ValueError, match="has no record"):
            operation()


@pytest.mark.parametrize("penalty", [None, float("nan"), pd.NA, float("inf"), 0, 12.5])
@pytest.mark.parametrize("operation", ["insert", "update", "insert_from", "update_from"])
def test_penalty_normalization(turns, movements, penalty, operation):
    if operation in ("update", "update_from"):
        key = turns.insert(**movements[0], penalty=8, modes="c")
        if operation == "update":
            turns.update(key, penalty=penalty)
        else:
            turns.update_from(pd.DataFrame({"restriction_id": [key], "penalty": [penalty]}))
    elif operation == "insert":
        key = turns.insert(**movements[0], penalty=penalty, modes="c")
    else:
        (key,) = turns.insert_from(pd.DataFrame([{**movements[0], "penalty": penalty, "modes": "c"}]))

    stored = turns.get(key).penalty
    if penalty is None or pd.isna(penalty) or penalty == float("inf"):
        assert stored is None
    else:
        assert stored == penalty


@pytest.mark.parametrize("operation", ["insert_from", "update_from"])
@pytest.mark.parametrize(
    "dtype, penalties, expected",
    [
        (object, ["3.5", None], [3.5, None]),
        (object, ["nan", 0], [None, 0.0]),
        ("string", ["3.5", "inf"], [3.5, None]),
        ("Float64", [pd.NA, 2], [None, 2.0]),
    ],
)
def test_bulk_penalty_conversion_preserves_input(turns, movements, operation, dtype, penalties, expected):
    frame = pd.DataFrame(movements, index=[4, 9]).assign(modes="c")
    frame["penalty"] = pd.Series(penalties, index=frame.index, dtype=dtype)
    if operation == "update_from":
        keys = turns.insert_from(pd.DataFrame(movements).assign(modes="c", penalty=8))
        frame = frame.assign(restriction_id=keys)[["restriction_id", "penalty"]]
    original = frame.copy(deep=True)

    if operation == "insert_from":
        keys = turns.insert_from(frame)
    else:
        turns.update_from(frame)

    pd.testing.assert_frame_equal(frame, original)
    assert [turns.get(key).penalty for key in keys] == expected


@pytest.mark.parametrize(
    "values, error",
    [
        ({"penalty": -1}, ValueError),
        ({"penalty": -float("inf")}, ValueError),
        ({"penalty": "bad"}, ValueError),
        ({"modes": "?"}, sqlite3.IntegrityError),
        ({"modes": ""}, sqlite3.IntegrityError),
        ({"modes": "cc"}, sqlite3.IntegrityError),
        ({"modes": None}, sqlite3.IntegrityError),
        ({"modes": float("nan")}, sqlite3.IntegrityError),
        ({"modes": 1}, sqlite3.IntegrityError),
    ],
)
@pytest.mark.parametrize("operation", ["insert", "update", "insert_from", "update_from"])
def test_invalid_values_leave_table_unchanged(turns, movements, values, error, operation):
    key = turns.insert(**movements[0], penalty=8, modes="c")
    original = turns.get(key)

    with pytest.raises(error):
        if operation == "insert":
            turns.insert(**(movements[1] | {"modes": "c"} | values))
        elif operation == "insert_from":
            turns.insert_from(pd.DataFrame([movements[1] | {"modes": "c"} | values]))
        elif operation == "update":
            turns.update(key, **values)
        else:
            turns.update_from(pd.DataFrame([{"restriction_id": key} | values]))

    assert list(turns) == [original]


@pytest.mark.parametrize("operation", ["insert", "insert_from"])
def test_insert_requires_modes(turns, movements, operation):
    with pytest.raises(sqlite3.IntegrityError):
        if operation == "insert":
            turns.insert(**movements[0])
        else:
            turns.insert_from(pd.DataFrame(movements))
    assert len(turns) == 0


def test_mode_order_is_preserved(sioux_falls_example, turns, movements):
    sioux_falls_example.network.modes.insert(mode_id="x", mode_name="test mode x")
    key = turns.insert(**movements[0], modes="xc")
    assert turns.get(key).modes == "xc"

    turns.update(key, modes="cx")
    assert turns.get(key).modes == "cx"
    turns.update(key, penalty=1)
    assert turns.get(key).modes == "cx"

    turns.delete(key)
    frame = pd.DataFrame(movements).assign(modes=["xc", "cx"])
    original = frame.copy(deep=True)
    keys = turns.insert_from(frame)
    pd.testing.assert_frame_equal(frame, original)
    assert [turns.get(key).modes for key in keys] == ["xc", "cx"]

    turns.update_from(pd.DataFrame({"restriction_id": keys, "modes": ["cx", "xc"]}))
    assert [turns.get(key).modes for key in keys] == ["cx", "xc"]


def test_bulk_defaults_and_updates_do_not_change_frames(turns, movements):
    frame = pd.DataFrame(movements).assign(modes="c")
    original = frame.copy(deep=True)
    keys = turns.insert_from(frame)
    pd.testing.assert_frame_equal(frame, original)
    assert keys == [1, 2]
    assert all(turns.get(key).modes == "c" and turns.get(key).penalty is None for key in keys)

    updates = pd.DataFrame({"restriction_id": keys, "penalty": [float("inf"), 5]})
    original = updates.copy(deep=True)
    assert turns.update_from(updates) == 2
    pd.testing.assert_frame_equal(updates, original)
    assert turns.get(keys[0]).penalty is None
    assert turns.get(keys[1]).penalty == 5
    assert all(turns.get(key).modes == "c" for key in keys)


def test_bulk_insert_failure_keeps_outer_transaction(sioux_falls_example, turns, movements):
    with sioux_falls_example.transaction():
        key = turns.insert(**movements[0], modes="c")
        with pytest.raises(sqlite3.IntegrityError):
            # The second row overlaps an existing turn; the first must roll back too.
            turns.insert_from(pd.DataFrame([movements[1], movements[0]]).assign(modes="c"))
        assert [record.restriction_id for record in turns] == [key]
    assert key in turns


def test_bulk_update_failure_rolls_back_earlier_rows(turns, movements):
    keys = turns.insert_from(pd.DataFrame(movements).assign(modes="c"))
    original = list(turns)
    updates = pd.DataFrame(
        [
            {"restriction_id": keys[0], "penalty": 5, "from_node": movements[0]["from_node"]},
            {"restriction_id": keys[1], "penalty": 8, "from_node": movements[1]["via_node"]},
        ]
    )
    with pytest.raises(sqlite3.IntegrityError):
        turns.update_from(updates)
    assert list(turns) == original


def test_writes_and_clear_follow_project_transaction(sioux_falls_example, turns, movements):
    key = turns.insert(**movements[0], penalty=8, modes="c")
    original = turns.get(key)

    with pytest.raises(RuntimeError, match="discard"):
        with sioux_falls_example.transaction():
            turns.update(key, penalty=None)
            turns.insert(**movements[1], modes="c")
            assert turns.clear_restrictions() == 2
            assert len(turns) == 0
            raise RuntimeError("discard")

    assert list(turns) == [original]
    assert turns.clear_restrictions() == 1
    assert turns.clear_restrictions() == 0


def test_bulk_delete(turns, movements):
    keys = turns.insert_from(pd.DataFrame(movements).assign(modes="c"))
    original = keys.copy()

    with pytest.raises(ValueError, match="keys which do not exist"):
        turns.delete_from([keys[0], max(keys) + 1])
    assert len(turns) == 2

    assert turns.delete_from(keys) == 2
    assert keys == original
    assert len(turns) == 0
    assert turns.delete_from(keys, allow_missing=True) == 2


def test_user_fields(turns, movements):
    turns.fields.add("source", "Where the restriction came from", "TEXT")
    key = turns.insert(**movements[0], modes="c", source="survey")
    assert turns.get(key).source == "survey"
    assert "source" in turns.columns
    assert turns.data.iloc[0].source == "survey"
