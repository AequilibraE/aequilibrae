import pandas as pd
import pytest

from aequilibrae.project.network.modes import Modes
from aequilibrae.utils.db_utils import NestedTransactionManager


@pytest.fixture
def modes():
    manager = NestedTransactionManager(":memory:")
    manager._connection.execute(
        """CREATE TABLE modes (
            mode_name TEXT UNIQUE NOT NULL,
            mode_id TEXT PRIMARY KEY CHECK (length(mode_id) = 1),
            description TEXT,
            pce NUMERIC NOT NULL DEFAULT 1,
            vot NUMERIC NOT NULL DEFAULT 0,
            ppv NUMERIC NOT NULL DEFAULT 1
        )"""
    )
    yield Modes(manager)
    manager.close()


def test_bulk_insert_and_update_use_explicit_string_keys(modes):
    additions = pd.DataFrame(
        {
            "mode_id": ["c", "b"],
            "mode_name": ["car", "bicycle"],
            "description": ["Motorized", "Human powered"],
        }
    )
    original = additions.copy()

    assert modes.insert_from(additions) == ["c", "b"]
    pd.testing.assert_frame_equal(additions, original)

    changes = pd.DataFrame({"mode_id": ["c", "b"], "vot": [12.5, 3.0]})
    assert modes.update_from(changes) == 2
    assert {mode.mode_id: mode.vot for mode in modes} == {"c": 12.5, "b": 3}


def test_bulk_insert_requires_mode_id(modes):
    frame = pd.DataFrame({"mode_name": ["car"]})
    with pytest.raises(ValueError, match="non-numeric key tables"):
        modes.insert_from(frame)
