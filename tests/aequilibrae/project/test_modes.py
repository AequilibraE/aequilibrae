import sqlite3
from dataclasses import FrozenInstanceError
from string import ascii_letters, ascii_lowercase, ascii_uppercase

import pytest

from aequilibrae.project.network.modes import Modes
from aequilibrae.utils.db_utils import NestedTransactionManager


@pytest.fixture
def modes():
    manager = NestedTransactionManager(sqlite3.connect(":memory:"))
    manager._connection.executescript(
        """
        CREATE TABLE modes (
            mode_name TEXT UNIQUE NOT NULL,
            mode_id TEXT UNIQUE NOT NULL PRIMARY KEY CHECK (length(mode_id) = 1),
            description TEXT,
            pce NUMERIC NOT NULL DEFAULT 1.0,
            vot NUMERIC NOT NULL DEFAULT 0,
            ppv NUMERIC NOT NULL DEFAULT 1.0
        );
        CREATE TABLE attributes_documentation (
            name_table TEXT NOT NULL,
            attribute TEXT NOT NULL,
            description TEXT,
            PRIMARY KEY (name_table, attribute)
        );
        INSERT INTO modes (mode_name, mode_id, description)
        VALUES ('car', 'c', 'All motorized vehicles'), ('walk', 'w', 'Walking links');
        """
    )
    yield Modes(manager)
    manager.close()


def test_container_and_lookup_interfaces(modes):
    assert len(modes) == 2
    assert "c" in modes
    assert "x" not in modes
    assert {mode.mode_id for mode in modes} == {"c", "w"}
    assert modes.get("c") == modes.get_by_name("car")

    with pytest.raises(ValueError, match="modes has no record with mode_id='x'"):
        modes.get("x")
    with pytest.raises(ValueError, match="Mode hovercraft does not exist"):
        modes.get_by_name("hovercraft")


def test_insert_update_delete_and_record_immutability(modes):
    key = modes.insert(mode_id="b", mode_name="bicycle", description="Human powered")
    assert key == "b"
    assert modes.get(key).pce == 1

    record = modes.get(key)
    modes.update(key, description="Bikes")
    assert record.description == "Human powered"
    assert modes.get(key).description == "Bikes"
    with pytest.raises(FrozenInstanceError, match="cannot assign to field 'description'"):
        record.description = "mutable"

    modes.delete(key)
    assert key not in modes
    with pytest.raises(ValueError, match="modes has no record with mode_id='b'"):
        modes.delete(key)


def test_mode_constraints_and_fields_interface(modes):
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        modes.insert(mode_id="ab", mode_name="invalid")
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed: modes.mode_name"):
        modes.insert(mode_id="x", mode_name="car")

    editor = modes.fields
    assert editor._table == "modes"
    assert set(editor.all_fields()) == set(modes.columns)

    editor.add("fare", "Typical fare", "NUMERIC")
    modes.update("c", fare=2.5)
    assert modes.get("c").fare == 2.5


def test_available_ids(modes):
    assert set(modes.available_ids()) == set(ascii_letters) - {"c", "w"}
    assert set(modes.available_ids(full_list=ascii_lowercase)) == set(ascii_lowercase) - {"c", "w"}
    assert set(modes.available_ids(full_list=ascii_uppercase)) == set(ascii_uppercase)
    assert set(modes.available_ids(full_list=[])) == set()
