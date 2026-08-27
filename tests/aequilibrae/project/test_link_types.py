from dataclasses import FrozenInstanceError
import sqlite3

import pandas as pd
import pytest

from aequilibrae.project.network.link_types import LinkTypes
from aequilibrae.utils.db_utils import NestedTransactionManager


@pytest.fixture
def link_types():
    manager = NestedTransactionManager(sqlite3.connect(":memory:"))
    manager._connection.executescript(
        """
        CREATE TABLE link_types (
            link_type TEXT UNIQUE NOT NULL,
            link_type_id TEXT UNIQUE NOT NULL CHECK (length(link_type_id) = 1),
            description TEXT,
            lanes NUMERIC,
            lane_capacity NUMERIC,
            speed NUMERIC
        );
        CREATE TABLE attributes_documentation (
            name_table TEXT NOT NULL,
            attribute TEXT NOT NULL,
            description TEXT,
            PRIMARY KEY (name_table, attribute)
        );
        INSERT INTO link_types
            (link_type, link_type_id, description, lanes, lane_capacity)
        VALUES
            ('centroid_connector', 'z', 'Virtual connectors', 10, 10000),
            ('default', 'y', 'Default link type', 2, 900);
        """
    )
    yield LinkTypes(manager)
    manager.close()


def test_container_and_name_lookup_interfaces(link_types):
    assert len(link_types) == 2
    assert "y" in link_types
    assert {record.link_type_id for record in link_types} == {"y", "z"}
    assert link_types.get("y") == link_types.get_by_name("default")

    with pytest.raises(ValueError, match="link_types has no record with link_type_id='x'"):
        link_types.get("x")
    with pytest.raises(ValueError, match="Link type motorway does not exist"):
        link_types.get_by_name("motorway")


def test_crud_and_immutable_record(link_types):
    assert (
        link_types.insert(
            link_type_id="a", link_type="arterial", description="Arterial", lanes=3, lane_capacity=1200
        )
        == "a"
    )
    record = link_types.get("a")

    link_types.update("a", speed=60, description="Major arterial")
    assert record.speed is None
    assert link_types.get("a").speed == 60
    with pytest.raises(FrozenInstanceError, match="cannot assign to field 'speed'"):
        record.speed = 30

    link_types.delete("a")
    assert "a" not in link_types


def test_bulk_and_fields_interfaces(link_types):
    additions = pd.DataFrame(
        {
            "link_type_id": ["a", "l"],
            "link_type": ["arterial", "local"],
            "lanes": [3, 1],
        }
    )
    assert link_types.insert_from(additions) == ["a", "l"]
    assert link_types.update_from(pd.DataFrame({"link_type_id": ["a", "l"], "speed": [60, 30]})) == 2
    assert link_types.get_by_name("local").speed == 30

    editor = link_types.fields
    assert set(editor.all_fields()) == set(link_types.columns)
