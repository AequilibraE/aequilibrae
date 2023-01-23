from uuid import uuid4
import pandas as pd
import pytest

from aequilibrae.transit import Transit
from aequilibrae.project.database_connection import database_connection
from aequilibrae.transit.functions.data import get_table
from aequilibrae.transit.functions.db_utils import list_tables_in_db
from aequilibrae.utils.create_example import create_example


@pytest.fixture
def create_project(tmp_path):
    path = tmp_path / uuid4().hex
    project = create_example(path)
    Transit(project).create_transit_database()

    yield project
    project.close()


def test_get_table(create_project):
    conn = database_connection(db_type="transit")

    tables = get_table("routes", conn)

    assert type(tables) == pd.DataFrame


def test_list_tables_in_db(create_project):
    conn = database_connection(db_type="transit")

    list_table = list_tables_in_db(conn)

    assert len(list_table) >= 5
