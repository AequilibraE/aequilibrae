import logging
import sqlite3

import pandas as pd
import pytest

from aequilibrae.project.data.result_record import ResultRecord
from aequilibrae.project.data.results import Results


class MockProject:
    def __init__(self, db_connection, results_connection):
        self.db_connection = db_connection
        self.results_connection = results_connection
        self.logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def results_sql_path(test_data_path):
    return test_data_path.parent.parent / "aequilibrae/project/database_specification/network/tables/results.sql"


# Fixtures for ResultRecord tests
@pytest.fixture
def db_connections(results_sql_path):
    project_conn = sqlite3.connect(":memory:")
    results_conn = sqlite3.connect(":memory:")

    with open(results_sql_path, "r") as f:
        project_conn.executescript(f.read())

    yield project_conn, results_conn

    project_conn.close()
    results_conn.close()


@pytest.fixture
def project(db_connections):
    project_conn, results_conn = db_connections
    return MockProject(project_conn, results_conn)


@pytest.fixture
def sample_data():
    return {
        "table_name": "test_result",
        "procedure": "test_procedure",
        "procedure_id": "test_id_123",
        "procedure_report": '{"status": "success", "items": 100}',
        "timestamp": "2000-01-01 12:00:00",
        "description": "Test result description",
        "year": "2020",
        "scenario": "testing",
        "reference_table": "links",
    }


# Fixtures for Results tests
@pytest.fixture
def populated_db_connections(results_sql_path):
    project_conn = sqlite3.connect(":memory:")
    results_conn = sqlite3.connect(":memory:")

    with open(results_sql_path, "r") as f:
        project_conn.executescript(f.read())

    project_conn.execute(
        "INSERT INTO results (table_name, procedure, procedure_id, procedure_report, timestamp, description) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("test_result1", "test_procedure", "id1", '{"status": "success"}', "2000-01-01 12:00:00", "Test 1"),
    )
    project_conn.execute(
        "INSERT INTO results (table_name, procedure, procedure_id, procedure_report, timestamp, description) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("test_result2", "test_procedure", "id2", '{"status": "success"}', "2000-01-01 13:00:00", "Test 2"),
    )
    project_conn.commit()

    yield project_conn, results_conn

    project_conn.close()
    results_conn.close()


@pytest.fixture
def populated_project(populated_db_connections):
    project_conn, results_conn = populated_db_connections
    return MockProject(project_conn, results_conn)


# ResultRecord tests
def test_record_init(db_connections, project, sample_data):
    project_conn, results_conn = db_connections
    record = ResultRecord(sample_data, project, project_conn, results_conn)

    assert record.table_name == "test_result"
    assert record.procedure == "test_procedure"
    assert record.procedure_id == "test_id_123"
    assert record.timestamp == "2000-01-01 12:00:00"
    assert record.description == "Test result description"
    assert record._exists


def test_save_new_record(db_connections, project, sample_data):
    project_conn, results_conn = db_connections
    record = ResultRecord(sample_data, project, project_conn, results_conn)
    record.save()

    cursor = project_conn.execute(
        "SELECT scenario, year, table_name, reference_table, procedure, procedure_id, "
        "procedure_report, timestamp, description FROM results WHERE table_name=?",
        ["test_result"],
    )
    result = cursor.fetchone()

    assert result is not None
    assert result[0] == "testing"
    assert result[1] == "2020"
    assert result[2] == "test_result"
    assert result[3] == "links"
    assert result[4] == "test_procedure"
    assert result[5] == "test_id_123"


def test_save_update_existing_record(db_connections, project, sample_data):
    project_conn, results_conn = db_connections
    record = ResultRecord(sample_data, project, project_conn, results_conn)
    record.save()

    record.description = "Updated description"
    record.save()

    cursor = project_conn.execute("SELECT description FROM results WHERE table_name=?", ["test_result"])
    result = cursor.fetchone()

    assert result[0] == "Updated description"


def test_delete(db_connections, project, sample_data):
    project_conn, results_conn = db_connections
    record = ResultRecord(sample_data, project, project_conn, results_conn)
    record.save()

    results_conn.execute(f"CREATE TABLE {record.table_name} (id INTEGER, value TEXT)")

    record.delete()

    cursor = project_conn.execute("SELECT * FROM results WHERE table_name=?", ["test_result"])
    result = cursor.fetchone()
    assert result is None

    cursor = results_conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", ["test_result"])
    result = cursor.fetchone()
    assert result is None

    assert not record._exists


def test_get_data(db_connections, project, sample_data):
    project_conn, results_conn = db_connections
    record = ResultRecord(sample_data, project, project_conn, results_conn)

    test_data = pd.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})
    test_data.to_sql(record.table_name, results_conn, index=False)

    result_data = record.get_data()

    assert isinstance(result_data, pd.DataFrame)
    assert len(result_data) == 3
    assert list(result_data.columns) == ["id", "value"]


def test_set_data(db_connections, project, sample_data):
    project_conn, results_conn = db_connections
    record = ResultRecord(sample_data, project, project_conn, results_conn)

    test_data = pd.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})

    record.set_data(test_data, index=False)

    cursor = results_conn.execute(f"SELECT * FROM {record.table_name}")
    results = cursor.fetchall()

    assert len(results) == 3
    assert results[0] == (1, "a")


def test_setattr_table_name_validation(db_connections, project, sample_data):
    project_conn, results_conn = db_connections
    record1 = ResultRecord(sample_data, project, project_conn, results_conn)
    record1.save()

    data2 = sample_data.copy()
    data2["table_name"] = "test_result2"
    record2 = ResultRecord(data2, project, project_conn, results_conn)

    with pytest.raises(ValueError):
        record2.table_name = "test_result"


# Results tests
def test_results_init(populated_db_connections, populated_project):
    project_conn, results_conn = populated_db_connections
    results = Results(populated_project, project_conn, results_conn)

    assert len(results._Results__items) == 2
    assert "test_result1" in results._Results__items
    assert "test_result2" in results._Results__items


def test_reload(populated_db_connections, populated_project):
    project_conn, results_conn = populated_db_connections
    results = Results(populated_project, project_conn, results_conn)

    project_conn.execute(
        "INSERT INTO results (table_name, procedure, procedure_id, procedure_report, timestamp, description) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("test_result3", "test_procedure", "id3", '{"status": "success"}', "2000-01-01 14:00:00", "Test 3"),
    )
    project_conn.commit()

    results.reload()

    assert len(results._Results__items) == 3
    assert "test_result3" in results._Results__items


def test_clear_database(populated_db_connections, populated_project):
    project_conn, results_conn = populated_db_connections
    results = Results(populated_project, project_conn, results_conn)

    results_conn.execute("CREATE TABLE test_result1 (id INTEGER)")

    results.clear_database()

    cursor = project_conn.execute("SELECT table_name FROM results")
    remaining_results = [row[0] for row in cursor.fetchall()]

    assert "test_result1" in remaining_results
    assert "test_result2" not in remaining_results


def test_update_database(populated_db_connections, populated_project):
    project_conn, results_conn = populated_db_connections
    results = Results(populated_project, project_conn, results_conn)

    results_conn.execute("CREATE TABLE new_result (id INTEGER)")

    results.update_database()

    cursor = project_conn.execute("SELECT table_name FROM results WHERE table_name='new_result'")
    result = cursor.fetchone()

    assert result is not None
    assert result[0] == "new_result"


def test_list(populated_db_connections, populated_project):
    project_conn, results_conn = populated_db_connections
    results = Results(populated_project, project_conn, results_conn)

    df = results.list()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "table_name" in df.columns
    assert "procedure" in df.columns


def test_get_results(populated_db_connections, populated_project):
    project_conn, results_conn = populated_db_connections
    results = Results(populated_project, project_conn, results_conn)

    test_data = pd.DataFrame({"id": [1, 2], "value": ["a", "b"]})
    test_data.to_sql("test_result1", results_conn, index=False)

    df = results.get_results("test_result1")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_get_record(populated_db_connections, populated_project):
    project_conn, results_conn = populated_db_connections
    results = Results(populated_project, project_conn, results_conn)

    record = results.get_record("test_result1")

    assert isinstance(record, ResultRecord)
    assert record.table_name == "test_result1"


def test_get_record_not_exists(populated_db_connections, populated_project):
    project_conn, results_conn = populated_db_connections
    results = Results(populated_project, project_conn, results_conn)

    with pytest.raises(ValueError):
        results.get_record("non_existent")


def test_check_exists(populated_db_connections, populated_project):
    project_conn, results_conn = populated_db_connections
    results = Results(populated_project, project_conn, results_conn)

    assert results.check_exists("test_result1") is True
    assert results.check_exists("non_existent") is False


def test_delete_record(populated_db_connections, populated_project):
    project_conn, results_conn = populated_db_connections
    results = Results(populated_project, project_conn, results_conn)

    results_conn.execute("CREATE TABLE test_result1 (id INTEGER)")

    results.delete_record("test_result1")

    cursor = project_conn.execute("SELECT * FROM results WHERE table_name='test_result1'")
    result = cursor.fetchone()
    assert result is None


def test_new_record(populated_db_connections, populated_project):
    project_conn, results_conn = populated_db_connections
    results = Results(populated_project, project_conn, results_conn)

    record = results.new_record(
        "new_test_result",
        procedure="new_procedure",
        procedure_id="new_id",
        procedure_report={"status": "pending"},
        timestamp="2000-01-01 15:00:00",
        description="New test result",
    )

    assert isinstance(record, ResultRecord)
    assert record.table_name == "new_test_result"
    assert record.procedure == "new_procedure"

    cursor = project_conn.execute("SELECT * FROM results WHERE table_name='new_test_result'")
    result = cursor.fetchone()
    assert result is not None


def test_new_record_with_scenario(populated_db_connections, populated_project):
    project_conn, results_conn = populated_db_connections
    results = Results(populated_project, project_conn, results_conn)

    record = results.new_record(
        "new_test_result",
        procedure="new_procedure",
        procedure_id="new_id",
        procedure_report={"status": "pending"},
        timestamp="2000-01-01 15:00:00",
        description="New test result",
        scenario="testing",
        year="2020",
    )

    assert isinstance(record, ResultRecord)
    assert record.scenario == "testing"
    assert record.year == "2020"

    cursor = project_conn.execute("SELECT * FROM results WHERE table_name='new_test_result'")
    result = cursor.fetchone()
    assert result is not None


def test_new_record_duplicate_name(populated_db_connections, populated_project):
    project_conn, results_conn = populated_db_connections
    results = Results(populated_project, project_conn, results_conn)

    with pytest.raises(ValueError):
        results.new_record("test_result1")


def test_clear(populated_db_connections, populated_project):
    project_conn, results_conn = populated_db_connections
    results = Results(populated_project, project_conn, results_conn)

    assert len(results._Results__items) == 2

    results._clear()

    assert len(results._Results__items) == 0
