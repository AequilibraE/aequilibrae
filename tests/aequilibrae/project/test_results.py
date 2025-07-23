import unittest
import sqlite3
import json
import pandas as pd
import tempfile
import pathlib
from unittest import TestCase
from datetime import datetime
import logging

from aequilibrae.project.data.results import Results
from aequilibrae.project.data.result_record import ResultRecord


RESULTS_SQL = (
    pathlib.Path(__file__).parent.parent.parent.parent
    / "aequilibrae"
    / "project"
    / "database_specification"
    / "network"
    / "tables"
    / "results.sql"
)


class MockProject:
    def __init__(self, db_connection, results_connection):
        self.db_connection = db_connection
        self.results_connection = results_connection
        self.logger = logging.getLogger(__name__)


class TestResultRecord(TestCase):
    def setUp(self):
        self.project_conn = sqlite3.connect(":memory:")
        self.results_conn = sqlite3.connect(":memory:")

        with open(RESULTS_SQL, "r") as f:
            self.project_conn.executescript(f.read())

        self.project = MockProject(self.project_conn, self.results_conn)

        self.sample_data = {
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

    def tearDown(self):
        self.project_conn.close()
        self.results_conn.close()

    def test_init(self):
        record = ResultRecord(self.sample_data, self.project, self.project_conn, self.results_conn)

        self.assertEqual(record.table_name, "test_result")
        self.assertEqual(record.procedure, "test_procedure")
        self.assertEqual(record.procedure_id, "test_id_123")
        self.assertEqual(record.timestamp, "2000-01-01 12:00:00")
        self.assertEqual(record.description, "Test result description")
        self.assertTrue(record._exists)

    def test_save_new_record(self):
        record = ResultRecord(self.sample_data, self.project, self.project_conn, self.results_conn)
        record.save()

        cursor = self.project_conn.execute(
            "SELECT scenario, year, table_name, reference_table, procedure, procedure_id, "
            "procedure_report, timestamp, description FROM results WHERE table_name=?",
            ["test_result"],
        )
        result = cursor.fetchone()

        self.assertIsNotNone(result)
        self.assertEqual(
            result,
            (
                "testing",
                "2020",
                "test_result",
                "links",
                "test_procedure",
                "test_id_123",
                '{"status": "success", "items": 100}',
                "2000-01-01 12:00:00",
                "Test result description",
            ),
        )

    def test_save_update_existing_record(self):
        record = ResultRecord(self.sample_data, self.project, self.project_conn, self.results_conn)
        record.save()

        record.description = "Updated description"
        record.save()

        cursor = self.project_conn.execute("SELECT description FROM results WHERE table_name=?", ["test_result"])
        result = cursor.fetchone()

        self.assertEqual(result[0], "Updated description")

    def test_delete(self):
        record = ResultRecord(self.sample_data, self.project, self.project_conn, self.results_conn)
        record.save()

        self.results_conn.execute(f"CREATE TABLE {record.table_name} (id INTEGER, value TEXT)")

        record.delete()

        cursor = self.project_conn.execute("SELECT * FROM results WHERE table_name=?", ["test_result"])
        result = cursor.fetchone()
        self.assertIsNone(result)

        cursor = self.results_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", ["test_result"]
        )
        result = cursor.fetchone()
        self.assertIsNone(result)

        self.assertFalse(record._exists)

    def test_get_data(self):
        record = ResultRecord(self.sample_data, self.project, self.project_conn, self.results_conn)

        test_data = pd.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})
        test_data.to_sql(record.table_name, self.results_conn, index=False)

        result_data = record.get_data()

        self.assertIsInstance(result_data, pd.DataFrame)
        self.assertEqual(len(result_data), 3)
        self.assertListEqual(list(result_data.columns), ["id", "value"])

    def test_set_data(self):
        record = ResultRecord(self.sample_data, self.project, self.project_conn, self.results_conn)

        test_data = pd.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})

        record.set_data(test_data, index=False)

        cursor = self.results_conn.execute(f"SELECT * FROM {record.table_name}")
        results = cursor.fetchall()

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], (1, "a"))

    def test_setattr_table_name_validation(self):
        record1 = ResultRecord(self.sample_data, self.project, self.project_conn, self.results_conn)
        record1.save()

        data2 = self.sample_data.copy()
        data2["table_name"] = "test_result2"
        record2 = ResultRecord(data2, self.project, self.project_conn, self.results_conn)

        with self.assertRaises(ValueError):
            record2.table_name = "test_result"


class TestResults(TestCase):
    def setUp(self):
        self.project_conn = sqlite3.connect(":memory:")
        self.results_conn = sqlite3.connect(":memory:")

        with open(RESULTS_SQL, "r") as f:
            self.project_conn.executescript(f.read())

        self.project = MockProject(self.project_conn, self.results_conn)

        self.project_conn.execute(
            "INSERT INTO results (table_name, procedure, procedure_id, procedure_report, timestamp, description) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("test_result1", "test_procedure", "id1", '{"status": "success"}', "2000-01-01 12:00:00", "Test 1"),
        )
        self.project_conn.execute(
            "INSERT INTO results (table_name, procedure, procedure_id, procedure_report, timestamp, description) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("test_result2", "test_procedure", "id2", '{"status": "success"}', "2000-01-01 13:00:00", "Test 2"),
        )
        self.project_conn.commit()

    def tearDown(self):
        self.project_conn.close()
        self.results_conn.close()

    def test_init(self):
        results = Results(self.project, self.project_conn, self.results_conn)

        self.assertEqual(len(results._Results__items), 2)
        self.assertIn("test_result1", results._Results__items)
        self.assertIn("test_result2", results._Results__items)

    def test_reload(self):
        results = Results(self.project, self.project_conn, self.results_conn)

        self.project_conn.execute(
            "INSERT INTO results (table_name, procedure, procedure_id, procedure_report, timestamp, description) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("test_result3", "test_procedure", "id3", '{"status": "success"}', "2000-01-01 14:00:00", "Test 3"),
        )
        self.project_conn.commit()

        results.reload()

        self.assertEqual(len(results._Results__items), 3)
        self.assertIn("test_result3", results._Results__items)

    def test_clear_database(self):
        results = Results(self.project, self.project_conn, self.results_conn)

        self.results_conn.execute("CREATE TABLE test_result1 (id INTEGER)")

        results.clear_database()

        cursor = self.project_conn.execute("SELECT table_name FROM results")
        remaining_results = [row[0] for row in cursor.fetchall()]

        self.assertIn("test_result1", remaining_results)
        self.assertNotIn("test_result2", remaining_results)

    def test_update_database(self):
        results = Results(self.project, self.project_conn, self.results_conn)

        self.results_conn.execute("CREATE TABLE new_result (id INTEGER)")

        results.update_database()

        cursor = self.project_conn.execute("SELECT table_name FROM results WHERE table_name='new_result'")
        result = cursor.fetchone()

        self.assertIsNotNone(result)
        self.assertEqual(result[0], "new_result")

    def test_list(self):
        results = Results(self.project, self.project_conn, self.results_conn)

        df = results.list()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn("table_name", df.columns)
        self.assertIn("procedure", df.columns)

    def test_get_results(self):
        results = Results(self.project, self.project_conn, self.results_conn)

        test_data = pd.DataFrame({"id": [1, 2], "value": ["a", "b"]})
        test_data.to_sql("test_result1", self.results_conn, index=False)

        df = results.get_results("test_result1")

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)

    def test_get_record(self):
        results = Results(self.project, self.project_conn, self.results_conn)

        record = results.get_record("test_result1")

        self.assertIsInstance(record, ResultRecord)
        self.assertEqual(record.table_name, "test_result1")

    def test_get_record_not_exists(self):
        results = Results(self.project, self.project_conn, self.results_conn)

        with self.assertRaises(ValueError):
            results.get_record("non_existent")

    def test_check_exists(self):
        results = Results(self.project, self.project_conn, self.results_conn)

        self.assertTrue(results.check_exists("test_result1"))
        self.assertFalse(results.check_exists("non_existent"))

    def test_delete_record(self):
        results = Results(self.project, self.project_conn, self.results_conn)

        self.results_conn.execute("CREATE TABLE test_result1 (id INTEGER)")

        results.delete_record("test_result1")

        cursor = self.project_conn.execute("SELECT * FROM results WHERE table_name='test_result1'")
        result = cursor.fetchone()
        self.assertIsNone(result)

    def test_new_record(self):
        results = Results(self.project, self.project_conn, self.results_conn)

        record = results.new_record(
            "new_test_result",
            procedure="new_procedure",
            procedure_id="new_id",
            procedure_report={"status": "pending"},
            timestamp="2000-01-01 15:00:00",
            description="New test result",
        )

        self.assertIsInstance(record, ResultRecord)
        self.assertEqual(record.table_name, "new_test_result")
        self.assertEqual(record.procedure, "new_procedure")

        cursor = self.project_conn.execute("SELECT * FROM results WHERE table_name='new_test_result'")
        result = cursor.fetchone()
        self.assertIsNotNone(result)

    def test_new_record_with_scenario(self):
        results = Results(self.project, self.project_conn, self.results_conn)

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

        self.assertIsInstance(record, ResultRecord)
        self.assertEqual(record.scenario, "testing")
        self.assertEqual(record.year, "2020")

        cursor = self.project_conn.execute("SELECT * FROM results WHERE table_name='new_test_result'")
        result = cursor.fetchone()
        self.assertIsNotNone(result)

    def test_new_record_duplicate_name(self):
        results = Results(self.project, self.project_conn, self.results_conn)

        with self.assertRaises(ValueError):
            results.new_record("test_result1")

    def test_clear(self):
        results = Results(self.project, self.project_conn, self.results_conn)

        self.assertEqual(len(results._Results__items), 2)

        results._clear()

        self.assertEqual(len(results._Results__items), 0)
