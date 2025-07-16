from aequilibrae.utils.create_example import create_example
from aequilibrae.project.tools import MigrationManager, MigrationStatus
from aequilibrae.utils.db_utils import AequilibraEConnection

from unittest import TestCase
import tempfile
import sqlite3
import pathlib


class TestMigrationManager(TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.connections = {"conn": sqlite3.connect(":memory:", factory=AequilibraEConnection)}
        self.main_connection = "conn"

        self.migrations_file = pathlib.Path(__file__).parent.parent.parent / "data" / "mock_migrations" / "init.py"
        self.migrations_duplicate = (
            pathlib.Path(__file__).parent.parent.parent / "data" / "mock_migrations" / "duplicate_init.py"
        )
        self.migrations_negative = (
            pathlib.Path(__file__).parent.parent.parent / "data" / "mock_migrations" / "negative_init.py"
        )

    def tearDown(self):
        self.connections[self.main_connection].close()
        self.tmpdir.cleanup()

    def test_migration_manager_init(self):
        manager = MigrationManager(self.migrations_file)

        # Check migrations were loaded correctly
        self.assertEqual(len(manager.migrations), 6)
        self.assertEqual(manager.migrations[0].name, "initial_migration")
        self.assertEqual(manager.migrations[1].name, "add_users")
        self.assertEqual(manager.migrations[2].name, "add_posts")
        self.assertEqual(manager.migrations[3].name, "add_comments")
        self.assertEqual(manager.migrations[4].name, "invalid_migration")
        self.assertEqual(manager.migrations[5].name, "non_callable_migrate")

    def test_migration_manager_duplicate_ids(self):
        with self.assertRaises(ValueError):
            MigrationManager(self.migrations_duplicate)

    def test_migration_manager_invalid_id(self):
        with self.assertRaises(ValueError):
            MigrationManager(self.migrations_negative)

    def test_status(self):
        manager = MigrationManager(self.migrations_file)
        conn = self.connections[self.main_connection]

        # Initially all should be missing except initial which gets auto-applied
        status = manager.status(conn)
        self.assertEqual(status[0], MigrationStatus.APPLIED)
        self.assertEqual(status[1], MigrationStatus.MISSING)
        self.assertEqual(status[2], MigrationStatus.MISSING)
        self.assertEqual(status[3], MigrationStatus.MISSING)

        # Check migrations table was created
        sql = "SELECT name FROM sqlite_master WHERE type='table' AND name='migrations'"
        self.assertIsNotNone(conn.execute(sql).fetchone())

    def test_mark_all_as_seen(self):
        manager = MigrationManager(self.migrations_file)
        conn = self.connections[self.main_connection]
        manager.mark_all_as_seen(conn)

        # All should be marked as MISSING
        status = manager.status(conn)
        for id, stat in status.items():
            if id == 0:
                self.assertEqual(stat, MigrationStatus.APPLIED)
            else:
                self.assertEqual(stat, MigrationStatus.MISSING)

        # Check entries exist in migrations table
        rows = conn.execute("SELECT id, status FROM migrations ORDER BY id").fetchall()
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0][1], "APPLIED")
        self.assertEqual(rows[1][1], "MISSING")
        self.assertEqual(rows[2][1], "MISSING")
        self.assertEqual(rows[3][1], "MISSING")
        self.assertEqual(rows[4][1], "MISSING")
        self.assertEqual(rows[5][1], "MISSING")

    def test_find_applicable(self):
        manager = MigrationManager(self.migrations_file)
        conn = self.connections[self.main_connection]

        # Should find all non-initial migrations
        applicable = manager.find_applicable(conn)
        self.assertEqual(len(applicable), 5)
        self.assertEqual(applicable[0].id, 1)
        self.assertEqual(applicable[1].id, 2)
        self.assertEqual(applicable[2].id, 3)
        self.assertEqual(applicable[3].id, 4)
        self.assertEqual(applicable[4].id, 5)

        # Apply first and second migration
        applicable[0].apply(conn, self.connections)
        applicable[0].mark_as(conn, MigrationStatus.APPLIED)

        applicable[1].apply(conn, self.connections)
        applicable[1].mark_as(conn, MigrationStatus.APPLIED)

        # Should now find only remaining migrations
        applicable = manager.find_applicable(conn)
        self.assertEqual(len(applicable), 3)
        self.assertEqual(applicable[0].id, 3)
        self.assertEqual(applicable[1].id, 4)
        self.assertEqual(applicable[2].id, 5)

    def test_out_of_order_migrations(self):
        manager = MigrationManager(self.migrations_file)
        conn = self.connections[self.main_connection]

        # Apply migrations 0, 1, and 3 but not 2
        manager.migrations[0].apply(conn, self.connections)
        manager.migrations[0].mark_as(conn, MigrationStatus.APPLIED)

        manager.migrations[1].apply(conn, self.connections)
        manager.migrations[1].mark_as(conn, MigrationStatus.APPLIED)

        manager.migrations[3].apply(conn, self.connections)
        manager.migrations[3].mark_as(conn, MigrationStatus.APPLIED)

        # Should raise error because migration 2 was skipped
        with self.assertRaises(RuntimeError):
            manager.find_applicable(conn)

    def test_upgrade(self):
        manager = MigrationManager(self.migrations_file)
        del manager.migrations[4]  # drop the duds
        del manager.migrations[5]

        # Upgrade should apply all migrations
        manager.upgrade(self.main_connection, self.connections)

        # Check all migrations were applied
        status = manager.status(self.connections[self.main_connection])
        for id, stat in status.items():
            self.assertEqual(stat, MigrationStatus.APPLIED)

        # Check tables were created
        tables = (
            self.connections[self.main_connection]
            .execute("SELECT name FROM sqlite_master WHERE type='table'")
            .fetchall()
        )
        table_names = [t[0] for t in tables]
        self.assertIn("migrations", table_names)
        self.assertIn("users", table_names)
        self.assertIn("posts", table_names)
        self.assertIn("comments", table_names)

    def test_upgrade_with_skip(self):
        manager = MigrationManager(self.migrations_file)
        conn = self.connections[self.main_connection]
        del manager.migrations[4]
        del manager.migrations[5]

        manager.mark_all_as_seen(conn)

        # Skip migration 2
        manager.upgrade(self.main_connection, self.connections, skip={2})

        # Check migrations 1 and 3 were applied, 2 was skipped
        status = manager.status(conn)
        self.assertEqual(status[0], MigrationStatus.APPLIED)
        self.assertEqual(status[1], MigrationStatus.APPLIED)
        self.assertEqual(status[2], MigrationStatus.SKIPPED)
        self.assertEqual(status[3], MigrationStatus.APPLIED)

        # There are no applicable upgrades now
        applicable = manager.find_applicable(conn)
        self.assertListEqual(applicable, [])

        # Check tables were created (should have users and comments but not posts)
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = [t[0] for t in tables]
        self.assertIn("migrations", table_names)
        self.assertIn("users", table_names)
        self.assertNotIn("posts", table_names)  # Was skipped
        self.assertIn("comments", table_names)

        manager.migrations[2].apply(conn, self.connections)
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        self.assertIn("posts", [t[0] for t in tables])  # Was just applied

        status = manager.status(conn)
        self.assertEqual(status[2], MigrationStatus.APPLIED)
