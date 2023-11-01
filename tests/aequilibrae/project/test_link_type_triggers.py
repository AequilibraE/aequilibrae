import os
import sqlite3
from unittest import TestCase
from warnings import warn

from tests.models_for_test import ModelsTest


class TestProject(TestCase):
    def setUp(self) -> None:
        tm = ModelsTest()
        self.proj = tm.no_triggers()

        # Modes to add
        sql = "INSERT INTO modes (mode_name, mode_id) VALUES (?, ?);"
        for mid in ["p", "l", "g", "x", "y", "d", "k", "a", "r", "n", "m"]:
            self.proj.conn.execute(sql, [f"mode_{mid}", mid])

        self.proj.conn.commit()

        root = os.path.dirname(os.path.realpath(__file__)).replace("tests", "")
        qry_file = os.path.join(root, "database_specification/network/triggers/link_type_table_triggers.sql")
        with open(qry_file, "r") as sql_file:
            self.queries = sql_file.read()
        self.queries = [cmd for cmd in self.queries.split("#")]

    def tearDown(self) -> None:
        self.proj.close()

    @property
    def rtree(self) -> bool:
        try:
            self.proj.conn.execute("SELECT rtreecheck('idx_nodes_geometry');")
        except Exception as e:
            warn(f"RTREE not available --> {e.args}")
            return False
        return True

    def __get_query(self, qry: str) -> str:
        for query in self.queries:
            if qry in query:
                return query
        raise FileNotFoundError("QUERY DOES NOT EXIST")

    def test_all_tests_considered(self):
        tests_added = list(self.__dir__())
        tests_added = [x[5:] for x in tests_added if x[:5] == "test_"]

        for trigger in self.queries:
            if "TRIGGER" in trigger.upper():
                found = [x for x in tests_added if x in trigger]
                if not found:
                    self.fail(f"Trigger not tested. {trigger}")

    def test_link_type_single_letter_update(self):
        sql = "UPDATE 'link_types' SET link_type_id= 'ttt' where link_type_id='t'"
        with self.assertRaises(sqlite3.IntegrityError):
            self.proj.conn.execute(sql)

    def test_link_type_single_letter_insert(self):
        sql = "INSERT INTO 'link_types' (link_type, link_type_id) VALUES(?, ?)"
        with self.assertRaises(sqlite3.IntegrityError):
            self.proj.conn.execute(sql, ["test1b", "mm"])

    def test_link_type_keep_if_in_use_updating(self):
        cmd = self.__get_query("link_type_keep_if_in_use_updating")

        sql = "UPDATE 'link_types' SET link_type= 'ttt' where link_type='test'"
        self.proj.conn.execute(sql)

        self.proj.conn.execute(cmd)

        sql = "UPDATE 'link_types' SET link_type= 'QQQ' where link_type='test2'"
        with self.assertRaises(sqlite3.IntegrityError):
            self.proj.conn.execute(sql)

    def test_link_type_keep_if_in_use_deleting(self):
        cmd = self.__get_query("link_type_keep_if_in_use_deleting")

        sql = "DELETE FROM 'link_types' where link_type='test3'"
        self.proj.conn.execute(sql)

        self.proj.conn.execute(cmd)

        sql = "DELETE FROM 'link_types' where link_type='test4'"
        with self.assertRaises(sqlite3.IntegrityError):
            self.proj.conn.execute(sql)

    def test_link_type_on_links_update(self):
        cmd = self.__get_query("link_type_on_links_update")

        sql = "UPDATE 'links' SET link_type= 'rrr' where link_type='test3'"
        self.proj.conn.execute(sql)

        self.proj.conn.execute(cmd)

        sql = "UPDATE 'links' SET link_type= 'not_valid_type' where link_type='test4'"
        with self.assertRaises(sqlite3.IntegrityError):
            self.proj.conn.execute(sql)

    def test_link_type_on_links_insert(self):
        cmd = self.__get_query("link_type_on_links_insert")

        if self.rtree:
            f = self.proj.conn.execute("pragma table_info(links)").fetchall()
            fields = {x[1]: x[0] for x in f}

            sql = "select * from links where link_id=70"
            a = [x for x in self.proj.conn.execute(sql).fetchone()]
            a[fields["link_type"]] = "something indeed silly123"
            a[fields["link_id"]] = 456789
            a[fields["a_node"]] = 777
            a[fields["b_node"]] = 999
            a[0] = 456789

            idx = ",".join(["?"] * len(a))
            self.proj.conn.execute(f"insert into links values ({idx})", a)
            self.proj.conn.execute("delete from links where link_id=456789")

            self.proj.conn.execute(cmd)

            with self.assertRaises(sqlite3.IntegrityError):
                self.proj.conn.execute(f"insert into links values ({idx})", a)

            sql = "select link_type from link_types;"

            a[fields["link_type"]] = self.proj.conn.execute(sql).fetchone()[0]
            self.proj.conn.execute(f"insert into links values ({idx})", a)

    def test_link_type_on_links_delete_protected_link_type(self):
        cmd = self.__get_query("link_type_on_links_delete_protected_link_type")

        self.proj.conn.execute(cmd)
        with self.assertRaises(sqlite3.IntegrityError):
            self.proj.conn.execute('delete from link_types where link_type_id="z"')

        with self.assertRaises(sqlite3.IntegrityError):
            self.proj.conn.execute('delete from link_types where link_type_id="y"')

    def test_link_type_id_keep_if_protected_type(self):
        cmd = self.__get_query("link_type_id_keep_if_protected_type")

        self.proj.conn.execute(cmd)

        with self.assertRaises(sqlite3.IntegrityError):
            self.proj.conn.execute('update link_types set link_type_id="x" where link_type_id="y"')

        with self.assertRaises(sqlite3.IntegrityError):
            self.proj.conn.execute('update link_types set link_type_id="x" where link_type_id="z"')

    def test_link_type_keep_if_protected_type(self):
        cmd = self.__get_query("link_type_keep_if_protected_type")
        self.proj.conn.execute(cmd)

        with self.assertRaises(sqlite3.IntegrityError):
            self.proj.conn.execute('update link_types set link_type="xsdfg" where link_type_id="z"')

        with self.assertRaises(sqlite3.IntegrityError):
            self.proj.conn.execute('update link_types set link_type="xsdfg" where link_type_id="y"')

    def test_link_type_on_nodes_table_update_nodes_link_type(self):
        cmd = self.__get_query("link_type_on_nodes_table_update_nodes_link_type")
        self.proj.conn.execute(cmd)

        self.proj.conn.execute('update nodes set link_types="qwerrreyrtuyiuio" where node_id=1')

        lts = self.proj.conn.execute("select link_types from nodes where node_id=1").fetchone()[0]

        self.assertEqual(lts, "etuw", "link_types was allowed to be corrupted in the nodes table")

    def test_link_type_on_nodes_table_update_links_link_type(self):
        cmd = self.__get_query("link_type_on_nodes_table_update_links_link_type")
        self.proj.conn.execute(cmd)

        self.proj.conn.execute('update links set link_type="test" where link_id=15')

        lts = self.proj.conn.execute("select link_types from nodes where node_id=6").fetchone()[0]

        self.assertEqual(lts, "grtw", "link_types on nodes table not updated with new link type in the links")

        lts = self.proj.conn.execute("select link_types from nodes where node_id=5").fetchone()[0]

        self.assertEqual(lts, "egrtw", "link_types was allowed to be corrupted in the nodes table")

    def test_link_type_on_nodes_table_update_links_a_node(self):
        cmd = self.__get_query("link_type_on_nodes_table_update_links_a_node")
        self.proj.conn.execute(cmd)

        self.proj.conn.execute("update links set a_node=1 where link_id=15")

        lts = self.proj.conn.execute("select link_types from nodes where node_id=1").fetchone()[0]

        self.assertEqual(lts, "etuw", "link_types on nodes table not updated with new link type in the links")

        lts = self.proj.conn.execute("select link_types from nodes where node_id=6").fetchone()[0]

        self.assertEqual(lts, "grw", "link_types was allowed to be corrupted in the nodes table")

    def test_link_type_on_nodes_table_update_links_b_node(self):
        cmd = self.__get_query("link_type_on_nodes_table_update_links_b_node")
        self.proj.conn.execute(cmd)

        self.proj.conn.execute("update links set b_node=1 where link_id=15")

        lts = self.proj.conn.execute("select link_types from nodes where node_id=1").fetchone()[0]

        self.assertEqual(lts, "etuw", "link_types on nodes table not updated with new link type in the links")

        lts = self.proj.conn.execute("select link_types from nodes where node_id=5").fetchone()[0]

        self.assertEqual(lts, "egrtw", "link_types was allowed to be corrupted in the nodes table")
