from unittest import TestCase
import tempfile
import os
from random import choice
from warnings import warn
from shutil import copytree, rmtree
from aequilibrae.project import Project
import uuid
import sqlite3
from aequilibrae import logger
from ...data import no_triggers_project


class TestProject(TestCase):
    def setUp(self) -> None:
        self.temp_proj_folder = os.path.join(tempfile.gettempdir(), uuid.uuid4().hex)
        copytree(no_triggers_project, self.temp_proj_folder)
        self.proj = Project()
        self.proj.open(self.temp_proj_folder)
        self.curr = self.proj.conn.cursor()

        # Modes to add
        sql = "INSERT INTO modes (mode_name, mode_id) VALUES (?, ?);"
        for mid in ["p", "l", "g", "x", "y", "d", "k", "a", "r", "n", "m"]:
            self.curr.execute(sql, [f"mode_{mid}", mid])

        self.proj.conn.commit()

        root = os.path.dirname(os.path.realpath(__file__)).replace("tests", "")
        qry_file = os.path.join(root, "database_specification/triggers/link_type_table_triggers.sql")
        with open(qry_file, "r") as sql_file:
            self.queries = sql_file.read()
        self.queries = [cmd for cmd in self.queries.split("#")]

        curr = self.proj.conn.cursor()
        self.rtree = True
        try:
            curr.execute("SELECT rtreecheck('idx_nodes_geometry');")
        except Exception as e:
            self.rtree = False
            warn(f"RTREE not available --> {e.args}")

    def tearDown(self) -> None:
        self.proj.close()

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
        cmd = self.__get_query("link_type_single_letter_update")
        sql = "UPDATE 'link_types' SET link_type_id= 'ttt' where link_type_id='t'"
        self.curr.execute(sql)

        self.curr.execute(cmd)

        sql = "UPDATE 'link_types' SET link_type_id= 'ww' where link_type_id='w'"
        with self.assertRaises(sqlite3.IntegrityError):
            self.curr.execute(sql)

    def test_link_type_single_letter_insert(self):
        cmd = self.__get_query("link_type_single_letter_insert")
        sql = "INSERT INTO 'link_types' (link_type, link_type_id) VALUES(?, ?)"
        self.curr.execute(sql, ["test1a", "more_than_one"])

        self.curr.execute(cmd)

        with self.assertRaises(sqlite3.IntegrityError):
            self.curr.execute(sql, ["test1b", "mm"])

    def test_link_type_keep_if_in_use_updating(self):
        cmd = self.__get_query("link_type_keep_if_in_use_updating")

        sql = "UPDATE 'link_types' SET link_type= 'ttt' where link_type='test'"
        self.curr.execute(sql)

        self.curr.execute(cmd)

        sql = "UPDATE 'link_types' SET link_type= 'QQQ' where link_type='test2'"
        with self.assertRaises(sqlite3.IntegrityError):
            self.curr.execute(sql)

    def test_link_type_keep_if_in_use_deleting(self):
        cmd = self.__get_query("link_type_keep_if_in_use_deleting")

        sql = "DELETE FROM 'link_types' where link_type='test3'"
        self.curr.execute(sql)

        self.curr.execute(cmd)

        sql = "DELETE FROM 'link_types' where link_type='test4'"
        with self.assertRaises(sqlite3.IntegrityError):
            self.curr.execute(sql)

    def test_link_type_on_links_update(self):
        cmd = self.__get_query("link_type_on_links_update")

        sql = "UPDATE 'links' SET link_type= 'rrr' where link_type='test3'"
        self.curr.execute(sql)

        self.curr.execute(cmd)

        sql = "UPDATE 'links' SET link_type= 'not_valid_type' where link_type='test4'"
        with self.assertRaises(sqlite3.IntegrityError):
            self.curr.execute(sql)

    def test_link_type_on_links_insert(self):
        cmd = self.__get_query("link_type_on_links_insert")

        if self.rtree:
            self.curr.execute("pragma table_info(links)")
            f = self.curr.fetchall()
            fields = {x[1]: x[0] for x in f}

            sql = "select * from links where link_id=70"
            self.curr.execute(sql)
            a = [x for x in self.curr.fetchone()]
            a[fields["link_type"]] = "something indeed silly123"
            a[fields["link_id"]] = 456789
            a[fields["a_node"]] = 777
            a[fields["b_node"]] = 999
            a[0] = 456789

            idx = ",".join(["?"] * len(a))
            self.curr.execute(f"insert into links values ({idx})", a)
            self.curr.execute("delete from links where link_id=456789")

            self.curr.execute(cmd)

            with self.assertRaises(sqlite3.IntegrityError):
                self.curr.execute(f"insert into links values ({idx})", a)

            self.curr.execute("select link_type from link_types;")
            a[fields["link_type"]] = self.curr.fetchone()[0]
            self.curr.execute(f"insert into links values ({idx})", a)

    def test_link_type_on_links_delete_protected_link_type(self):
        cmd = self.__get_query("link_type_on_links_delete_protected_link_type")

        self.curr.execute(cmd)
        with self.assertRaises(sqlite3.IntegrityError):
            self.curr.execute('delete from link_types where link_type_id="z"')

        with self.assertRaises(sqlite3.IntegrityError):
            self.curr.execute('delete from link_types where link_type_id="y"')

    def test_link_type_id_keep_if_protected_type(self):
        cmd = self.__get_query("link_type_id_keep_if_protected_type")

        self.curr.execute(cmd)

        with self.assertRaises(sqlite3.IntegrityError):
            self.curr.execute('update link_types set link_type_id="x" where link_type_id="y"')

        with self.assertRaises(sqlite3.IntegrityError):
            self.curr.execute('update link_types set link_type_id="x" where link_type_id="z"')

    def test_link_type_keep_if_protected_type(self):
        cmd = self.__get_query("link_type_keep_if_protected_type")
        self.curr.execute(cmd)

        with self.assertRaises(sqlite3.IntegrityError):
            self.curr.execute('update link_types set link_type="xsdfg" where link_type_id="z"')

        with self.assertRaises(sqlite3.IntegrityError):
            self.curr.execute('update link_types set link_type="xsdfg" where link_type_id="y"')

    def test_link_type_on_nodes_table_update_nodes_link_type(self):
        cmd = self.__get_query("link_type_on_nodes_table_update_nodes_link_type")
        self.curr.execute(cmd)

        self.curr.execute('update nodes set link_types="qwerrreyrtuyiuio" where node_id=1')

        self.curr.execute("select link_types from nodes where node_id=1")
        lts = self.curr.fetchone()[0]

        self.assertEqual(lts, "etuw", "link_types was allowed to be corrupted in the nodes table")

    def test_link_type_on_nodes_table_update_links_link_type(self):
        cmd = self.__get_query("link_type_on_nodes_table_update_links_link_type")
        self.curr.execute(cmd)

        self.curr.execute('update links set link_type="test" where link_id=15')

        self.curr.execute("select link_types from nodes where node_id=6")
        lts = self.curr.fetchone()[0]

        self.assertEqual(lts, "rtw", "link_types on nodes table not updated with new link type in the links")

        self.curr.execute("select link_types from nodes where node_id=5")
        lts = self.curr.fetchone()[0]

        self.assertEqual(lts, "ertw", "link_types was allowed to be corrupted in the nodes table")

    def test_link_type_on_nodes_table_update_links_a_node(self):
        cmd = self.__get_query("link_type_on_nodes_table_update_links_a_node")
        self.curr.execute(cmd)

        self.curr.execute("update links set a_node=1 where link_id=15")

        self.curr.execute("select link_types from nodes where node_id=1")
        lts = self.curr.fetchone()[0]

        self.assertEqual(lts, "etuw", "link_types on nodes table not updated with new link type in the links")

        self.curr.execute("select link_types from nodes where node_id=6")
        lts = self.curr.fetchone()[0]

        self.assertEqual(lts, "rw", "link_types was allowed to be corrupted in the nodes table")

    def test_link_type_on_nodes_table_update_links_b_node(self):
        cmd = self.__get_query("link_type_on_nodes_table_update_links_b_node")
        self.curr.execute(cmd)

        self.curr.execute("update links set b_node=1 where link_id=15")

        self.curr.execute("select link_types from nodes where node_id=1")
        lts = self.curr.fetchone()[0]

        self.assertEqual(lts, "etuw", "link_types on nodes table not updated with new link type in the links")

        self.curr.execute("select link_types from nodes where node_id=5")
        lts = self.curr.fetchone()[0]

        self.assertEqual(lts, "ertw", "link_types was allowed to be corrupted in the nodes table")
