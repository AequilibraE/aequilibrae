import os
import sqlite3
from random import choice
from unittest import TestCase
from warnings import warn

from tests.models_for_test import ModelsTest


class TestProject(TestCase):
    def setUp(self) -> None:
        tm = ModelsTest()
        self.proj = tm.no_triggers()
        self.curr = self.proj.conn.cursor()

        # Modes to add
        sql = "INSERT INTO modes (mode_name, mode_id) VALUES (?, ?);"
        for mid in ["p", "l", "g", "x", "y", "d", "k", "a", "r", "n", "m"]:
            self.curr.execute(sql, [f"mode_{mid}", mid])

        self.proj.conn.commit()

        curr = self.proj.conn.cursor()
        self.rtree = True
        try:
            curr.execute("SELECT rtreecheck('idx_nodes_geometry');")
        except Exception as e:
            self.rtree = False
            warn(f"RTREE not available --> {e.args}")

        root = os.path.dirname(os.path.realpath(__file__)).replace("tests", "")
        qry_file = os.path.join(root, "database_specification/network/triggers/modes_table_triggers.sql")
        with open(qry_file, "r") as sql_file:
            self.queries = sql_file.read()
        self.queries = [cmd for cmd in self.queries.split("#")]

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

    def test_mode_single_letter_update(self):
        with self.assertRaises(sqlite3.IntegrityError):
            sql = "UPDATE 'modes' SET mode_id= 'ttt' where mode_id='b'"
            self.curr.execute(sql)

    def test_mode_single_letter_insert(self):
        sql = "INSERT INTO 'modes' (mode_name, mode_id) VALUES(?, ?)"

        with self.assertRaises(sqlite3.IntegrityError):
            self.curr.execute(sql, ["testasdasd", "pp"])

    def test_mode_keep_if_in_use_updating(self):
        cmd = self.__get_query("mode_keep_if_in_use_updating")

        sql = "UPDATE 'modes' SET mode_id= 'h' where mode_id='g'"
        self.curr.execute(sql)

        self.curr.execute(cmd)

        sql = "UPDATE 'modes' SET mode_id= 'j' where mode_id='l'"
        with self.assertRaises(sqlite3.IntegrityError):
            self.curr.execute(sql)

    def test_mode_keep_if_in_use_deleting(self):
        cmd = self.__get_query("mode_keep_if_in_use_deleting")

        sql = "DELETE FROM 'modes' where mode_id='p'"
        self.curr.execute(sql)

        self.curr.execute(cmd)

        sql = "DELETE FROM 'modes' where mode_id='c'"
        with self.assertRaises(sqlite3.IntegrityError):
            self.curr.execute(sql)

    def test_modes_on_links_update(self):
        cmd = self.__get_query("modes_on_links_update")

        sql = "UPDATE 'links' SET modes= 'qwerty' where link_id=55"
        self.curr.execute(sql)

        self.curr.execute(cmd)

        sql = "UPDATE 'links' SET modes= 'azerty' where link_id=56"
        with self.assertRaises(sqlite3.IntegrityError):
            self.curr.execute(sql)

    def test_modes_length_on_links_update(self):
        with self.assertRaises(sqlite3.IntegrityError):
            sql = "UPDATE 'links' SET modes= '' where modes='wb'"

            self.curr.execute(sql)

    def test_modes_on_nodes_table_update_a_node(self):
        cmd = self.__get_query("modes_on_nodes_table_update_a_node")

        sql = "UPDATE 'links' SET a_node= 1 where a_node=3"
        self.curr.execute(sql)

        sql = "SELECT modes from nodes where node_id=1"
        self.curr.execute(sql)
        i = self.curr.fetchone()[0]
        self.assertEqual(i, "ct")

        self.curr.execute(cmd)

        k = ""
        for n in [2, 5]:
            for f in ["a_node", "b_node"]:
                self.curr.execute(f"SELECT modes from links where {f}={n}")
                k += self.curr.fetchone()[0]

        existing = set(k)

        sql = "UPDATE 'links' SET a_node= 2 where a_node=5"
        self.curr.execute(sql)

        sql = "SELECT modes from nodes where node_id=2"
        self.curr.execute(sql)
        i = set(self.curr.fetchone()[0])
        self.assertEqual(i, existing)

    def test_modes_on_nodes_table_update_b_node(self):
        cmd = self.__get_query("modes_on_nodes_table_update_b_node")
        sql = "UPDATE 'links' SET b_node= 1 where b_node=3"
        self.curr.execute(sql)

        sql = "SELECT modes from nodes where node_id=1"
        self.curr.execute(sql)
        i = self.curr.fetchone()[0]
        self.assertEqual(i, "ct")

        self.curr.execute(cmd)

        sql = "UPDATE 'links' SET b_node= 2 where b_node=4"
        self.curr.execute(sql)

        sql = "SELECT modes from nodes where node_id=2"
        self.curr.execute(sql)
        i = self.curr.fetchone()[0]
        self.assertEqual(i, "ctw")

    def test_modes_on_nodes_table_update_links_modes(self):
        cmd = self.__get_query("modes_on_nodes_table_update_links_modes")
        sql = "UPDATE 'links' SET modes= 'x' where a_node=24"
        self.curr.execute(sql)

        sql = "SELECT modes from nodes where node_id=24"
        self.curr.execute(sql)
        i = self.curr.fetchone()[0]
        self.assertEqual(i, "c")

        self.curr.execute(cmd)

        sql = "UPDATE 'links' SET 'modes'= 'y' where a_node=24"
        self.curr.execute(sql)

        sql = "SELECT modes from nodes where node_id=24"
        self.curr.execute(sql)
        i = self.curr.fetchone()[0]
        self.assertIn("c", i)
        self.assertIn("y", i)

        sql = "UPDATE 'links' SET 'modes'= 'r' where b_node=24"
        self.curr.execute(sql)

        sql = "SELECT modes from nodes where node_id=24"
        self.curr.execute(sql)
        i = self.curr.fetchone()[0]
        self.assertIn("r", i)
        self.assertIn("y", i)

    def test_modes_on_links_insert(self):
        cmd = self.__get_query("modes_on_links_insert")
        if self.rtree:
            self.curr.execute("pragma table_info(links)")
            f = self.curr.fetchall()
            fields = {x[1]: x[0] for x in f}

            sql = "select * from links where link_id=10"
            self.curr.execute(sql)
            a = [x for x in self.curr.fetchone()]
            a[fields["modes"]] = "as12"
            a[fields["link_id"]] = 1234
            a[fields["a_node"]] = 999
            a[fields["b_node"]] = 888
            a[0] = 1234

            idx = ",".join(["?"] * len(a))
            self.curr.execute(f"insert into links values ({idx})", a)
            self.curr.execute("delete from links where link_id=1234")

            self.curr.execute(cmd)

            with self.assertRaises(sqlite3.IntegrityError):
                self.curr.execute(f"insert into links values ({idx})", a)

    def test_modes_length_on_links_insert(self):
        if not self.rtree:
            return

        self.curr.execute("pragma table_info(links)")
        f = self.curr.fetchall()
        fields = {x[1]: x[0] for x in f}

        sql = "select * from links where link_id=70"
        self.curr.execute(sql)
        a = [x for x in self.curr.fetchone()]
        a[fields["modes"]] = ""
        a[fields["link_id"]] = 4321
        a[fields["a_node"]] = 888
        a[fields["b_node"]] = 999
        a[0] = 4321

        idx = ",".join(["?"] * len(a))
        with self.assertRaises(sqlite3.IntegrityError):
            self.curr.execute(f"insert into links values ({idx})", a)

    def test_keep_at_least_one(self):
        cmd = self.__get_query("mode_keep_at_least_one")

        self.curr.execute("Delete from modes;")
        self.curr.execute("select count(*) from modes;")
        self.assertEqual(self.curr.fetchone()[0], 0, "We were not able to clear the m odes table as expected")

        self.curr.execute('insert into modes(mode_id, mode_name) VALUES("k", "some_name")')
        self.curr.execute(cmd)
        with self.assertRaises(sqlite3.IntegrityError):
            self.curr.execute("Delete from modes;")

    def test_modes_on_nodes_table_update_nodes_modes(self):
        cmd = self.__get_query("modes_on_nodes_table_update_nodes_modes")
        self.curr.execute("select node_id, modes from nodes where length(modes)>0")
        dt = self.curr.fetchall()

        x = choice(dt)

        self.curr.execute(f'update nodes set modes="abcdefgq" where node_id={x[0]}')
        self.curr.execute(f"select node_id, modes from nodes where node_id={x[0]}")
        z = self.curr.fetchone()
        if z == x:
            self.fail("Modes field on nodes layer is being preserved by unknown mechanism")

        self.curr.execute(cmd)

        y = choice(dt)
        while y == x:
            y = choice(dt)

        # We try to force the change to make sure it was correctly filled to begin with
        self.curr.execute(f'update nodes set modes="hgfedcba" where node_id={y[0]}')

        self.curr.execute(f"select node_id, modes from nodes where node_id={y[0]}")
        k = self.curr.fetchone()

        self.curr.execute(f'update nodes set modes="abcdefgq" where node_id={y[0]}')

        self.curr.execute(f"select node_id, modes from nodes where node_id={y[0]}")
        z = self.curr.fetchone()

        self.assertEqual(z, k, "Failed to preserve the information on modes for the nodes")
