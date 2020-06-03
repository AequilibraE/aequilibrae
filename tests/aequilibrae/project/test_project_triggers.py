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
        sql = 'INSERT INTO modes (mode_name, mode_id) VALUES (?, ?);'
        for mid in ['p', 'l', 'g', 'x', 'y', 'd', 'k', 'a', 'r', 'n', 'm']:
            self.curr.execute(sql, [f'mode_{mid}', mid])

        self.proj.conn.commit()

        curr = self.proj.conn.cursor()
        self.rtree = True
        try:
            curr.execute("SELECT rtreecheck('idx_nodes_geometry');")
        except Exception as e:
            self.rtree = False
            warn(f'RTREE not available --> {e.args}')

    def tearDown(self) -> None:
        self.proj.close()
        rmtree(self.temp_proj_folder)

    def test_link_type_triggers(self):
        root = os.path.dirname(os.path.realpath(__file__)).replace('tests', '')
        qry_file = os.path.join(root, "database_specification/triggers/link_type_table_triggers.sql")
        with open(qry_file, "r") as sql_file:
            query_list = sql_file.read()
            query_list = [cmd for cmd in query_list.split("#")]

            def reboot_cursor():
                self.proj.conn.commit()
                self.curr = self.proj.conn.cursor()

        for cmd in query_list:
            if 'link_type_single_letter_update' in cmd:
                sql = "UPDATE 'link_types' SET link_type_id= 'ttt' where link_type_id='t'"
                self.curr.execute(sql)

                self.curr.execute(cmd)
                reboot_cursor()

                sql = "UPDATE 'link_types' SET link_type_id= 'ww' where link_type_id='w'"
                with self.assertRaises(sqlite3.IntegrityError):
                    self.curr.execute(sql)
                reboot_cursor()

            elif 'link_type_single_letter_insert' in cmd:
                sql = "INSERT INTO 'link_types' (link_type, link_type_id) VALUES(?, ?)"
                self.curr.execute(sql, ['test1a', 'more_than_one'])

                self.curr.execute(cmd)
                reboot_cursor()

                with self.assertRaises(sqlite3.IntegrityError):
                    self.curr.execute(sql, ['test1b', 'mm'])
                reboot_cursor()

            elif 'link_type_keep_if_in_use_updating' in cmd:
                sql = "UPDATE 'link_types' SET link_type= 'ttt' where link_type='test'"
                self.curr.execute(sql)

                self.curr.execute(cmd)
                reboot_cursor()

                sql = "UPDATE 'link_types' SET link_type= 'QQQ' where link_type='test2'"
                with self.assertRaises(sqlite3.IntegrityError):
                    self.curr.execute(sql)
                reboot_cursor()

            elif 'link_type_keep_if_in_use_deleting' in cmd:
                sql = "DELETE FROM 'link_types' where link_type='test3'"
                self.curr.execute(sql)

                self.curr.execute(cmd)
                reboot_cursor()

                sql = "DELETE FROM 'link_types' where link_type='test4'"
                with self.assertRaises(sqlite3.IntegrityError):
                    self.curr.execute(sql)
                reboot_cursor()

            elif 'link_type_on_links_update' in cmd:
                sql = "UPDATE 'links' SET link_type= 'rrr' where link_type='test3'"
                self.curr.execute(sql)

                self.curr.execute(cmd)
                reboot_cursor()

                sql = "UPDATE 'links' SET link_type= 'not_valid_type' where link_type='test4'"
                with self.assertRaises(sqlite3.IntegrityError):
                    self.curr.execute(sql)
                reboot_cursor()

            elif 'link_type_on_links_insert' in cmd:
                if self.rtree:
                    self.curr.execute('pragma table_info(links)')
                    f = self.curr.fetchall()
                    fields = {x[1]: x[0] for x in f}

                    sql = 'select * from links where link_id=70'
                    self.curr.execute(sql)
                    a = [x for x in self.curr.fetchone()]
                    a[fields['link_type']] = 'something indeed silly123'
                    a[fields['link_id']] = 456789
                    a[fields['a_node']] = 777
                    a[fields['b_node']] = 999
                    a[0] = 456789

                    idx = ','.join(['?'] * len(a))
                    self.curr.execute(f'insert into links values ({idx})', a)
                    self.curr.execute('delete from links where link_id=456789')

                    self.curr.execute(cmd)
                    reboot_cursor()

                    with self.assertRaises(sqlite3.IntegrityError):
                        self.curr.execute(f'insert into links values ({idx})', a)

                    self.curr.execute('select link_type from link_types;')
                    a[fields['link_type']] = self.curr.fetchone()[0]
                    self.curr.execute(f'insert into links values ({idx})', a)

            elif 'link_type_on_links_delete_protected_link_type' in cmd:
                self.curr.execute(cmd)
                reboot_cursor()
                with self.assertRaises(sqlite3.IntegrityError):
                    self.curr.execute('delete from link_types where link_type_id="z"')

                with self.assertRaises(sqlite3.IntegrityError):
                    self.curr.execute('delete from link_types where link_type_id="y"')

            elif 'link_type_id_keep_if_protected_type' in cmd:
                self.curr.execute(cmd)
                reboot_cursor()

                with self.assertRaises(sqlite3.IntegrityError):
                    self.curr.execute('update link_types set link_type_id="x" where link_type_id="y"')

                with self.assertRaises(sqlite3.IntegrityError):
                    self.curr.execute('update link_types set link_type_id="x" where link_type_id="z"')

            elif 'link_type_keep_if_protected_type' in cmd:
                self.curr.execute(cmd)
                reboot_cursor()

                with self.assertRaises(sqlite3.IntegrityError):
                    self.curr.execute('update link_types set link_type="xsdfg" where link_type_id="z"')

                with self.assertRaises(sqlite3.IntegrityError):
                    self.curr.execute('update link_types set link_type="xsdfg" where link_type_id="y"')

            else:
                if 'TRIGGER' in cmd.upper():
                    logger.warning(cmd)
                    self.fail(f'Missing test for triggers in link_types table. {cmd}')

    def test_mode_triggers(self):
        root = os.path.dirname(os.path.realpath(__file__)).replace('tests', '')
        qry_file = os.path.join(root, "database_specification/triggers/modes_table_triggers.sql")
        with open(qry_file, "r") as sql_file:
            query_list = sql_file.read()
            query_list = [cmd for cmd in query_list.split("#")]

            def reboot_cursor():
                self.proj.conn.commit()
                self.curr = self.proj.conn.cursor()

        for cmd in query_list:
            if 'mode_single_letter_update' in cmd:
                sql = "UPDATE 'modes' SET mode_id= 'ttt' where mode_id='b'"
                self.curr.execute(sql)

                self.curr.execute(cmd)
                reboot_cursor()

                with self.assertRaises(sqlite3.IntegrityError):
                    sql = "UPDATE 'modes' SET mode_id= 'gg' where mode_id='w'"
                    self.curr.execute(sql)
                reboot_cursor()

            elif 'mode_single_letter_insert' in cmd:
                sql = "INSERT INTO 'modes' (mode_name, mode_id) VALUES(?, ?)"
                self.curr.execute(sql, ['testasdasd', 'pp'])

                self.curr.execute(cmd)
                reboot_cursor()

                with self.assertRaises(sqlite3.IntegrityError):
                    self.curr.execute(sql, ['test1b', 'mm'])
                reboot_cursor()

            elif 'mode_keep_if_in_use_updating' in cmd:
                sql = "UPDATE 'modes' SET mode_id= 'h' where mode_id='g'"
                self.curr.execute(sql)

                self.curr.execute(cmd)
                reboot_cursor()

                with self.assertRaises(sqlite3.IntegrityError):
                    sql = "UPDATE 'modes' SET mode_id= 'j' where mode_id='l'"
                    self.curr.execute(sql)
                reboot_cursor()

            elif 'mode_keep_if_in_use_deleting' in cmd:
                sql = "DELETE FROM 'modes' where mode_id='p'"
                self.curr.execute(sql)

                self.curr.execute(cmd)
                reboot_cursor()

                with self.assertRaises(sqlite3.IntegrityError):
                    sql = "DELETE FROM 'modes' where mode_id='c'"
                    self.curr.execute(sql)
                reboot_cursor()

            elif 'modes_on_links_update' in cmd:
                sql = "UPDATE 'links' SET modes= 'qwerty' where link_id=55"
                self.curr.execute(sql)

                self.curr.execute(cmd)
                reboot_cursor()

                with self.assertRaises(sqlite3.IntegrityError):
                    sql = "UPDATE 'links' SET modes= 'azerty' where link_id=56"
                    self.curr.execute(sql)
                reboot_cursor()

            elif 'modes_length_on_links_update' in cmd:
                sql = "UPDATE 'links' SET modes= '' where modes='wb'"
                self.curr.execute(sql)

                self.curr.execute(cmd)
                reboot_cursor()

                with self.assertRaises(sqlite3.IntegrityError):
                    sql = "UPDATE 'links' SET modes= '' where modes='bw'"
                    self.curr.execute(sql)
                reboot_cursor()

            elif 'modes_on_nodes_table_update_a_node' in cmd:
                sql = "UPDATE 'links' SET a_node= 1 where a_node=3"
                self.curr.execute(sql)

                sql = "SELECT modes from nodes where node_id=1"
                self.curr.execute(sql)
                i = self.curr.fetchone()[0]
                self.assertEqual(i, 'ct')

                self.curr.execute(cmd)
                reboot_cursor()

                k = ''
                for n in [2, 5]:
                    for f in ['a_node', 'b_node']:
                        self.curr.execute(f"SELECT modes from links where {f}={n}")
                        k += self.curr.fetchone()[0]

                existing = set(k)

                sql = "UPDATE 'links' SET a_node= 2 where a_node=5"
                self.curr.execute(sql)

                sql = "SELECT modes from nodes where node_id=2"
                self.curr.execute(sql)
                i = set(self.curr.fetchone()[0])
                self.assertEqual(i, existing)

            elif 'modes_on_nodes_table_update_b_node' in cmd:
                sql = "UPDATE 'links' SET b_node= 1 where b_node=3"
                self.curr.execute(sql)

                sql = "SELECT modes from nodes where node_id=1"
                self.curr.execute(sql)
                i = self.curr.fetchone()[0]
                self.assertEqual(i, 'ct')

                self.curr.execute(cmd)
                reboot_cursor()

                sql = "UPDATE 'links' SET b_node= 2 where b_node=4"
                self.curr.execute(sql)

                sql = "SELECT modes from nodes where node_id=2"
                self.curr.execute(sql)
                i = self.curr.fetchone()[0]
                self.assertEqual(i, 'ctw')

            elif 'modes_on_nodes_table_update_links_modes' in cmd:
                sql = "UPDATE 'links' SET modes= 'x' where a_node=24"
                self.curr.execute(sql)

                sql = "SELECT modes from nodes where node_id=24"
                self.curr.execute(sql)
                i = self.curr.fetchone()[0]
                self.assertEqual(i, 'c')

                self.curr.execute(cmd)
                reboot_cursor()

                sql = "UPDATE 'links' SET 'modes'= 'y' where a_node=24"
                self.curr.execute(sql)

                sql = "SELECT modes from nodes where node_id=24"
                self.curr.execute(sql)
                i = self.curr.fetchone()[0]
                self.assertIn('c', i)
                self.assertIn('y', i)

                sql = "UPDATE 'links' SET 'modes'= 'r' where b_node=24"
                self.curr.execute(sql)

                sql = "SELECT modes from nodes where node_id=24"
                self.curr.execute(sql)
                i = self.curr.fetchone()[0]
                self.assertIn('r', i)
                self.assertIn('y', i)

            elif 'modes_on_links_insert' in cmd:
                if self.rtree:
                    self.curr.execute('pragma table_info(links)')
                    f = self.curr.fetchall()
                    fields = {x[1]: x[0] for x in f}

                    sql = 'select * from links where link_id=10'
                    self.curr.execute(sql)
                    a = [x for x in self.curr.fetchone()]
                    a[fields['modes']] = 'as12'
                    a[fields['link_id']] = 1234
                    a[fields['a_node']] = 999
                    a[fields['b_node']] = 888
                    a[0] = 1234

                    idx = ','.join(['?'] * len(a))
                    self.curr.execute(f'insert into links values ({idx})', a)
                    self.curr.execute('delete from links where link_id=1234')

                    self.curr.execute(cmd)
                    reboot_cursor()

                    with self.assertRaises(sqlite3.IntegrityError):
                        self.curr.execute(f'insert into links values ({idx})', a)

            elif 'modes_length_on_links_insert' in cmd:
                if self.rtree:
                    self.curr.execute('pragma table_info(links)')
                    f = self.curr.fetchall()
                    fields = {x[1]: x[0] for x in f}

                    sql = 'select * from links where link_id=70'
                    self.curr.execute(sql)
                    a = [x for x in self.curr.fetchone()]
                    a[fields['modes']] = ''
                    a[fields['link_id']] = 4321
                    a[fields['a_node']] = 888
                    a[fields['b_node']] = 999
                    a[0] = 4321

                    idx = ','.join(['?'] * len(a))
                    self.curr.execute(f'insert into links values ({idx})', a)
                    self.curr.execute('delete from links where link_id=4321')

                    self.curr.execute(cmd)
                    reboot_cursor()

                    with self.assertRaises(sqlite3.IntegrityError):
                        self.curr.execute(f'insert into links values ({idx})', a)

            elif 'modes_on_nodes_table_update_nodes_modes' in cmd:
                self.curr.execute('select node_id, modes from nodes where length(modes)>0')
                dt = self.curr.fetchall()

                x = choice(dt)

                self.curr.execute(f'update nodes set modes="abcdefgq" where node_id={x[0]}')
                self.curr.execute(f'select node_id, modes from nodes where node_id={x[0]}')
                z = self.curr.fetchone()
                if z == x:
                    self.fail('Modes field on nodes layer is being preserved by unknown mechanism')

                self.curr.execute(cmd)
                reboot_cursor()

                y = choice(dt)
                while y == x:
                    y = choice(dt)

                # We try to force the change to make sure it was correctly filled to begin with
                self.curr.execute(f'update nodes set modes="hgfedcba" where node_id={y[0]}')

                self.curr.execute(f'select node_id, modes from nodes where node_id={y[0]}')
                k = self.curr.fetchone()

                self.curr.execute(f'update nodes set modes="abcdefgq" where node_id={y[0]}')

                self.curr.execute(f'select node_id, modes from nodes where node_id={y[0]}')
                z = self.curr.fetchone()

                self.assertEqual(z, k, 'Failed to preserve the information on modes for the nodes')

            else:
                if 'TRIGGER' in cmd.upper():
                    i = cmd.upper().find('TRIGGER')
                    e = cmd.upper().find('BEGIN')
                    self.fail(f'Missing test for triggers in modes table --> {cmd[i:e]}')
