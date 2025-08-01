import inspect
import sqlite3
from pathlib import Path
from random import choice
from warnings import warn

import pytest

from aequilibrae.project import about
from aequilibrae.utils.db_utils import read_and_close


@pytest.fixture
def project(empty_no_triggers_project):
    # Modes to add
    sql = "INSERT INTO modes (mode_name, mode_id) VALUES (?, ?);"
    with empty_no_triggers_project.db_connection as conn:
        for mid in ["p", "l", "g", "x", "y", "d", "k", "a", "r", "n", "m"]:
            conn.execute(sql, [f"mode_{mid}", mid])
    return empty_no_triggers_project


@pytest.fixture
def queries():
    qry = Path(inspect.getfile(about)).parent / "database_specification/network/triggers/modes_table_triggers.sql"
    with open(qry, "r") as sql_file:
        queries = sql_file.read()

    return list(queries.split("#"))


def get_query(queries, qry):
    for query in queries:
        if qry in query:
            return query
    raise FileNotFoundError("QUERY DOES NOT EXIST")


def check_rtree(project) -> bool:
    with project.db_connection as conn:
        try:
            conn.execute("SELECT rtreecheck('idx_nodes_geometry');")
        except Exception as e:
            warn(f"RTREE not available --> {e.args}")
            return False
        return True


def test_all_tests_considered(queries):
    test_names = [name for name in globals() if name.startswith("test_")]
    tests_added = [x[5:] for x in test_names]

    for trigger in queries:
        if "TRIGGER" in trigger.upper():
            found = [x for x in tests_added if x in trigger]
            if not found:
                pytest.fail(f"Trigger not tested. {trigger}")


def test_mode_single_letter_update(sioux_falls_example):
    with sioux_falls_example.db_connection as conn:
        sql = "UPDATE 'modes' SET mode_id= 'ttt' where mode_id='c'"
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(sql)


def test_mode_single_letter_insert(project):
    with project.db_connection as conn:
        sql = "INSERT INTO 'modes' (mode_name, mode_id) VALUES(?, ?)"
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(sql, ["testasdasd", "pp"])


def test_mode_keep_if_in_use_updating(no_triggers_test, queries):
    with no_triggers_test.db_connection as conn:
        conn.execute("UPDATE 'modes' SET mode_id= 'h' where mode_id='g'")
        conn.execute(get_query(queries, "aequilibrae_mode_keep_if_in_use_updating"))
        conn.commit()
        sql = "UPDATE 'modes' SET mode_id= 'j' where mode_id='c'"
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(sql)


def test_mode_keep_if_in_use_deleting(no_triggers_test, queries):
    with read_and_close(no_triggers_test.path_to_file) as conn:
        cmd = get_query(queries, "aequilibrae_mode_keep_if_in_use_deleting")
        sql = "DELETE FROM 'modes' where mode_id='p'"
        conn.execute(sql)
        conn.execute(cmd)
        sql = "DELETE FROM 'modes' where mode_id='c'"
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(sql)


def test_modes_on_links_update(no_triggers_test, queries):
    with read_and_close(no_triggers_test.path_to_file) as conn:
        cmd = get_query(queries, "aequilibrae_modes_on_links_update")
        sql = "UPDATE 'links' SET modes= 'qwerty' where link_id=55"
        conn.execute(sql)
        conn.execute(cmd)
        sql = "UPDATE 'links' SET modes= 'azerty' where link_id=56"
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(sql)


def test_modes_length_on_links_update(sioux_falls_test):
    with read_and_close(sioux_falls_test.path_to_file) as conn:
        sql = "UPDATE 'links' SET modes= '' where modes='c'"
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(sql)


def test_modes_on_nodes_table_update_a_node(no_triggers_test, queries):
    with read_and_close(no_triggers_test.path_to_file) as conn:
        cmd = get_query(queries, "aequilibrae_modes_on_nodes_table_update_a_node")
        sql = "UPDATE 'links' SET a_node= 1 where a_node=3"
        conn.execute(sql)
        i = conn.execute("SELECT modes from nodes where node_id=1").fetchone()[0]
        assert i == "ct"
        conn.execute(cmd)
        k = ""
        for n in [2, 5]:
            for f in ["a_node", "b_node"]:
                mode = conn.execute(f"SELECT modes from links where {f}={n}").fetchone()[0]
                k += mode
        existing = set(k)
        sql = "UPDATE 'links' SET a_node= 2 where a_node=5"
        conn.execute(sql)
        i = set(conn.execute("SELECT modes from nodes where node_id=2").fetchone()[0])
        assert i == existing


def test_modes_on_nodes_table_update_b_node(no_triggers_test, queries):
    with read_and_close(no_triggers_test.path_to_file) as conn:
        cmd = get_query(queries, "aequilibrae_modes_on_nodes_table_update_b_node")
        sql = "UPDATE 'links' SET b_node= 1 where b_node=3"
        conn.execute(sql)
        sql = "SELECT modes from nodes where node_id=1"
        i = conn.execute(sql).fetchone()[0]
        assert i == "ct"
        conn.execute(cmd)
        sql = "UPDATE 'links' SET b_node= 2 where b_node=4"
        conn.execute(sql)
        sql = "SELECT modes from nodes where node_id=2"
        i = conn.execute(sql).fetchone()[0]
        assert i == "ctw"


def test_modes_on_nodes_table_update_links_modes(no_triggers_test, queries):
    with read_and_close(no_triggers_test.path_to_file) as conn:
        cmd = get_query(queries, "aequilibrae_modes_on_nodes_table_update_links_modes")
        sql = "UPDATE 'links' SET modes= 'x' where a_node=24"
        conn.execute(sql)
        conn.commit()
        sql = "SELECT modes from nodes where node_id=24"
        i = conn.execute(sql).fetchone()[0]
        assert i == "c"
        conn.execute(cmd)
        sql = "UPDATE 'links' SET 'modes'= 'w' where a_node=24"
        conn.execute(sql)
        conn.commit()
        sql = "SELECT modes from nodes where node_id=24"
        i = conn.execute(sql).fetchone()[0]
        assert "c" in i and "w" in i
        sql = "UPDATE 'links' SET 'modes'= 'w' where b_node=24"
        conn.execute(sql)
        sql = "SELECT modes from nodes where node_id=24"
        i = conn.execute(sql).fetchone()[0]
        assert "w" == i


def test_modes_on_links_insert(no_triggers_test, queries):
    with read_and_close(no_triggers_test.path_to_file, spatial=True) as conn:
        cmd = get_query(queries, "aequilibrae_modes_on_links_insert")
        if check_rtree(no_triggers_test):
            fds = conn.execute("pragma table_info(links)").fetchall()
            fields = {x[1]: x[0] for x in fds}
            sql = "select * from links where link_id=10"
            a = list(conn.execute(sql).fetchone())
            a[fields["modes"]] = "as12"
            a[fields["link_id"]] = 1234
            a[fields["a_node"]] = 999
            a[fields["b_node"]] = 888
            a[0] = 1234
            idx = ",".join(["?"] * len(a))
            conn.execute(f"insert into links values ({idx})", a)
            conn.execute("delete from links where link_id=1234")
            conn.execute(cmd)
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(f"insert into links values ({idx})", a)


def test_modes_length_on_links_insert(sioux_falls_test):
    if not check_rtree(sioux_falls_test):
        return
    with sioux_falls_test.db_connection_spatial as conn:
        f = conn.execute("pragma table_info(links)").fetchall()
        fields = {x[1]: x[0] for x in f}
        sql = "select * from links where link_id=70"
        a = list(conn.execute(sql).fetchone())
        a[fields["modes"]] = ""
        a[fields["link_id"]] = 4321
        a[fields["a_node"]] = 888
        a[fields["b_node"]] = 999
        a[0] = 4321
        idx = ",".join(["?"] * len(a))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(f"insert into links values ({idx})", a)


def test_keep_at_least_one(no_triggers_test, queries):
    with no_triggers_test.db_connection as conn:
        cmd = get_query(queries, "aequilibrae_mode_keep_at_least_one")
        conn.execute("Delete from modes;")
        cnt = conn.execute("select count(*) from modes;").fetchone()[0]
        assert cnt == 0
        conn.execute('insert into modes(mode_id, mode_name) VALUES("k", "some_name")')
        conn.execute(cmd)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("Delete from modes;")


def test_modes_on_nodes_table_update_nodes_modes(no_triggers_test, queries):
    with no_triggers_test.db_connection as conn:
        cmd = get_query(queries, "aequilibrae_modes_on_nodes_table_update_nodes_modes")
        sql = "select node_id, modes from nodes where length(modes)>0"
        dt = conn.execute(sql).fetchall()
        x = choice(dt)
        conn.execute(f'update nodes set modes="abcdefgq" where node_id={x[0]}')
        sql = f"select node_id, modes from nodes where node_id={x[0]}"
        z = conn.execute(sql).fetchone()
        if z == x:
            pytest.fail("Modes field on nodes layer is being preserved by unknown mechanism")
        conn.execute(cmd)
        y = choice(dt)
        while y == x:
            y = choice(dt)
        conn.execute(f'update nodes set modes="hgfedcba" where node_id={y[0]}')
        sql = f"select node_id, modes from nodes where node_id={y[0]}"
        k = conn.execute(sql).fetchone()
        conn.execute(f'update nodes set modes="abcdefgq" where node_id={y[0]}')
        sql = f"select node_id, modes from nodes where node_id={y[0]}"
        z = conn.execute(sql).fetchone()
        assert z == k
