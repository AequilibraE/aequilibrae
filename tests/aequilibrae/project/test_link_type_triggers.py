import os
import sqlite3

import pytest


@pytest.fixture
def queries():
    root = os.path.dirname(os.path.realpath(__file__)).replace("tests", "")
    qry_file = os.path.join(root, "database_specification/network/triggers/link_type_table_triggers.sql")
    with open(qry_file, "r") as sql_file:
        queries = sql_file.read()
    return list(queries.split("#"))


def get_query(queries, qry: str) -> str:
    for query in queries:
        if qry in query:
            return query
    raise FileNotFoundError("QUERY DOES NOT EXIST")


def test_all_tests_considered(queries):
    import sys

    current_module = sys.modules[__name__]
    tests_added = dir(current_module)
    tests_added = [x[5:] for x in tests_added if x[:5] == "test_"]

    for trigger in queries:
        if "TRIGGER" in trigger.upper():
            found = [x for x in tests_added if x in trigger]
            if not found:
                pytest.fail(f"Trigger not tested. {trigger}")


def test_link_type_single_letter_update(sioux_falls_example):
    sql = "UPDATE 'link_types' SET link_type_id= 'ttt' where link_type_id='z'"
    with pytest.raises(sqlite3.IntegrityError):
        with sioux_falls_example.db_connection as conn:
            conn.execute(sql)


def test_link_type_single_letter_insert(empty_no_triggers_project):
    sql = "INSERT INTO 'link_types' (link_type, link_type_id) VALUES(?, ?)"
    with pytest.raises(sqlite3.IntegrityError):
        with empty_no_triggers_project.db_connection as conn:
            conn.execute(sql, ["test1b", "mm"])


def test_link_type_keep_if_in_use_updating(no_triggers_test, queries):
    with no_triggers_test.db_connection as conn:
        sql = "UPDATE 'link_types' SET link_type= 'ttt' where link_type='test'"
        conn.execute(sql)

        cmd = get_query(queries, "link_type_keep_if_in_use_updating")
        conn.execute(cmd)

        conn.commit()
        sql = "UPDATE 'link_types' SET link_type= 'QQQ' where link_type='test2'"
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(sql)


def test_link_type_keep_if_in_use_deleting(no_triggers_test, queries):
    with no_triggers_test.db_connection as conn:
        cmd = get_query(queries, "link_type_keep_if_in_use_deleting")

        sql = "DELETE FROM 'link_types' where link_type='test3'"
        conn.execute(sql)

        conn.execute(cmd)

        sql = "DELETE FROM 'link_types' where link_type='test4'"
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(sql)


def test_link_type_on_links_update(no_triggers_test, queries):
    with no_triggers_test.db_connection as conn:
        cmd = get_query(queries, "link_type_on_links_update")

        sql = "UPDATE 'links' SET link_type= 'rrr' where link_type='test3'"
        conn.execute(sql)

        conn.execute(cmd)

        sql = "UPDATE 'links' SET link_type= 'not_valid_type' where link_type='test4'"
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(sql)


def test_link_type_on_links_insert(no_triggers_test, queries):
    with no_triggers_test.db_connection_spatial as conn:
        cmd = get_query(queries, "link_type_on_links_insert")

        f = conn.execute("pragma table_info(links)").fetchall()
        fields = {x[1]: x[0] for x in f}

        sql = "select * from links where link_id=70"
        a = list(conn.execute(sql).fetchone())
        a[fields["link_type"]] = "something indeed silly123"
        a[fields["link_id"]] = 456789
        a[fields["a_node"]] = 777
        a[fields["b_node"]] = 999
        a[0] = 456789

        idx = ",".join(["?"] * len(a))
        conn.execute(f"insert into links values ({idx})", a)
        conn.execute("delete from links where link_id=456789")

        conn.execute(cmd)

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(f"insert into links values ({idx})", a)

        sql = "select link_type from link_types;"

        a[fields["link_type"]] = conn.execute(sql).fetchone()[0]
        conn.execute(f"insert into links values ({idx})", a)


def test_link_type_on_links_delete_protected_link_type(empty_no_triggers_project, queries):
    with empty_no_triggers_project.db_connection as conn:
        cmd = get_query(queries, "link_type_on_links_delete_protected_link_type")

        conn.execute(cmd)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute('delete from link_types where link_type_id="z"')

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute('delete from link_types where link_type_id="y"')


def test_link_type_id_keep_if_protected_type(empty_no_triggers_project, queries):
    with empty_no_triggers_project.db_connection as conn:
        cmd = get_query(queries, "link_type_id_keep_if_protected_type")

        conn.execute(cmd)

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute('update link_types set link_type_id="x" where link_type_id="y"')

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute('update link_types set link_type_id="x" where link_type_id="z"')


def test_link_type_keep_if_protected_type(empty_no_triggers_project, queries):
    with empty_no_triggers_project.db_connection as conn:
        cmd = get_query(queries, "link_type_keep_if_protected_type")
        conn.execute(cmd)

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute('update link_types set link_type="xsdfg" where link_type_id="z"')

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute('update link_types set link_type="xsdfg" where link_type_id="y"')


def test_link_type_on_nodes_table_update_nodes_link_type(no_triggers_test, queries):
    with no_triggers_test.db_connection as conn:
        cmd = get_query(queries, "link_type_on_nodes_table_update_nodes_link_type")
        conn.execute(cmd)

        conn.execute('update nodes set link_types="qwerrreyrtuyiuio" where node_id=1')

        lts = conn.execute("select link_types from nodes where node_id=1").fetchone()[0]

        assert lts == "etuw", "link_types was allowed to be corrupted in the nodes table"


def test_link_type_on_nodes_table_update_links_link_type(no_triggers_test, queries):
    with no_triggers_test.db_connection as conn:
        cmd = get_query(queries, "link_type_on_nodes_table_update_links_link_type")
        conn.execute(cmd)

        conn.execute('update links set link_type="test" where link_id=15')

        lts = conn.execute("select link_types from nodes where node_id=6").fetchone()[0]

        assert lts == "grtw", "link_types on nodes table not updated with new link type in the links"

        lts = conn.execute("select link_types from nodes where node_id=5").fetchone()[0]

        assert lts == "egrtw", "link_types was allowed to be corrupted in the nodes table"


def test_link_type_on_nodes_table_update_links_a_node(no_triggers_test, queries):
    with no_triggers_test.db_connection as conn:
        cmd = get_query(queries, "link_type_on_nodes_table_update_links_a_node")
        conn.execute(cmd)

        conn.execute("update links set a_node=1 where link_id=15")

        lts = conn.execute("select link_types from nodes where node_id=1").fetchone()[0]

        assert lts == "etuw", "link_types on nodes table not updated with new link type in the links"

        lts = conn.execute("select link_types from nodes where node_id=6").fetchone()[0]

        assert lts == "grw", "link_types was allowed to be corrupted in the nodes table"


def test_link_type_on_nodes_table_update_links_b_node(no_triggers_test, queries):
    with no_triggers_test.db_connection as conn:
        cmd = get_query(queries, "link_type_on_nodes_table_update_links_b_node")
        conn.execute(cmd)

        conn.execute("update links set b_node=1 where link_id=15")

        lts = conn.execute("select link_types from nodes where node_id=1").fetchone()[0]

        assert lts == "etuw", "link_types on nodes table not updated with new link type in the links"

        lts = conn.execute("select link_types from nodes where node_id=5").fetchone()[0]

        assert lts == "egrtw", "link_types was allowed to be corrupted in the nodes table"
