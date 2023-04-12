import logging
import re
from os.path import join, dirname, realpath
from sqlite3 import Connection
from aequilibrae.project.database_connection import database_connection

req_link_flds = ["link_id", "a_node", "b_node", "direction", "distance", "modes", "link_type"]
req_node_flds = ["node_id", "is_centroid"]
protected_fields = ["ogc_fid", "geometry"]


def initialize_tables(project, db_type: str) -> None:
    conn = database_connection(db_type)
    create_base_tables(conn, project.logger, db_type)
    add_triggers(conn, project.logger, db_type)


def create_base_tables(conn: Connection, logger: logging.Logger, db_type: str) -> None:
    spec_folder = join(dirname(realpath(__file__)), "database_specification", db_type, "tables")
    with open(join(spec_folder, "table_list.txt"), "r") as file_list:
        all_tables = file_list.readlines()
    all_tables = [x.rstrip() for x in all_tables]
    for f in all_tables:
        qry_file = join(spec_folder, f"{f}.sql")
        run_queries_from_sql_file(conn, logger, qry_file)


def add_triggers(conn: Connection, logger: logging.Logger, db_type: str) -> None:
    """Adds consistency triggers to the project"""
    spec_folder = join(dirname(realpath(__file__)), "database_specification", db_type, "triggers")
    with open(join(spec_folder, "triggers_list.txt"), "r") as file_list:
        all_trigger_sets = file_list.readlines()
    all_trigger_sets = [x.rstrip() for x in all_trigger_sets]
    for f in all_trigger_sets:
        qry_file = join(spec_folder, f"{f}.sql")
        run_queries_from_sql_file(conn, logger, qry_file)


def remove_triggers(conn: Connection, logger: logging.Logger, db_type: str) -> None:
    spec_folder = join(dirname(realpath(__file__)), "database_specification", db_type, "triggers")
    with open(join(spec_folder, "triggers_list.txt"), "r") as file_list:
        all_trigger_sets = file_list.readlines()

    create_drop_regex = re.compile(r"create\s+trigger\s+(\w+)", flags=re.I)
    for table in all_trigger_sets:
        qry_file = join(spec_folder, f"{table.rstrip()}.sql")

        with open(qry_file, "r") as sql_file:
            query_list = sql_file.read()

        # Running one query/command at a time helps debugging in the case a particular command fails
        for cmd in query_list.split("--#"):
            for qry in cmd.split("\n"):
                if qry[:2] == "--":
                    continue
                while "  " in qry:
                    qry = qry.replace("  ", " ")

                m = re.search(create_drop_regex, qry)
                if m:
                    try:
                        conn.execute(f"drop trigger if exists {m.group(1).lower()}")
                    except Exception as e:
                        logger.error(f"Failed removing triggers table - > {e.args}")
                        logger.error(f"Point of failure - > {qry}")
        conn.commit()


def run_queries_from_sql_file(conn: Connection, logger: logging.Logger, qry_file: str) -> None:
    with open(qry_file, "r") as sql_file:
        query_list = sql_file.read()

    # Running one query/command at a time helps debugging in the case a particular command fails
    for cmd in query_list.split("--#"):
        try:
            conn.execute(cmd)
        except Exception as e:
            msg = f"Error running SQL command: {e.args}"
            logger.error(msg)
            logger.info(cmd)
            raise e
    conn.commit()
