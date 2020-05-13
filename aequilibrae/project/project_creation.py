import os
from aequilibrae import Parameters, logger
from aequilibrae.paths import release_version

meta_table = 'attributes_documentation'
req_link_flds = ["link_id", "a_node", "b_node", "direction", "distance", "modes", "link_type"]
req_node_flds = ["node_id", "is_centroid"]
protected_fields = ['ogc_fid', 'geometry']


def initialize_tables(conn) -> None:
    parameters = Parameters()._default
    create_about_table(conn)
    create_meta_table(conn)
    create_modes_table(conn, parameters)
    create_link_type_table(conn, parameters)
    create_network_tables(conn, parameters)
    populate_meta_extra_attributes(conn)
    add_triggers(conn)


def create_meta_table(conn) -> None:
    cursor = conn.cursor()
    create_query = f"""CREATE TABLE '{meta_table}' (name_table  VARCHAR NOT NULL,
                                                    attribute VARCHAR NOT NULL,
                                                    description VARCHAR);"""
    cursor.execute(create_query)
    conn.commit()


def create_modes_table(conn, parameters) -> None:
    create_query = """CREATE TABLE 'modes' (mode_name VARCHAR UNIQUE NOT NULL,
                                            mode_id VARCHAR PRIMARY KEY UNIQUE NOT NULL,
                                            description VARCHAR,
                                            alpha NUMERIC,
                                            beta NUMERIC,
                                            gamma NUMERIC,
                                            delta NUMERIC,
                                            epsilon NUMERIC,
                                            zeta NUMERIC,
                                            iota NUMERIC,
                                            sigma NUMERIC,
                                            phi NUMERIC,
                                            tau NUMERIC);"""
    cursor = conn.cursor()
    cursor.execute(create_query)
    modes = parameters["network"]["modes"]

    for mode in modes:
        nm = list(mode.keys())[0]
        descr = mode[nm]["description"]
        mode_id = mode[nm]["letter"]
        par = [f'"{p}"' for p in [nm, mode_id, descr]]
        par = ",".join(par)
        sql = f"INSERT INTO 'modes' (mode_name, mode_id, description) VALUES({par})"
        cursor.execute(sql)
    conn.commit()


def create_link_type_table(conn, parameters) -> None:
    create_query = """CREATE TABLE 'link_types' (link_type VARCHAR PRIMARY KEY UNIQUE NOT NULL,
                                                 link_type_id VARCHAR UNIQUE NOT NULL,
                                                 description VARCHAR,
                                                 lanes NUMERIC,
                                                 lane_capacity NUMERIC,
                                                 alpha NUMERIC,
                                                 beta NUMERIC,
                                                 gamma NUMERIC,
                                                 delta NUMERIC,
                                                 epsilon NUMERIC,
                                                 zeta NUMERIC,
                                                 iota NUMERIC,
                                                 sigma NUMERIC,
                                                 phi NUMERIC,
                                                 tau NUMERIC);"""

    cursor = conn.cursor()
    cursor.execute(create_query)

    link_types = parameters["network"]["links"]["link_types"]
    sql = "INSERT INTO 'link_types' (link_type, link_type_id, description, lanes, lane_capacity) VALUES(?, ?, ?, ?, ?)"
    for lt in link_types:
        nm = list(lt.keys())[0]
        args = (nm, lt[nm]["link_type_id"], lt[nm]["description"], lt[nm]["lanes"], lt[nm]["lane_capacity"])

        cursor.execute(sql, args)

    conn.commit()


def populate_meta_extra_attributes(conn) -> None:
    fields = []
    fields.append(['link_types', 'link_type', 'Link type name. E.g. arterial, or connector'])
    fields.append(['link_types', 'link_type_id', 'Single letter identifying the mode. E.g. a, for arterial'])
    fields.append(['link_types', 'description',
                   'Description of the same. E.g. Arterials are streets like AequilibraE Avenue'])
    fields.append(['link_types', 'lanes', 'Default number of lanes in each direction. E.g. 2'])
    fields.append(['link_types', 'lane_capacity', 'Default vehicle capacity per lane. E.g.  900'])

    fields.append(['modes', 'mode_name', 'Link type name. E.g. arterial, or connector'])
    fields.append(['modes', 'mode_id', 'Single letter identifying the mode. E.g. a, for arterial'])
    fields.append(
        ['modes', 'description', 'Description of the same. E.g. Arterials are streets like AequilibraE Avenue'])

    extra_keys = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'iota', 'sigma', 'phi', 'tau']
    extra_keys = [[x, 'Available for user convenience'] for x in extra_keys]

    cursor = conn.cursor()
    for table_name, f, d in fields:
        sql = f"INSERT INTO '{meta_table}' (name_table, attribute, description) VALUES('{table_name}','{f}', '{d}')"
        cursor.execute(sql)

    for table_name in ['link_types', 'modes']:
        for f, d in extra_keys:
            sql = f"INSERT INTO '{meta_table}' (name_table, attribute, description) VALUES('{table_name}','{f}', '{d}')"
            cursor.execute(sql)
    conn.commit()


def create_network_tables(conn, parameters) -> None:
    """Creates empty network tables for future filling"""
    curr = conn.cursor()
    # Create the links table
    fields = parameters["network"]["links"]["fields"]

    sql = """CREATE TABLE 'links' (
                      ogc_fid INTEGER PRIMARY KEY,
                      link_id INTEGER UNIQUE,
                      a_node INTEGER,
                      b_node INTEGER,
                      direction INTEGER NOT NULL DEFAULT 0,
                      distance NUMERIC,
                      modes TEXT NOT NULL,
                      link_type TEXT REFERENCES link_types(link_type) ON UPDATE RESTRICT ON DELETE RESTRICT,
                      {});"""

    flds = fields["one-way"]

    # returns first key in the dictionary
    def fkey(f):
        return list(f.keys())[0]

    owlf = ["{} {}".format(fkey(f), f[fkey(f)]["type"]) for f in flds if fkey(f).lower() not in req_link_flds]

    flds = fields["two-way"]
    twlf = []
    for f in flds:
        nm = fkey(f)
        tp = f[nm]["type"]
        twlf.extend([f"{nm}_ab {tp}", f"{nm}_ba {tp}"])

    link_fields = owlf + twlf

    if link_fields:
        sql = sql.format(",".join(link_fields))
    else:
        sql = sql.format("")

    curr.execute(sql)

    sql = """CREATE TABLE 'nodes' (ogc_fid INTEGER PRIMARY KEY,
                             node_id INTEGER UNIQUE NOT NULL,
                             is_centroid INTEGER NOT NULL DEFAULT 0,
                             modes VARCHAR,
                             link_types VARCHAR {});"""

    flds = parameters["network"]["nodes"]["fields"]
    ndflds = [f"{fkey(f)} {f[fkey(f)]['type']}" for f in flds if fkey(f).lower() not in req_node_flds]

    if ndflds:
        sql = sql.format("," + ",".join(ndflds))
    else:
        sql = sql.format("")
    curr.execute(sql)

    curr.execute("""SELECT AddGeometryColumn( 'links', 'geometry', 4326, 'LINESTRING', 'XY' )""")
    curr.execute("""SELECT AddGeometryColumn( 'nodes', 'geometry', 4326, 'POINT', 'XY' )""")
    conn.commit()


def add_triggers(conn) -> None:
    """Adds consistency triggers to the project"""
    add_network_triggers(conn)
    add_mode_triggers(conn)
    add_link_type_triggers(conn)


def add_network_triggers(conn) -> None:
    logger.info("Adding network triggers")
    add_trigger_from_file(conn, "network_triggers.sql")


def add_mode_triggers(conn) -> None:
    logger.info("Adding mode table triggers")
    add_trigger_from_file(conn, "modes_table_triggers.sql")


def add_link_type_triggers(conn) -> None:
    logger.info("Adding link type table triggers")
    add_trigger_from_file(conn, "link_type_table_triggers.sql")


def add_trigger_from_file(conn, qry_file: str) -> None:
    qry_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "network/database_triggers", qry_file)
    curr = conn.cursor()
    sql_file = open(qry_file, "r")
    query_list = sql_file.read()
    sql_file.close()

    # Run one query/command at a time
    for cmd in query_list.split("#"):
        try:
            curr.execute(cmd)
        except Exception as e:
            msg = f"Error creating trigger: {e.args}"
            logger.error(msg)
            logger.info(cmd)
    conn.commit()


def create_about_table(conn) -> None:
    create_query = """CREATE TABLE 'about' (infoname VARCHAR UNIQUE NOT NULL,
                                            infovalue VARCHAR);"""
    cursor = conn.cursor()
    cursor.execute(create_query)

    sql = "INSERT INTO 'about' (infoname) VALUES(?)"
    fields = ['model_name', 'region', 'description', 'author', 'license', 'model_version', 'aequilibrae_version']
    for lt in fields:
        cursor.execute(sql, [lt])

    cursor.execute(f"UPDATE 'about' set infovalue='{release_version}' where infoname='aequilibrae_version'")
