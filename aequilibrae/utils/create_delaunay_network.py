from itertools import combinations
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

from aequilibrae.project.database_connection import database_connection

DELAUNAY_TABLE = 'delaunay_network'


def create_delaunay_network(source='zones', overwrite=False):
    """Creates a delaunay network based on the existing model

    Args:
        *source* (:obj:`str`, `Optional`): Source of the centroids/zones. Defaults to *zones*, but can be *network*
        *overwrite path* (:obj:`bool`, `Optional`): Whether we should overwrite am existing Delaunay Network.
        Defaults to False
        """

    if source not in ['zones', 'network']:
        raise ValueError("Source must be 'zones' or 'network'")

    conn = database_connection()

    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type ='table'", conn)
    if DELAUNAY_TABLE in tables.name.values:
        if not overwrite:
            raise ValueError("Delaunay network already exist. Use the overwrite flag to re-run it")
        conn.execute(f'DROP TABLE {DELAUNAY_TABLE}')
        conn.execute('delete from geometry_columns where f_table_name=?', [DELAUNAY_TABLE])
        conn.commit()

    zone_sql = 'select zone_id node_id, X(st_centroid(geometry)) x, Y(st_centroid(geometry)) y from zones'
    network_sql = 'select node_id, X(geometry) x, Y(geometry) y from nodes where is_centroid=1'

    points = pd.read_sql(zone_sql, conn) if source == 'zones' else pd.read_sql(network_sql, conn)
    dpoints = np.array(points[['x', 'y']])
    all_edges = Delaunay(np.array(dpoints)).simplices
    edges = []
    for triangle in all_edges:
        links = list(combinations(triangle, 2))
        for i in links:
            f, t = sorted(list(i))
            edges.append([points.at[f, 'node_id'], points.at[t, 'node_id']])

    edges = pd.DataFrame(edges)
    edges.drop_duplicates(inplace=True)
    edges.columns = ['a_node', 'b_node']
    edges = edges.assign(direction=0, distance=0, link_id=np.arange(edges.shape[0]) + 1)
    edges = edges[['link_id', 'direction', 'a_node', 'b_node', 'distance']]
    edges.to_sql(DELAUNAY_TABLE, conn, index=False)

    # Now we create the geometries for the delaunay triangulation
    conn.execute("select AddGeometryColumn( 'delaunay_network', 'geometry', 4326, 'LINESTRING', 'XY', 0);")
    conn.execute("CREATE UNIQUE INDEX unique_link_id_delaunay on delaunay_network(link_id)")

    node_geo_sql = '''INSERT INTO delaunay_network (link_id, geometry)
                         select lnk.link_id, MakeLine(nd.geometry, nf.geometry) from delaunay_network lnk
                                inner join nodes nd on lnk.a_node=nd.node_id
                                inner join nodes nf on lnk.b_node=nf.node_id
                       ON CONFLICT(link_id) DO UPDATE SET geometry=excluded.geometry'''

    zone_geo_sql = '''INSERT INTO delaunay_network (link_id, geometry)
                         select lnk.link_id, MakeLine(st_centroid(za.geometry), st_centroid(zb.geometry)) from delaunay_network lnk
                                inner join zones za on lnk.a_node=za.zone_id
                                inner join zones zb on lnk.b_node=zb.zone_id
                      ON CONFLICT(link_id) DO UPDATE SET geometry=excluded.geometry;'''

    sql = zone_geo_sql if source == 'zones' else node_geo_sql
    conn.execute(sql)
    # Updates link distance
    conn.execute('update delaunay_network set distance=GeodesicLength(geometry);')
    conn.commit()
