import sqlite3
import uuid
from itertools import combinations
from os.path import join

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph, TrafficClass, TrafficAssignment

DELAUNAY_TABLE = "delaunay_network"


class DelaunayAnalysis:
    def __init__(self, project):
        """Start a Delaunay analysis

        :Arguments:
            **project** (:obj:`Project`): The Project to connect to
        """

        self.project = project
        self.procedure_id = uuid.uuid4().hex

    def create_network(self, source="zones", overwrite=False):
        """Creates a delaunay network based on the existing model

        :Arguments:
            **source** (:obj:`str`, optional): Source of the centroids/zones. Either ``zones`` or ``network``. Default ``zones``

            **overwrite path** (:obj:`bool`, optional): Whether to should overwrite an existing Delaunay Network. Default ``False``

        """

        if source not in ["zones", "network"]:
            raise ValueError("Source must be 'zones' or 'network'")

        conn = self.project.connect()

        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type ='table'", conn)
        if DELAUNAY_TABLE in tables.name.values:
            if not overwrite:
                raise ValueError("Delaunay network already exist. Use the overwrite flag to re-run it")
            conn.execute(f"DROP TABLE {DELAUNAY_TABLE}")
            conn.execute("delete from geometry_columns where f_table_name=?", [DELAUNAY_TABLE])
            conn.commit()

        zone_sql = "select zone_id node_id, X(st_centroid(geometry)) x, Y(st_centroid(geometry)) y from zones"
        network_sql = "select node_id, X(geometry) x, Y(geometry) y from nodes where is_centroid=1"

        points = pd.read_sql(zone_sql, conn) if source == "zones" else pd.read_sql(network_sql, conn)
        dpoints = np.array(points[["x", "y"]])
        all_edges = Delaunay(np.array(dpoints)).simplices
        edges = []
        for triangle in all_edges:
            links = list(combinations(triangle, 2))
            for i in links:
                f, t = sorted(list(i))
                edges.append([points.at[f, "node_id"], points.at[t, "node_id"]])

        edges = pd.DataFrame(edges)
        edges.drop_duplicates(inplace=True)
        edges.columns = ["a_node", "b_node"]
        edges = edges.assign(direction=0, distance=0, link_id=np.arange(edges.shape[0]) + 1)
        edges = edges[["link_id", "direction", "a_node", "b_node", "distance"]]
        edges.to_sql(DELAUNAY_TABLE, conn, index=False)

        # Now we create the geometries for the delaunay triangulation
        conn.execute(f"select AddGeometryColumn( '{DELAUNAY_TABLE}', 'geometry', 4326, 'LINESTRING', 'XY', 0);")
        conn.execute("CREATE UNIQUE INDEX unique_link_id_delaunay on delaunay_network(link_id)")

        node_geo_sql = """INSERT INTO delaunay_network (link_id, geometry)
                             select lnk.link_id, MakeLine(nd.geometry, nf.geometry) from delaunay_network lnk
                                    inner join nodes nd on lnk.a_node=nd.node_id
                                    inner join nodes nf on lnk.b_node=nf.node_id
                           ON CONFLICT(link_id) DO UPDATE SET geometry=excluded.geometry"""

        zone_geo_sql = """INSERT INTO delaunay_network (link_id, geometry)
                             select lnk.link_id, MakeLine(st_centroid(za.geometry), st_centroid(zb.geometry)) from delaunay_network lnk
                                    inner join zones za on lnk.a_node=za.zone_id
                                    inner join zones zb on lnk.b_node=zb.zone_id
                          ON CONFLICT(link_id) DO UPDATE SET geometry=excluded.geometry;"""

        sql = zone_geo_sql if source == "zones" else node_geo_sql
        conn.execute(sql)
        # Updates link distance
        conn.execute("update delaunay_network set distance=GeodesicLength(geometry);")
        conn.commit()
        conn.close()

    def assign_matrix(self, matrix: AequilibraeMatrix, result_name: str):
        conn = self.project.connect()

        sql = f"select link_id, direction, a_node, b_node, distance, 1 capacity from {DELAUNAY_TABLE}"

        df = pd.read_sql(sql, conn)
        centroids = np.array(np.unique(np.hstack((df.a_node.values, df.b_node.values))), int)

        g = Graph()
        g.mode = "delaunay"
        g.network = df
        g.prepare_graph(centroids)
        g.set_blocked_centroid_flows(True)

        tc = TrafficClass("delaunay", g, matrix)
        ta = TrafficAssignment(self.project)
        ta.set_classes([tc])
        ta.set_time_field("distance")
        ta.set_capacity_field("capacity")
        ta.set_vdf("BPR")
        ta.set_vdf_parameters({"alpha": 0, "beta": 1.0})
        ta.set_algorithm("all-or-nothing")
        ta.execute()

        report = {"setup": str(ta.info())}
        data = [result_name, "Delaunay assignment", self.procedure_id, str(report), ta.procedure_date, ""]
        conn.execute(
            """Insert into results(table_name, procedure, procedure_id, procedure_report, timestamp,
                                            description) Values(?,?,?,?,?,?)""",
            data,
        )
        conn.commit()
        conn.close()

        cols = []
        for x in matrix.view_names:
            cols.extend([f"{x}_ab", f"{x}_ba", f"{x}_tot"])
        df = ta.results()[cols]

        conn = sqlite3.connect(join(self.project.project_base_path, "results_database.sqlite"))
        df.to_sql(result_name, conn)
        conn.close()
