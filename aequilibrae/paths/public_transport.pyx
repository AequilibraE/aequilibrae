# cython: language_level=3

import pandas as pd
import os
from uuid import uuid4
from datetime import datetime
import socket

import numpy as np
import multiprocessing
cimport numpy as cnp
cimport openmp

from aequilibrae.project.database_connection import database_connection
from aequilibrae.context import get_active_project
import sqlite3

include 'hyperpath.pyx'


class HyperpathGenerating:
    def __init__(self, edges, tail="tail", head="head", trav_time="trav_time", freq="freq", check_edges=False):
        # load the edges
        if check_edges:
            self._check_edges(edges, tail, head, trav_time, freq)
        self._edges = edges[["link_id", tail, head, trav_time, freq]].copy(deep=True)
        self.edge_count = len(self._edges)

        # remove inf values if any, and values close to zero
        self._edges[trav_time] = np.where(
            self._edges[trav_time] > DATATYPE_INF_PY, DATATYPE_INF_PY, self._edges[trav_time]
        )
        self._edges[trav_time] = np.where(
            self._edges[trav_time] < A_VERY_SMALL_TIME_INTERVAL_PY,
            A_VERY_SMALL_TIME_INTERVAL_PY,
            self._edges[trav_time],
        )
        self._edges[freq] = np.where(self._edges[freq] > INF_FREQ_PY, INF_FREQ_PY, self._edges[freq])
        self._edges[freq] = np.where(self._edges[freq] < MIN_FREQ_PY, MIN_FREQ_PY, self._edges[freq])

        # create an edge index column
        self._edges = self._edges.reset_index(drop=True)
        data_col = "edge_idx"
        self._edges[data_col] = self._edges.index

        # convert to CSC format
        self.vertex_count = self._edges[[tail, head]].max().max() + 1
        rs_indptr, _, rs_data = convert_graph_to_csc_uint32(self._edges, tail, head, data_col, self.vertex_count)
        self._indptr = rs_indptr.astype(np.uint32)
        self._edge_idx = rs_data.astype(np.uint32)

        # edge attributes
        self._trav_time = self._edges[trav_time].values.astype(DATATYPE_PY)
        self._freq = self._edges[freq].values.astype(DATATYPE_PY)
        self._tail = self._edges[tail].values.astype(np.uint32)
        self._head = self._edges[head].values.astype(np.uint32)

        self.procedure_id = uuid4().hex
        self.procedure_date = str(datetime.today())
        self.description = "Hyperpath"

    def run(self, origin, destination, volume, return_inf=False):
        # column storing the resulting edge volumes
        self._edges["volume"] = 0.0
        self.u_i_vec = None

        # input check
        if type(origin) is not list:
            origin = [origin]
        if type(volume) is not list:
            volume = [volume]
        assert len(origin) == len(volume)

        for i, item in enumerate(origin):
            self._check_vertex_idx(item)
            self._check_volume(volume[i])
        self._check_vertex_idx(destination)
        assert isinstance(return_inf, bool)

        o_vert_ids = np.array(origin, dtype=np.uint32)
        d_vert_ids = np.array([destination], dtype=np.uint32)
        demand_vls = np.array(volume, dtype=DATATYPE_PY)

        destination_vertex_indices = d_vert_ids  # Only one index allowed so must be unique

        cdef cnp.float64_t *u_i_vec

        u_i_vec = compute_SF_in_parallel(
            self._indptr[:],
            self._edge_idx[:],
            self._trav_time[:],
            self._freq[:],
            self._tail[:],
            self._head[:],
            d_vert_ids[:],
            destination_vertex_indices[:],
            o_vert_ids[:],
            demand_vls[:],
            self._edges["volume"].values,
            True,
            self.vertex_count,
            self._edges["volume"].shape[0],
            1,  # Single destination so no reason to parallelise
        )

        if u_i_vec != NULL:
            self.u_i_vec = np.asarray(<cnp.float64_t[:self.vertex_count]> u_i_vec)


    def _check_vertex_idx(self, idx):
        assert isinstance(idx, int)
        assert idx >= 0
        assert idx < self.vertex_count

    def _check_volume(self, v):
        assert isinstance(v, float)
        assert v >= 0.0

    def _check_edges(self, edges, tail, head, trav_time, freq):
        if type(edges) != pd.core.frame.DataFrame:
            raise TypeError("edges should be a pandas DataFrame")

        for col in [tail, head, trav_time, freq]:
            if col not in edges:
                raise KeyError(f"edge column '{col}' not found in graph edges dataframe")

        if edges[[tail, head, trav_time, freq]].isna().any().any():
            raise ValueError(
                " ".join(
                    [
                        f"edges[[{tail}, {head}, {trav_time}, {freq}]] ",
                        "should not have any missing value",
                    ]
                )
            )

        for col in [tail, head]:
            if not pd.api.types.is_integer_dtype(edges[col].dtype):
                raise TypeError(f"column '{col}' should be of integer type")

        for col in [trav_time, freq]:
            if not pd.api.types.is_numeric_dtype(edges[col].dtype):
                raise TypeError(f"column '{col}' should be of numeric type")

            if edges[col].min() < 0.0:
                raise ValueError(f"column '{col}' should be nonnegative")

    def assign(
        self,
        demand,
        origin_column="orig_vert_idx",
        destination_column="dest_vert_idx",
        demand_column="demand",
        check_demand=False,
        threads=0
    ):
        # check the input demand paramater
        if check_demand:
            self._check_demand(demand, origin_column, destination_column, demand_column)
        demand = demand[demand[demand_column] > 0]

        # initialize the column storing the resulting edge volumes
        self._edges["volume"] = 0.0

        # travel time is computed but not saved into an array in the following
        self.u_i_vec = None

        o_vert_ids = demand[origin_column].values.astype(np.uint32)
        d_vert_ids = demand[destination_column].values.astype(np.uint32)
        demand_vls = demand[demand_column].values.astype(np.float64)

        # get the list of all destinations
        destination_vertex_indices = np.unique(d_vert_ids)

        compute_SF_in_parallel(
            self._indptr[:],
            self._edge_idx[:],
            self._trav_time[:],
            self._freq[:],
            self._tail[:],
            self._head[:],
            d_vert_ids[:],
            destination_vertex_indices[:],
            o_vert_ids[:],
            demand_vls[:],
            self._edges["volume"].values,
            False,
            self.vertex_count,
            self._edges["volume"].shape[0],
            (multiprocessing.cpu_count() if threads < 1 else threads)
        )

    def _check_demand(self, demand, origin_column, destination_column, demand_column):
        if type(demand) != pd.core.frame.DataFrame:
            raise TypeError("demand should be a pandas DataFrame")

        for col in [origin_column, destination_column, demand_column]:
            if col not in demand:
                raise KeyError(f"demand column '{col}' not found in demand dataframe")

        if demand[[origin_column, destination_column, demand_column]].isna().any().any():
            raise ValueError(
                " ".join(
                    [
                        f"demand[[{origin_column}, {destination_column}, {demand_column}]] ",
                        "should not have any missing value",
                    ]
                )
            )

        for col in [origin_column, destination_column]:
            if not pd.api.types.is_integer_dtype(demand[col].dtype):
                raise TypeError(f"column '{col}' should be of integer type")

        col = demand_column

        if not pd.api.types.is_numeric_dtype(demand[col].dtype):
            raise TypeError(f"column '{col}' should be of numeric type")

        if demand[col].min() < 0.0:
            raise ValueError(f"column '{col}' should be nonnegative")

    def info(self) -> dict:
        info = {
            "Algorithm": "Spiess, Heinz & Florian, Michael Hyperpath generation",
            "Computer name": socket.gethostname(),
            "Procedure ID": self.procedure_id,
        }

        return info

    def save_results(self, table_name: str, keep_zero_flows=True, project=None) -> None:
        """Saves the assignment results to results_database.sqlite

        Method fails if table exists

        :Arguments:
            **table_name** (:obj:`str`): Name of the table to hold this assignment result
            **keep_zero_flows** (:obj:`bool`): Whether we should keep records for zero flows. Defaults to True
            **project** (:obj:`Project`, Optional): Project we want to save the results to. Defaults to the active project
        """

        df = self._edges
        if not keep_zero_flows:
            df = df[df.volume > 0]

        if not project:
            project = project or get_active_project()
        conn = sqlite3.connect(os.path.join(project.project_base_path, "results_database.sqlite"))
        df.to_sql(table_name, conn, index=False)
        conn.close()

        conn = database_connection("transit", project.project_base_path)
        report = {"setup": self.info()}
        data = [table_name, "hyperpath assignment", self.procedure_id, str(report), self.procedure_date, self.description]
        conn.execute(
            """Insert into results(table_name, procedure, procedure_id, procedure_report, timestamp,
                                            description) Values(?,?,?,?,?,?)""",
            data,
        )
        conn.commit()
        conn.close()

