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
from scipy import sparse

from aequilibrae.project.database_connection import database_connection
from aequilibrae.context import get_active_project
import sqlite3

include 'hyperpath.pyx'


class HyperpathGenerating:
    tail = "a_node"
    head = "b_node"
    origin_column="orig_vert_idx",
    destination_column="dest_vert_idx",
    demand_column="demand",

    def __init__(
            self,
            graph,
            matrix,
            assignment_config: dict,
            check_edges: bool =False,
            check_demand: bool =False,
            threads: int = 0
    ):
        """A class for hyperpath generation.

        :Arguments:
            **graph** (:obj:`TransitGraph`): TransitGraph object

            **matrix** (:obj:`AequilibraEMatrix`): AequilbraE Matrix object for the demand

            **assignment_conffig** (:obj:`dict[str, str]`): Dictionary containing the `Time field` and `Frequency field` columns names.

            **check_edges** (:obj:`bool`): If True, check the validity of the edges (optional, default is False).

            **check_demand** (:obj:`bool`): If True, check the validity of the demand data (optional, default is False).

            **threads** (:obj:`int`): The number of threads to use for computation (optional, default is 0, using all available threads).
        """

        edges = graph.graph
        trav_time = assignment_config["Time field"]
        freq = assignment_config["Frequency field"]
        self._assignment_config = assignment_config
        self.__od_node_mapping = graph.od_node_mapping

        # This is a sparse representation of the AequilibraE Matrix object, the index is from od to od, *NOT* origin to destination. We'll use the od_node_mapping to convert index to node_id before the assignment
        self.__demand = sparse.coo_matrix(matrix, dtype=np.float64)

        self.check_edges = check_edges
        self.check_demand = check_demand
        self.threads = threads

        # load the edges
        if self.check_edges:
            self._check_edges(edges, self.tail, self.head, trav_time, freq)
        self._edges = edges[["link_id", self.tail, self.head, trav_time, freq]].copy(deep=True)
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
        self.vertex_count = self._edges[[self.tail, self.head]].max().max() + 1
        rs_indptr, _, rs_data = convert_graph_to_csc_uint32(self._edges, self.tail, self.head, data_col, self.vertex_count)
        self._indptr = rs_indptr.astype(np.uint32)
        self._edge_idx = rs_data.astype(np.uint32)

        # edge attributes
        self._trav_time = self._edges[trav_time].values.astype(DATATYPE_PY)
        self._freq = self._edges[freq].values.astype(DATATYPE_PY)
        self._tail = self._edges[self.tail].values.astype(np.uint32)
        self._head = self._edges[self.head].values.astype(np.uint32)

    def run(self, origin, destination, volume):
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

    def execute(self, threads=0):
        """Assigns demand to the edges of the graph.

        :Arguments:
            **threads** (:obj:`int`):The number of threads to use for computation (optional, default is 0, using all available threads).
        """

        # check the input demand paramater
        if not threads:
            threads = self.threads
        # if self.check_demand:
        #     self._check_demand(self.demand, self.origin_column, self.destination_column, self.demand_column)
        # self.demand = self.demand[self.demand[self.demand_column] > 0]

        # initialize the column storing the resulting edge volumes
        self._edges["volume"] = 0.0

        # travel time is computed but not saved into an array in the following
        self.u_i_vec = None

        if len(self.__od_node_mapping.columns) == 2:
            o_vert_ids = self.__od_node_mapping.iloc[self.__demand.row]["node_id"].values.astype(np.uint32)
            d_vert_ids = self.__od_node_mapping.iloc[self.__demand.col]["node_id"].values.astype(np.uint32)
        else:
            o_vert_ids = self.__od_node_mapping.iloc[self.__demand.row]["o_node_id"].values.astype(np.uint32)
            d_vert_ids = self.__od_node_mapping.iloc[self.__demand.col]["d_node_id"].values.astype(np.uint32)
        demand_vls = self.__demand.data

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

    def results(self):
        return self._edges[["link_id", "a_node", "b_node", "trav_time", "freq", "volume"]].copy(deep=True)
