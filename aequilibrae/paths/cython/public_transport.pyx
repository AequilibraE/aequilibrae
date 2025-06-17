# cython: language_level=3

import multiprocessing
import socket
from pathlib import Path

import numpy as np
import pandas as pd

from aequilibrae.context import get_active_project
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.project.database_connection import database_path
from aequilibrae.utils.db_utils import commit_and_close

include 'hyperpath.pyx'

from typing import Union


class HyperpathGenerating:
    """A class for hyperpath generation.

    :Arguments:
        **edges** (:obj:`pandas.DataFrame`): The edges of the graph.

        **tail** (:obj:`str`): The column name for the tail of the edge (*Optional*, default is "tail").

        **head** (:obj:`str`): The column name for the head of the edge (*Optional*, default is "head").

        **trav_time** (:obj:`str`): The column name for the travel time of the edge
        (*Optional*, default is "trav_time").

        **freq** (:obj:`str`): The column name for the frequency of the edge (*Optional*, default is "freq").

        **check_edges** (:obj:`bool`): If True, check the validity of the edges (*Optional*, default is False).
    """

    def __init__(
            self,
            edges,
            tail="tail",
            head="head",
            trav_time="trav_time",
            freq="freq",
            check_edges=False,
            skim_cols = None,
            *,
            o_vert_ids=np.array([], dtype=np.int64),
            d_vert_ids=np.array([], dtype=np.int64),
            nodes_to_indices,
    ):
        skim_cols = self.check_skim_cols(skim_cols)

        # Take a copy as to not modify the users dataframe
        edges, skim_cols = self.compute_skim_cols(skim_cols, edges.copy(), trav_time)
        # load the edges
        if check_edges:
            self._check_edges(edges, tail, head, trav_time, freq, skim_cols)
        self._edges = edges[list(set([tail, head, trav_time, freq] + skim_cols))].copy(deep=True)
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

        self._o_vert_ids = o_vert_ids.astype(np.int64) # node_id for origin in the above taz_id
        self._d_vert_ids = d_vert_ids.astype(np.int64) # node_id for destination in the above taz_id

        self._nodes_to_indices = nodes_to_indices

        # map taz_id -> node index, removing origins and destinations where either is disconnected completely
        self._o_indices = nodes_to_indices[self._o_vert_ids]
        origin_mask = self._o_indices != -1

        # create a similar mapping for a destination to its origin index
        d_indices = nodes_to_indices[self._d_vert_ids]
        destination_mask = d_indices != -1

        d_indices = d_indices[origin_mask & destination_mask]
        o_indices = self._o_indices[origin_mask & destination_mask].astype("uint32")

        # We need to map each origin and destination index to the index of their respective taz_id. When centroid flow
        # is not blocked there is only "od" nodes, thus their indices are equal. When centroid flow is blocked, the
        # origin and destinations are unique nodes and thus have unique indices within the graph, thus there is no
        # overlap. We mask out the ODs that are completely disconnected.
        self._od_index_to_taz_index = np.full(max(o_indices.max(), d_indices.max()) + 1, -1, dtype="int64")
        tmp = np.arange(len(self._o_indices))[origin_mask & destination_mask]
        assert len(o_indices) == len(d_indices)
        self._od_index_to_taz_index[o_indices] = tmp
        self._od_index_to_taz_index[d_indices] = tmp

        self._o_indices = o_indices

        if self._skimming:
            self._is_travel_time = trav_time in skim_cols
            if self._is_travel_time:
                skim_cols.remove(trav_time)
            if not skim_cols:
                self._skim_cols = np.zeros((self._trav_time.shape[0], 1), dtype=DATATYPE_PY)
            else:
                self._skim_cols = self._edges.loc[:, skim_cols].values.astype(DATATYPE_PY)
                self._skim_cols = self._skim_cols.copy(order='C')

            self._skim_cols_names = skim_cols
            self._trav_time_name = trav_time

        else:
            self._skim_cols = np.zeros((self._trav_time.shape[0], 1), dtype=DATATYPE_PY)
            self._skim_cols_names = []
            self._is_travel_time = False

    def compute_skim_cols(self, skim_cols, edges: pd.DataFrame, trav_time: str):
        self._get_waiting_time = False
        edges_cols = set(edges.columns)
        skim_cols = set(skim_cols)

        discerete_link_types = {
            "boardings": "boarding",
            "alightings": "alighting",
            "inner_transfers": "inner_transfer",
            "outer_transfers": "outer_transfer",
            "transfers": ["inner_transfer", "outer_transfer"],
        }

        contig_link_types = {
            "on_board_trav_time": "on-board",
            "dwelling_time": "dwell",
            "egress_trav_time": "egress_connector",
            "access_trav_time": "access_connector",
            "walking_trav_time": "walking",
            "transfer_time": ["inner_transfer", "outer_transfer"],
            "in_vehicle_trav_time": ["on-board", "dwell"],
        }

        if any(
                item in skim_cols for item in discerete_link_types.keys() | contig_link_types.keys()
        ) and "link_type" not in edges_cols:
            raise ValueError("predefined skimming type requested but 'link_type' column not present on the graph")

        for name, col in discerete_link_types.items():
            if name in skim_cols and name not in edges_cols:
                if isinstance(col, list):
                    edges[name] = np.where(edges["link_type"].isin(col), 1, 0)
                else:
                    edges[name] = np.where(edges["link_type"] == col, 1, 0)

        if "waiting_time" in skim_cols and "waiting_time" not in edges_cols:
            skim_cols.remove("waiting_time")
            skim_cols = skim_cols | set([trav_time, "in_vehicle_trav_time", "egress_trav_time", "access_trav_time"])
            self._get_waiting_time = True

        for name, col in contig_link_types.items():
            if name in skim_cols and name not in edges_cols:
                if isinstance(col, list):
                    edges[name] = np.where(edges["link_type"].isin(col), edges[trav_time], 0)
                else:
                    edges[name] = np.where(edges["link_type"] == col, edges[trav_time], 0)

        return edges, list(skim_cols)

    def check_skim_cols(self, skim_cols: Union(list[str], tuple[str], set(str))):
        self._skimming = True
        if isinstance(skim_cols, (tuple, set)):
            skim_cols = list(skim_cols)

        if not skim_cols or not isinstance(skim_cols, list):
            skim_cols = []
            self._skimming = False

        return skim_cols

    def run(self, origin, destination, volume):
        self.assign(
            np.array([origin]),
            np.array([destination]),
            np.array([volume]),
            threads=1
        )

    def _check_vertex_idx(self, idx):
        assert isinstance(idx, int)
        assert idx >= 0
        assert idx < self.vertex_count

    def _check_volume(self, v):
        assert isinstance(v, float)
        assert v >= 0.0

    def _check_edges(self, edges, tail, head, trav_time, freq, skim_cols):
        if not isinstance(edges, pd.core.frame.DataFrame):
            raise TypeError("edges should be a pandas DataFrame")

        cols = [tail, head, trav_time, freq] + skim_cols

        for col in cols:
            if col not in edges:
                raise KeyError(f"edge column '{col}' not found in graph edges dataframe")

        if edges[cols].isna().any(axis=None):
            raise ValueError(
                " ".join(
                    [
                        f"edges[[{', '.join(map(str, cols))}]]",
                        "should not have any missing value",
                    ]
                )
            )

        for col in [tail, head]:
            if not pd.api.types.is_integer_dtype(edges[col].dtype):
                raise TypeError(f"column '{col}' should be of integer type")

        for col in [trav_time, freq] + skim_cols:
            if not pd.api.types.is_numeric_dtype(edges[col].dtype):
                raise TypeError(f"column '{col}' should be of numeric type")

            if edges[col].min() < 0.0:
                raise ValueError(f"column '{col}' should be nonnegative")

    def assign(
        self,
        origin_column,
        destination_column,
        demand_column,
        check_demand=False,
        threads=None
    ):
        """
        Assigns demand to the edges of the graph.

        Assumes the ``*_column`` arguments are provided as numpy arrays that form a COO sprase matrix.

        :Arguments:
            **origin_column** (:obj:`np.ndarray`): The column for the origin vertices (*Optional*, default is
              "orig_vert_idx").

            **destination_column** (:obj:`np.ndarray`): The column or the destination vertices (*Optional*, default is
              "dest_vert_idx").

            **demand_column** (:obj:`np.ndarray`): The column for the demand values (*Optional*, default is "demand").

            **check_demand** (:obj:`bool`): If True, check the validity of the demand data (*Optional*, default is
              ``False``).

            **threads** (:obj:`int`):The number of threads to use for computation (*Optional*, default is 0, using all
        available threads).
        """

        self.origin_column = origin_column.astype(np.uint32)
        self.destination_column = destination_column.astype(np.uint32)
        self.demand_column = demand_column.astype(DATATYPE_PY)
        # check the input demand parameter
        if check_demand:
            self._check_demand(origin_column, destination_column, demand_column)

        if threads is None:
            threads = 0  # Default to all threads

        # initialize the column storing the resulting edge volumes
        self._edges["volume"] = 0.0

        # travel time is computed but not saved into an array in the following
        self.u_i_vec = np.zeros(self.vertex_count, dtype=DATATYPE_PY)

        # get the list of all destinations, we use "rest of" for skimming
        destinations = np.unique(self.destination_column)
        if self._skimming:
            rest_of_destinations = self._d_vert_ids[
                np.isin(self._d_vert_ids, destinations, invert=True, assume_unique=True)
            ].astype("uint32")
        else:
            rest_of_destinations = np.array([], dtype="uint32")

        n_centroids = self._o_vert_ids.shape[0]
        n_skim_cols = len(self._skim_cols_names)
        skim_cols = self._skim_cols_names
        if self._is_travel_time:
            skim_cols = [self._trav_time_name] + skim_cols
            n_skim_cols = n_skim_cols + 1

        self.skim_matrix = np.zeros((n_centroids, n_centroids, n_skim_cols))

        compute_SF_in_parallel(
            self._indptr[:],
            self._edge_idx[:],
            self._trav_time[:],
            self._freq[:],
            self._tail[:],
            self._head[:],
            self.destination_column[:],
            destinations[:],
            rest_of_destinations[:],
            self.origin_column[:],
            self.demand_column[:],
            self._edges["volume"].values,
            self.vertex_count,
            self._edges["volume"].shape[0],
            (multiprocessing.cpu_count() if threads < 1 else threads),
            self._skim_cols[:],
            self.u_i_vec,
            self.skim_matrix,
            self._o_vert_ids[:],
            self._o_indices[:],
            self._od_index_to_taz_index[:],
            self._nodes_to_indices[:],
            self._skimming,
            self._is_travel_time,
            len(self._skim_cols_names)
        )

        if self._skimming:
            fmax = np.finfo(dtype="float64").max
            self.skim_matrix[self.skim_matrix == fmax] = 0.0
            if self._get_waiting_time:
                skim_matrix_dict = {}
                for i in range(n_skim_cols):
                    skim_matrix_dict[skim_cols[i]] = self.skim_matrix[:, :, i]

                skim_matrix_dict['waiting_time'] = (
                    skim_matrix_dict['trav_time']
                    - skim_matrix_dict['in_vehicle_trav_time']
                    - skim_matrix_dict['egress_trav_time']
                    - skim_matrix_dict['access_trav_time']
                )
                skim_cols = skim_cols + ['waiting_time']
                arr = np.dstack([skim_matrix_dict[k] for k in skim_cols])
            else:
                arr = self.skim_matrix

            self.skim_matrix = AequilibraeMatrix()
            self.skim_matrix.create_empty(zones=len(self._o_vert_ids), matrix_names=skim_cols)
            self.skim_matrix.index[:] = self._o_vert_ids
            self.skim_matrix.indices[:] = self.skim_matrix.indices.reshape(-1, 1)
            self.skim_matrix.computational_view()
            self.skim_matrix.matrices[:, :, :] = arr

        else:
            self.skim_matrix = None

    def _check_demand(self, origin_column, destination_column, demand_column):
        for col, col_name in zip(
                [origin_column, destination_column, demand_column],
                ["origin", "destination", "demand"]
        ):
            if not isinstance(col, (np.ndarray, np.generic)):
                raise TypeError(f"{col_name} should be a numpy array")

            if np.any(np.isnan(col)):
                raise ValueError(f"{col_name} should not have any missing value")

        for col, col_name in zip([origin_column, destination_column], ["origin", "destination"]):
            if not col.dtype == np.uint32:
                raise TypeError(f"column '{col_name}' should be of np.uint32")

        if not demand_column.dtype == np.float64:
            raise TypeError("demand column should be of np.float64 type")

        if demand_column.min() < 0.0:
            raise ValueError("demand column should be nonnegative")

    def info(self) -> dict:
        info = {
            "Algorithm": "Spiess, Heinz & Florian, Michael - Hyperpath generation",
            "Computer name": socket.gethostname(),
            "Procedure ID": self.procedure_id,
        }

        return info

    def save_results(self, table_name: str, keep_zero_flows=True, project=None) -> None:
        """
        Saves the assignment results to results_database.sqlite

        Method fails if table exists

        :Arguments:
            **table_name** (:obj:`str`): Name of the table to hold this assignment result

            **keep_zero_flows** (:obj:`bool`): Whether we should keep records for zero flows. Defaults to ``True``

            **project** (:obj:`Project`, *Optional*): Project we want to save the results to. Defaults to the active
              project
        """

        df = self._edges
        if not keep_zero_flows:
            df = df[df.volume > 0]

        if not project:
            project = project or get_active_project()
        with commit_and_close(Path(project.project_base_path) / "results_database.sqlite", missing_ok=True) as conn:
            df.to_sql(table_name, conn)

        rep = {"setup": self.info()}
        data = [table_name, "hyperpath assignment", self.procedure_id, str(rep), self.procedure_date, self.description]
        sql = """Insert into results(table_name, procedure, procedure_id, procedure_report, timestamp,
                                                                    description) Values(?,?,?,?,?,?)""",
        with commit_and_close(database_path("transit", project.project_base_path)) as conn:
            conn.execute(sql, data)
