# cython: language_level=3

from aequilibrae.utils.cython.openmp_helper import omp_get_max_threads
import socket
import logging
import json

import numpy as np
import pandas as pd

from aequilibrae.context import get_active_project
from aequilibrae.matrix.cython.dense_matrix cimport DenseMatrix
from aequilibrae.paths.cython.demand import validate_demand, choose_origins

from aequilibrae.utils.cython.bridge cimport Bridge

include 'hyperpath.pyx'

from typing import Union

logger = logging.getLogger(__name__)


class HyperpathGenerating:
    """A class for hyperpath generation.

    :Arguments:
        **edges** (:obj:`pandas.DataFrame`): The edges of the graph.

        **tail** (:obj:`str`, optional): The column name for the tail of the edge. Default is "tail".

        **head** (:obj:`str`, optional): The column name for the head of the edge. Default is "head".

        **trav_time** (:obj:`str`, optional): The column name for the travel time of the edge.
        Default is "trav_time".

        **freq** (:obj:`str`, optional): The column name for the frequency of the edge. Default is "freq".

        **check_edges** (:obj:`bool`, optional): If ``True``, check the validity of the edges.
        Default is ``False``.
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

        self._o_vert_ids = o_vert_ids.astype(np.int64)  # node_id for origin in the above taz_id
        self._d_vert_ids = d_vert_ids.astype(np.int64)  # node_id for destination in the above taz_id

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

        # Build derived skim columns first and assign once: pandas 3's chained-assignment check
        derived_cols = {}
        for name, col in discerete_link_types.items():
            if name in skim_cols and name not in edges_cols:
                if isinstance(col, list):
                    derived_cols[name] = np.where(edges["link_type"].isin(col), 1, 0)
                else:
                    derived_cols[name] = np.where(edges["link_type"] == col, 1, 0)

        if "waiting_time" in skim_cols and "waiting_time" not in edges_cols:
            skim_cols.remove("waiting_time")
            skim_cols = skim_cols | set([trav_time, "in_vehicle_trav_time", "egress_trav_time", "access_trav_time"])
            self._get_waiting_time = True

        for name, col in contig_link_types.items():
            if name in skim_cols and name not in edges_cols:
                if isinstance(col, list):
                    derived_cols[name] = np.where(edges["link_type"].isin(col), edges[trav_time], 0)
                else:
                    derived_cols[name] = np.where(edges["link_type"] == col, edges[trav_time], 0)

        if derived_cols:
            edges = edges.assign(**derived_cols)

        return edges, list(skim_cols)

    def check_skim_cols(self, skim_cols: Union(list[str], tuple[str], set(str))):
        self._skimming = True
        if isinstance(skim_cols, (tuple, set)):
            skim_cols = list(skim_cols)

        if not skim_cols or not isinstance(skim_cols, list):
            skim_cols = []
            self._skimming = False

        return skim_cols

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

    def assign(self, demand, threads=None):
        """
        Assigns demand to the edges of the graph.

        Demand is packed in [destinations, origins, fields] order, with one
        field.  Pack user-facing matrices with ``transpose=True`` before
        assigning.  Zone positions follow ``o_vert_ids`` and ``d_vert_ids``
        supplied at construction.  Skims are returned in a packed dense
        ``skim_matrix`` in [origins, destinations, fields] order.

        :Arguments:
            **demand** (:obj:`DenseMatrix` or :obj:`SparseMatrix`): Finite, non-negative float64 demand.

            **threads** (:obj:`int`, optional): The number of threads to use for computation.
            Default is ``0`` (using all available threads).
        """
        n_centroids = self._o_vert_ids.shape[0]
        shape = validate_demand(demand, n_centroids)
        if shape != (n_centroids, n_centroids, 1) or len(self._d_vert_ids) != n_centroids:
            raise ValueError("Transit assignment requires exactly one demand matrix matching the graph zones")

        cdef DenseMatrix skims
        cdef Bridge bridge

        if threads is None:
            threads = 0  # Default to all threads

        # Pandas 3+ can expose read-only buffers; use a writable array for Cython and assign back after.
        volume = np.zeros(self._edges.shape[0], dtype=DATATYPE_PY)

        # travel time is computed but not saved into an array in the following
        self.u_i_vec = np.zeros(self.vertex_count, dtype=DATATYPE_PY)

        # Transposed demand has one contiguous row per destination. Skimming also
        # visits destinations without demand, without constructing artificial OD entries.
        destinations = choose_origins(None, demand, self._skimming)

        skim_cols = list(self._skim_cols_names)
        if self._is_travel_time:
            skim_cols = [self._trav_time_name] + skim_cols

        if self._get_waiting_time:
            skim_cols.append("waiting_time")

        skims = DenseMatrix(n_centroids if self._skimming else 0, skim_cols, fill_value=0.0)
        skim_values = np.asarray(skims._values)

        with Bridge(logger) as bridge:
            compute_SF_in_parallel(
                indptr_view=self._indptr,
                edge_idx_view=self._edge_idx,
                trav_time_view=self._trav_time,
                freq_view=self._freq,
                tail_view=self._tail,
                head_view=self._head,
                demand=demand,
                destination_zones=destinations,
                o_vert_ids_view=self._o_vert_ids,
                d_vert_ids_view=self._d_vert_ids,
                edge_volume_view=volume,
                vertex_count=self.vertex_count,
                edge_count=volume.shape[0],
                num_threads=omp_get_max_threads() if threads < 1 else threads,
                skim_col_view=self._skim_cols,
                u_i_vec_view=self.u_i_vec,
                skim_matrix=skim_values,
                o_indices=self._o_indices,
                od_index_to_taz_index=self._od_index_to_taz_index,
                nodes_to_indices=self._nodes_to_indices,
                skimming=self._skimming,
                is_travel_time=self._is_travel_time,
                n_skim_cols=len(self._skim_cols_names),
                bridge=bridge,
            )

        self._edges["volume"] = volume

        if self._skimming:
            fmax = np.finfo(dtype="float64").max
            skim_values[skim_values == fmax] = 0.0
            if self._get_waiting_time:
                skim_matrix_dict = {name: skim_values[:, :, i] for i, name in enumerate(skim_cols)}
                skim_matrix_dict["waiting_time"][:] = (
                    skim_matrix_dict[self._trav_time_name]
                    - skim_matrix_dict["in_vehicle_trav_time"]
                    - skim_matrix_dict["egress_trav_time"]
                    - skim_matrix_dict["access_trav_time"]
                )
            self.skim_matrix = skims
        else:
            self.skim_matrix = None

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
            **table_name** (:obj:`str`): Name of the table to hold this assignment result.

            **keep_zero_flows** (:obj:`bool`): Whether we should keep records for zero flows.
            Defaults to ``True``.

            **project** (:obj:`Project`, optional): Project we want to save the results to.
            Defaults to the active project
        """

        df = self._edges
        if not keep_zero_flows:
            df = df[df.volume > 0]

        if not project:
            project = project or get_active_project()

        rep = {"setup": self.info()}
        project.results.create(
            table_name=table_name,
            data=df,
            procedure="Hyperpath assignment",
            procedure_id=self.procedure_id,
            procedure_report=json.dumps(rep),
            timestamp=self.procedure_date,
            description=self.description,
        )
