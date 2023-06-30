# cython: language_level=3

import pandas as pd

include 'hyperpath.pyx'


class HyperpathGenerating:
    def __init__(self, edges, tail="tail", head="head", trav_time="trav_time", freq="freq", check_edges=False):
        # load the edges
        if check_edges:
            self._check_edges(edges, tail, head, trav_time, freq)
        self._edges = edges[[tail, head, trav_time, freq]].copy(deep=True)
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

    def run(self, origin, destination, volume, return_inf=False):
        # column storing the resulting edge volumes
        self._edges["volume"] = 0.0
        self.u_i_vec = None

        # allocation of work arrays
        u_i_vec = np.empty(self.vertex_count, dtype=DATATYPE_PY)
        f_i_vec = np.empty(self.vertex_count, dtype=DATATYPE_PY)
        u_j_c_a_vec = np.empty(self.edge_count, dtype=DATATYPE_PY) 
        v_i_vec = np.empty(self.vertex_count, dtype=DATATYPE_PY)
        h_a_vec = np.empty(self.edge_count, dtype=bool)
        edge_indices = np.empty(self.edge_count, dtype=np.uint32)

        # input check
        if type(volume) is not list:
            volume = [volume]
        if type(origin) is not list:
            origin = [origin]
        assert len(origin) == len(volume)
        for i, item in enumerate(origin):
            self._check_vertex_idx(item)
            self._check_volume(volume[i])
        self._check_vertex_idx(destination)
        demand_indices = np.array(origin, dtype=np.uint32)
        assert isinstance(return_inf, bool)

        demand_values = np.array(volume, dtype=DATATYPE_PY)

        compute_SF_in(
            self._indptr,
            self._edge_idx,
            self._trav_time,
            self._freq,
            self._tail,
            self._head,
            demand_indices,
            demand_values,
            self._edges["volume"].values,
            u_i_vec,
            f_i_vec,
            u_j_c_a_vec,
            v_i_vec,
            h_a_vec,
            edge_indices,
            self.vertex_count,
            destination,
        )
        self.u_i_vec = u_i_vec

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
    ):
        # check the input demand paramater
        if check_demand:
            self._check_demand(demand, origin_column, destination_column, demand_column)
        demand = demand[demand[demand_column] > 0]

        # initialize the column storing the resulting edge volumes
        self._edges["volume"] = 0.0

        # travel time is computed but not saved into an array in the following
        self.u_i_vec = None

        o_vert_ids = demand[origin_column].values
        d_vert_ids = demand[destination_column].values
        demand_vls = demand[demand_column].values

        # get the list of all destinations
        destination_vertex_indices = np.unique(d_vert_ids)

        # allocation of work arrays
        u_i_vec = np.empty(self.vertex_count, dtype=DATATYPE_PY)
        f_i_vec = np.empty(self.vertex_count, dtype=DATATYPE_PY)
        u_j_c_a_vec = np.empty(self.edge_count, dtype=DATATYPE_PY) 
        v_i_vec = np.empty(self.vertex_count, dtype=DATATYPE_PY)
        h_a_vec = np.empty(self.edge_count, dtype=bool)
        edge_indices = np.empty(self.edge_count, dtype=np.uint32)

        # loop on destination vertices
        for destination_vertex_index in destination_vertex_indices:
            demand_indices = np.where(d_vert_ids == destination_vertex_index)[0]

            # list of origin vertices
            demand_origins = o_vert_ids[demand_indices]
            demand_origins = demand_origins.astype(np.uint32)

            # list of demand values
            demand_values = demand_vls[demand_indices]
            demand_values = demand_values.astype(DATATYPE_PY)

            # initialization of the edge volume vector
            edge_volume = np.zeros_like(self._edges["volume"])

            # S&F
            compute_SF_in(
                self._indptr,
                self._edge_idx,
                self._trav_time,
                self._freq,
                self._tail,
                self._head,
                demand_origins,
                demand_values,
                edge_volume,
                u_i_vec,
                f_i_vec,
                u_j_c_a_vec,
                v_i_vec,
                h_a_vec,
                edge_indices,
                self.vertex_count,
                destination_vertex_index,
            )
            self._edges["volume"] += edge_volume

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