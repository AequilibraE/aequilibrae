# cython: language_level=3

from aequilibrae.utils.cython.openmp_helper import omp_get_max_threads
import socket
import logging
import json

import numpy as np
import pandas as pd

from aequilibrae.context import get_active_project
from aequilibrae.matrix import AequilibraeMatrix

from aequilibrae.utils.cython.bridge cimport Bridge

include 'hyperpath.pyx'

logger = logging.getLogger(__name__)

# Skims that count the edges of these link types
COUNTED_SKIMS = {
    "boardings": ["boarding"],
    "alightings": ["alighting"],
    "inner_transfers": ["inner_transfer"],
    "outer_transfers": ["outer_transfer"],
}

# Skims that add up the time spent on edges of these link types
TIMED_SKIMS = {
    "on_board_trav_time": ["on-board"],
    "dwelling_time": ["dwell"],
    "egress_trav_time": ["egress_connector"],
    "access_trav_time": ["access_connector"],
    "walking_trav_time": ["walking"],
    "transfer_time": ["inner_transfer", "outer_transfer"],
    "in_vehicle_trav_time": ["on-board", "dwell"],
}

# Unweighted time of each link type on graphs from TransitGraphBuilder. Graphs without these columns, such as raw
# kernel inputs, use the travel time field instead.
PHYSICAL_TIME_FIELDS = {
    "on-board": "run_time",
    "dwell": "dwell_time",
    "access_connector": "walk_time",
    "egress_connector": "walk_time",
    "walking": "walk_time",
    "inner_transfer": "walk_time",
    "outer_transfer": "walk_time",
}

# Skims of the strategy itself, which the kernel computes from frequencies and headway fractions. Waits are the
# headway fraction over the combined frequency of the lines a stop's strategy boards, so they are not weighted by the
# class's wait weight. Transfers are expected boardings minus the probability of boarding at all.
STRATEGY_SKIMS = ["waiting_time", "first_waiting_time", "transfer_waiting_time", "transfers", "pt_probability"]

_WAITING_EDGES = "__waiting_edges__"


class HyperpathGenerating:
    """A class for hyperpath generation.

    Edges with infinite frequency have no wait, and edges with zero frequency are never used. A vertex whose strategy
    boards lines of combined frequency :math:`F_i` waits :math:`\\alpha_i / F_i` on average, perceived as
    :math:`\\kappa_i / F_i` with :math:`\\kappa_i = \\alpha_i \\omega_i`.

    :Arguments:
        **edges** (:obj:`pandas.DataFrame`): The edges of the graph.

        **tail** (:obj:`str`, optional): The column name for the tail of the edge. Default is "tail".

        **head** (:obj:`str`, optional): The column name for the head of the edge. Default is "head".

        **trav_time** (:obj:`str`, optional): The column name for the travel time of the edge.
        Default is "trav_time".

        **freq** (:obj:`str`, optional): The column name for the frequency of the edge. Default is "freq".

        **check_edges** (:obj:`bool`, optional): If ``True``, check the validity of the edges.
        Default is ``False``.

        **skim_cols** (:obj:`list[str]`, optional): Edge columns or predefined skims to compute.

        **headway_fraction** (:obj:`float` or :obj:`np.ndarray`, optional): Expected wait as a fraction of the
        headway, :math:`\\alpha`, for every vertex or per vertex index. Default is ``1.0``.

        **wait_weight** (:obj:`float` or :obj:`np.ndarray`, optional): Perception weight of waiting time,
        :math:`\\omega`. Default is ``1.0``.

        **cost_floor** (:obj:`float`, optional): Smallest cost of any edge, which keeps labels strictly increasing
        along attractive edges.
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
            headway_fraction=1.0,
            wait_weight=1.0,
            cost_floor=COST_FLOOR_PY,
    ):
        skim_cols = self.check_skim_cols(skim_cols)

        if check_edges:
            self._check_edges(edges, tail, head, trav_time, freq, [c for c in skim_cols if c in edges.columns])

        edges = edges.reset_index(drop=True)
        self._trav_time_name = trav_time
        self.edge_count = len(edges)
        self._tail = edges[tail].to_numpy().astype(np.uint32)
        self._head = edges[head].to_numpy().astype(np.uint32)
        self.vertex_count = int(max(self._tail.max(), self._head.max())) + 1 if self.edge_count else 0
        self._indptr, self._edge_idx = csc_structure(self._tail, self._head, self.vertex_count)

        self._edges = edges[[tail, head]].copy()
        self._prepare_skims(edges)

        self._o_vert_ids = np.asarray(o_vert_ids).astype(np.int64)  # node_id for origin in the above taz_id
        self._d_vert_ids = np.asarray(d_vert_ids).astype(np.int64)  # node_id for destination in the above taz_id
        self._nodes_to_indices = np.asarray(nodes_to_indices, dtype=np.int64)

        self.cost_floor = float(cost_floor)
        self._headway_fraction = None
        self._wait_weight = None
        self.update_costs(
            edges[trav_time].to_numpy(), edges[freq].to_numpy(), headway_fraction=headway_fraction,
            wait_weight=wait_weight,
        )
        self._workspaces = _Workspaces()

        self.u_i_vec = None
        self.skim_matrix = None
        self.unassigned_demand = 0.0
        self.weighted_waiting_time = 0.0

    def check_skim_cols(self, skim_cols):
        self._skimming = True
        if isinstance(skim_cols, (tuple, set)):
            skim_cols = list(skim_cols)

        if not skim_cols or not isinstance(skim_cols, list):
            skim_cols = []
            self._skimming = False

        self._requested_skims = list(dict.fromkeys(skim_cols))
        return skim_cols

    def _prepare_skims(self, edges: pd.DataFrame):
        """Turn the requested skims into kernel fields: edge values, how each combines along the strategy, and
        whether it adds the waits."""
        trav_time = self._trav_time_name
        self._is_travel_time = trav_time in self._requested_skims
        self._skim_cols_names = [name for name in self._requested_skims if name != trav_time]

        predefined = COUNTED_SKIMS.keys() | TIMED_SKIMS.keys()
        if any(n in predefined and n not in edges.columns for n in self._skim_cols_names) and (
            "link_type" not in edges.columns
        ):
            raise ValueError("predefined skimming type requested but 'link_type' column not present on the graph")

        fields = {}  # name -> (operator, waits, edge values)
        for name in self._skim_cols_names:
            if name in edges.columns:
                fields[name] = (SKIM_ADD_PY, 0, edges[name].to_numpy(dtype=np.float64))
            elif name in COUNTED_SKIMS:
                values = edges["link_type"].isin(COUNTED_SKIMS[name]).to_numpy(dtype=np.float64)
                fields[name] = (SKIM_ADD_PY, 0, values)
            elif name in TIMED_SKIMS:
                values = np.zeros(self.edge_count)
                for link_type in TIMED_SKIMS[name]:
                    rows = (edges["link_type"] == link_type).to_numpy()
                    column = PHYSICAL_TIME_FIELDS[link_type]
                    column = column if column in edges.columns else trav_time
                    values[rows] = edges[column].to_numpy(dtype=np.float64)[rows]
                fields[name] = (SKIM_ADD_PY, 0, values)
            elif name in ("waiting_time", "transfer_waiting_time", "first_waiting_time"):
                if name != "first_waiting_time":
                    fields.setdefault("waiting_time", (SKIM_ADD_PY, 1, None))
                if name != "waiting_time":
                    fields.setdefault("first_waiting_time", (SKIM_RESET_AT_WAIT_PY, 1, None))
            elif name in ("transfers", "pt_probability"):
                fields.setdefault("pt_probability", (SKIM_ANY_WAIT_PY, 0, None))
                if name == "transfers":
                    fields.setdefault(_WAITING_EDGES, (SKIM_COUNT_WAITS_PY, 0, None))
            else:
                raise KeyError(f"edge column '{name}' not found in graph edges dataframe")

        self._fields = list(fields)
        self._skim_operators = np.array([op for op, _, _ in fields.values()], dtype=np.uint8)
        self._skim_waits = np.array([waits for _, waits, _ in fields.values()], dtype=np.uint8)
        self._skim_cols = np.zeros((self.edge_count, max(len(fields), 1)), dtype=DATATYPE_PY)
        for k, (_, _, values) in enumerate(fields.values()):
            if values is not None:
                self._skim_cols[:, k] = values
                if not np.all(np.isfinite(values)) or np.any(values < 0):
                    raise ValueError(f"skim column '{self._fields[k]}' should be finite and nonnegative")
                if self._fields[k] not in edges.columns:
                    self._edges[self._fields[k]] = values
        self._skim_cols = np.ascontiguousarray(self._skim_cols[:, : len(fields)])

    def update_costs(self, trav_time=None, freq=None, *, headway_fraction=None, wait_weight=None):
        """Replace edge costs, frequencies or wait parameters, keeping the graph structure.

        Arrays that are already contiguous ``float64`` are used without copying, so changing them in place also
        changes the next assignment.

        :Arguments:
            **trav_time** (:obj:`np.ndarray`, optional): Nonnegative cost of every edge.

            **freq** (:obj:`np.ndarray`, optional): Frequency of every edge, ``inf`` for edges without a wait and
            ``0`` for edges that cannot be used.

            **headway_fraction** (:obj:`float` or :obj:`np.ndarray`, optional): Expected wait as a fraction of the
            headway.

            **wait_weight** (:obj:`float` or :obj:`np.ndarray`, optional): Perception weight of waiting time.
        """
        if trav_time is not None:
            cost = np.ascontiguousarray(trav_time, dtype=DATATYPE_PY)
            if cost.shape != (self.edge_count,):
                raise ValueError("there must be one cost per edge")
            if np.any(np.isnan(cost)) or np.any(cost < 0.0):
                raise ValueError("edge costs must be nonnegative")
            self._trav_time = cost
        if freq is not None:
            frequency = np.ascontiguousarray(freq, dtype=DATATYPE_PY)
            if frequency.shape != (self.edge_count,):
                raise ValueError("there must be one frequency per edge")
            if np.any(np.isnan(frequency)) or np.any(frequency < 0.0):
                raise ValueError("edge frequencies must be nonnegative")
            self._freq = frequency
        if headway_fraction is not None:
            self._headway_fraction = self._vertex_values(headway_fraction, "headway_fraction")
        if wait_weight is not None:
            self._wait_weight = self._vertex_values(wait_weight, "wait_weight")
        if headway_fraction is not None or wait_weight is not None:
            self._wait_constant = self._headway_fraction * self._wait_weight

    def _vertex_values(self, values, name):
        values = np.broadcast_to(np.asarray(values, dtype=DATATYPE_PY), (self.vertex_count,))
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(f"{name} must be finite and nonnegative")
        return np.ascontiguousarray(values)

    def run(self, origin, destination, volume):
        """Load ``volume`` trips from vertex ``origin`` to vertex ``destination`` and keep the destination's labels
        in ``u_i_vec``."""
        self.assign(
            np.array([origin]),
            np.array([destination]),
            np.array([volume], dtype=DATATYPE_PY),
            threads=1
        )

    def strategy(self, origin, destination) -> pd.DataFrame:
        """Return the optimal strategy of one trip between two vertices.

        Every edge the trip may use is listed with its expected ``volume`` and its ``split``, the share of the flow
        through its tail that takes it. Neither the edge volumes nor the skims of the last assignment change.

        :Arguments:
            **origin** (:obj:`int`): Origin vertex ID.

            **destination** (:obj:`int`): Destination vertex ID.

        :Returns:
            **strategy** (:obj:`pd.DataFrame`): One row per edge, indexed by the edge's position in the graph.
        """
        volume, _ = self._solve(
            np.array([origin], dtype=np.int64), np.array([destination], dtype=np.int64), np.ones(1), threads=1,
            skim=False, labels=False,
        )
        used = np.flatnonzero(volume > 1.0e-12)
        outflow = np.bincount(self._tail, weights=volume, minlength=self.vertex_count)
        frame = pd.DataFrame(
            {
                "tail": self._tail[used],
                "head": self._head[used],
                "volume": volume[used],
                "split": volume[used] / outflow[self._tail[used]],
            },
            index=pd.Index(used, name="edge"),
        )
        return frame

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

        Assumes the `*_column` arguments are provided as numpy arrays that form a COO sprase matrix. Demand from an
        origin that cannot reach its destination is not loaded, and is reported in ``unassigned_demand``.

        :Arguments:
            **origin_column** (:obj:`np.ndarray`, optional): The column for the origin vertices.
            Default is "orig_vert_idx".

            **destination_column** (:obj:`np.ndarray`, optional): The column or the destination vertices.
            Default is "dest_vert_idx".

            **demand_column** (:obj:`np.ndarray`, optional): The column for the demand values.
            Default is "demand".

            **check_demand** (:obj:`bool`, optional): If ``True``, check the validity of the demand data.
            Default is ``False``.

            **threads** (:obj:`int`, optional): The number of threads to use for computation.
            Default is ``0`` (using all available threads).

        """

        origin_column = np.asarray(origin_column).astype(np.uint32)
        destination_column = np.asarray(destination_column).astype(np.uint32)
        demand_column = np.asarray(demand_column).astype(DATATYPE_PY)
        # check the input demand parameter
        if check_demand:
            self._check_demand(origin_column, destination_column, demand_column)

        volume, labels = self._solve(
            origin_column.astype(np.int64), destination_column.astype(np.int64), demand_column, threads,
            skim=self._skimming, labels=True,
        )
        self._edges["volume"] = volume
        self.u_i_vec = labels

    def _solve(self, origins, destinations, demand, threads, skim, labels):
        """Run the kernel for vertex IDs and return edge volumes and, for a single destination, its labels."""
        if threads is None or threads < 1:
            threads = omp_get_max_threads()

        def indices(vertex_ids):
            inside = (vertex_ids >= 0) & (vertex_ids < len(self._nodes_to_indices))
            mapped = np.full(len(vertex_ids), -1, dtype=np.int64)
            mapped[inside] = self._nodes_to_indices[vertex_ids[inside]]
            return mapped

        origin_idx, destination_idx = indices(origins), indices(destinations)

        # Demand to or from vertices outside the graph cannot be loaded
        present = (origin_idx >= 0) & (destination_idx >= 0)
        absent = pd.DataFrame({"origin": origins[~present], "destination": destinations[~present],
                               "demand": demand[~present]})

        order = np.argsort(destination_idx[present], kind="stable")
        demand_origins = origin_idx[present][order].astype(np.uint32)
        demand_destinations = destination_idx[present][order].astype(np.uint32)
        demand_values = np.ascontiguousarray(demand[present][order], dtype=DATATYPE_PY)
        tasks, starts = np.unique(demand_destinations, return_index=True)
        demand_indptr = np.append(starts, len(demand_destinations)).astype(np.uint32)
        labels_task = 0 if labels and len(tasks) == 1 else -1

        zones = len(self._o_vert_ids)
        if skim:
            # Skim the destinations without demand too
            zone_destinations = indices(self._d_vert_ids)
            extra = zone_destinations[(zone_destinations >= 0) & ~np.isin(zone_destinations, tasks)]
            tasks = np.concatenate((tasks, extra)).astype(np.uint32)
            demand_indptr = np.append(demand_indptr, np.full(len(extra), len(demand_destinations))).astype(np.uint32)
            zone_of_vertex = np.full(self.vertex_count, -1, dtype=np.int64)
            valid = zone_destinations >= 0
            zone_of_vertex[zone_destinations[valid]] = np.arange(zones)[valid]
            destination_zones = zone_of_vertex[tasks]
            skim_matrix = np.full((zones, zones, len(self._fields) + int(self._is_travel_time)), np.inf)
        else:
            tasks = tasks.astype(np.uint32)
            destination_zones = np.full(len(tasks), -1, dtype=np.int64)
            skim_matrix = np.full((1, 1, 1), np.inf)

        blocks = max(1, min(threads, len(tasks)))
        block_volumes = np.zeros((blocks, self.edge_count), dtype=DATATYPE_PY)
        block_waits = np.zeros(blocks, dtype=DATATYPE_PY)
        unassigned = np.zeros(len(demand_values), dtype=DATATYPE_PY)
        vertex_labels = np.full(self.vertex_count, np.inf, dtype=DATATYPE_PY)

        with Bridge(logger) as bridge:
            assign_destinations(
                self._indptr,
                self._edge_idx,
                self._tail,
                self._head,
                self._trav_time,
                self._freq,
                self._wait_constant,
                self._headway_fraction,
                self.cost_floor,
                self._skim_cols,
                self._skim_operators,
                self._skim_waits,
                tasks,
                destination_zones,
                demand_indptr,
                demand_origins,
                demand_values,
                unassigned,
                block_volumes,
                block_waits,
                indices(self._o_vert_ids),
                skim_matrix,
                skim and self._is_travel_time,
                vertex_labels,
                labels_task,
                self._workspaces,
                threads,
                bridge,
            )

        # Fixed-order reduction, so a thread count always gives the same result
        volume = block_volumes[0].copy()
        for block in range(1, blocks):
            volume += block_volumes[block]

        unreached = pd.DataFrame(
            {
                "origin": self._indices_to_nodes(demand_origins),
                "destination": self._indices_to_nodes(demand_destinations),
                "demand": unassigned,
            }
        )
        unassigned_od = pd.concat([absent, unreached], ignore_index=True)
        self.unassigned_od = unassigned_od[unassigned_od.demand > 0].reset_index(drop=True)
        self.unassigned_demand = float(self.unassigned_od.demand.sum())
        self.weighted_waiting_time = float(sum(block_waits))

        if skim:
            self._store_skims(skim_matrix)
        return volume, (vertex_labels if labels_task == 0 else None)

    def _indices_to_nodes(self, indices):
        nodes = np.flatnonzero(self._nodes_to_indices >= 0)
        lookup = np.empty(self.vertex_count, dtype=np.int64)
        lookup[self._nodes_to_indices[nodes]] = nodes
        return lookup[indices]

    def _store_skims(self, skim_matrix):
        fields = ([self._trav_time_name] if self._is_travel_time else []) + self._fields
        values = {name: skim_matrix[:, :, i] for i, name in enumerate(fields)}

        def difference(total, part):
            # Pairs without a strategy stay infinite rather than becoming nan
            with np.errstate(invalid="ignore"):
                return np.where(np.isinf(total), np.inf, total - part)

        if "transfer_waiting_time" in self._skim_cols_names:
            values["transfer_waiting_time"] = difference(values["waiting_time"], values["first_waiting_time"])
        if "transfers" in self._skim_cols_names:
            values["transfers"] = difference(values[_WAITING_EDGES], values["pt_probability"])

        names = ([self._trav_time_name] if self._is_travel_time else []) + self._skim_cols_names
        self.skim_matrix = AequilibraeMatrix(index=self._o_vert_ids)
        self.skim_matrix.update({name: np.ascontiguousarray(values[name]) for name in names})

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
