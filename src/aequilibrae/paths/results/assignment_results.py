from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from aequilibrae.parameters import Parameters
from aequilibrae.paths.graph import _get_graph_to_network_mapping
from aequilibrae.utils.core_setter import clamp_cores, resolve_cores, resolve_elementwise_cores
from aequilibrae.utils.core_setter import resolve_threading_threshold


class AssignmentResultsBase(ABC):
    """Shared reporting and thread settings for traffic and transit results."""

    def __init__(self):
        self._system_parameters = Parameters().parameters["system"]
        self.set_cores(resolve_cores(self._system_parameters), resolve_threading_threshold(self._system_parameters))

    def set_cores(self, cores, threading_threshold=None, elementwise_cores=None):
        """Set thread counts without reallocating or clearing existing results."""
        self.cores = clamp_cores(cores)
        if threading_threshold is not None:
            if not isinstance(threading_threshold, int):
                raise ValueError("Threading threshold needs to be an integer")
            self.threading_threshold = threading_threshold
        self.elementwise_cores = (
            clamp_cores(elementwise_cores)
            if elementwise_cores is not None
            else resolve_elementwise_cores(self._system_parameters, self.cores)
        )

    @abstractmethod
    def reset(self):
        pass


class AssignmentResults(AssignmentResultsBase):
    """Public results for one traffic class, backed by compact output owners.

    Skims and selected OD demand are the routing output objects. Full-network
    link loads are reporting snapshots, not a second mutable assignment state.
    Every load and turn total is in the original demand units, without PCE.
    """

    def __init__(self):
        super().__init__()
        self._state = None
        self.classes = {"number": 0, "names": ()}
        self._heap = "4ary"
        self.save_path_file = False
        self.write_feather = True

    def bind(self, state, class_names):
        """Attach the accepted or latest AoN state at the start of execution."""
        self._state = state
        self.classes = {"number": len(class_names), "names": tuple(class_names)}

    @property
    def state(self):
        if self._state is None:
            raise RuntimeError("Assignment results are not prepared")
        return self._state

    @property
    def output(self):
        return self.state.output

    @property
    def link_loads(self):
        """Read-only snapshot of loads in supernetwork order."""
        return self.state.mapping.full_loads(self.output.loading.link_loads)

    @property
    def compact_link_loads(self):
        return self.output.loading.link_loads

    @property
    def total_link_loads(self):
        values = self.state.total_link_loads.view()
        values.flags.writeable = False
        return values

    @property
    def skims(self):
        return self.output.skimming

    @property
    def select_link_od(self):
        selected = self.output.select_link
        return None if selected is None else selected.od

    @property
    def select_link_loading(self):
        selected = self.output.select_link
        return {} if selected is None or selected.loading is None else selected.loading.loads

    @property
    def total_turn_penalty(self):
        return self.output.turn_cost_total

    @property
    def unassigned_demand(self):
        return self.output.unassigned_demand

    def reset(self):
        self.output.reset()
        self.state.update_totals()

    def set_heap(self, heap):
        # FIXME: Add assignment heap selection to PreparedAoN.
        if heap != "4ary":
            raise NotImplementedError("Assignment currently supports only the 4ary heap")
        self._heap = heap

    @staticmethod
    def get_heaps():
        return ["4ary"]

    def get_graph_to_network_mapping(self):
        mapping = self.state.mapping
        return _get_graph_to_network_mapping(mapping.link_ids, mapping.directions)

    def _load_frame(self, compact, prefix=""):
        mapping = self.state.mapping
        network = self.get_graph_to_network_mapping()
        loads = mapping.full_loads(compact)[mapping.graph_ids]
        link_ids = np.unique(mapping.link_ids)
        columns = {}
        for index, name in enumerate(self.classes["names"]):
            ab = np.zeros(len(link_ids), dtype=np.float64)
            ba = np.zeros(len(link_ids), dtype=np.float64)
            ab[network.network_ab_idx] = loads[network.graph_ab_idx, index]
            ba[network.network_ba_idx] = loads[network.graph_ba_idx, index]
            columns[f"{prefix}{name}_ab"] = ab
            columns[f"{prefix}{name}_ba"] = ba
            columns[f"{prefix}{name}_tot"] = ab + ba
        return pd.DataFrame(columns, index=link_ids)

    def get_load_results(self):
        """Return directional link loads in external network link order."""
        return self._load_frame(self.output.loading.link_loads)

    def get_sl_results(self):
        """Return directional link loads for each selection and demand column."""
        frames = [self._load_frame(loads, f"{name}_") for name, loads in self.select_link_loading.items()]
        if frames:
            return pd.concat(frames, axis=1)
        return pd.DataFrame(index=np.unique(self.state.mapping.link_ids))


class TransitAssignmentResults(AssignmentResultsBase):
    """Transit loads are owned by the hyperpath computation."""

    def __init__(self):
        super().__init__()
        self.link_loads = np.array([])

    def prepare(self, graph, matrix):
        self.reset()
        self.nodes = graph.num_nodes
        self.zones = graph.num_zones
        self.centroids = graph.centroids
        self.links = graph.num_links
        self.lids = graph.graph.link_id.to_numpy(copy=False)

    def reset(self):
        self.link_loads.fill(0)

    def get_load_results(self):
        if not self.link_loads.shape[0]:
            raise ValueError("Transit assignment has not been executed yet")
        return pd.DataFrame({"volume": self.link_loads}, index=self.lids)
