import numpy as np
import pandas as pd

from aequilibrae._assignment import AssignmentResultsBase
from aequilibrae.routing.cython.dijkstra import available_heaps


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
        values = self.state.mapping.project_loads(self.output.loading.link_loads)
        values.flags.writeable = False
        return values

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
        return self.state.skims

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
        self.state.reset()

    def set_heap(self, heap):
        if heap not in available_heaps():
            raise ValueError(f"heap must be one of {available_heaps()}")
        self._heap = heap

    @staticmethod
    def get_heaps():
        return available_heaps()

    def _load_frame(self, graph_loads, prefix=""):
        mapping = self.state.mapping
        loads = mapping.project_loads(graph_loads)
        indices = mapping.network.directed_link_indices
        valid = indices >= 0

        directional = np.zeros((len(indices), 2, loads.shape[1]), dtype=loads.dtype)
        directional[valid] = loads[indices[valid]]
        ab, ba = directional[:, 0], directional[:, 1]

        columns = {}
        for index, name in enumerate(self.classes["names"]):
            columns[f"{prefix}{name}_ab"] = ab[:, index]
            columns[f"{prefix}{name}_ba"] = ba[:, index]
            columns[f"{prefix}{name}_tot"] = ab[:, index] + ba[:, index]

        return pd.DataFrame(columns, index=mapping.network.physical_link_ids)

    def get_load_results(self):
        """Return directional link loads in external network link order."""
        return self._load_frame(self.output.loading.link_loads)

    def get_sl_results(self):
        """Return directional link loads for each selection and demand column."""
        frames = [self._load_frame(loads, f"{name}_") for name, loads in self.select_link_loading.items()]
        if frames:
            return pd.concat(frames, axis=1)
        return pd.DataFrame(index=self.state.mapping.network.physical_link_ids)
