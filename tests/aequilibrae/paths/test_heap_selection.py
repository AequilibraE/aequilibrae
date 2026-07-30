import logging

import numpy as np
import pytest

from aequilibrae.paths import PathResults, available_heaps

HEAP_CLASS_NAMES = {"4ary": "FourAryHeap", "pairing": "PairingHeap", "std": "StdPriorityQueueAdapter"}


def build_graph(project):
    project.network.build_graphs()
    g = project.network.graphs["c"]
    g.set_blocked_centroid_flows(False)
    g.set_graph(cost_field="distance")
    g.set_skimming("distance")
    return g


@pytest.mark.parametrize("heap", available_heaps())
def test_path_computation_identical_across_heaps(sioux_falls_example, heap):
    g = build_graph(sioux_falls_example)

    reference = PathResults(g, 1, 20)

    res = PathResults(g, 1, 20, heap=heap)

    assert res.path_nodes is not None and res.milepost is not None and res.skims is not None
    assert reference.milepost is not None and reference.skims is not None

    assert res._heap == heap
    assert res.path_nodes[0] == 1 and res.path_nodes[-1] == 20
    # Path cost and skims are invariant across heaps (tie-breaking may differ, costs may not)
    assert res.milepost[-1] == reference.milepost[-1]
    np.testing.assert_allclose(res.skims, reference.skims)


@pytest.mark.parametrize("heap", available_heaps())
def test_heap_name_surfaces_through_bridge(sioux_falls_example, heap):
    g = build_graph(sioux_falls_example)

    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    # The Bridge dispatches to the aequilibrae logger hierarchy; it is only spun up at DEBUG
    logger = logging.getLogger("aequilibrae")
    handler = Capture(level=logging.DEBUG)
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        PathResults(g, 1, 20, heap=heap)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)

    expected = f"{HEAP_CLASS_NAMES[heap]}: init_heap"
    assert any(expected in message for message in records), records
