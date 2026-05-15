from libc.stddef cimport size_t

from aequilibrae.paths.cython.pq_heap_types cimport FourAryHeap, PairingHeap, StdPriorityQueueAdapter
from aequilibrae.paths.cython.path_finding cimport dijkstra
from aequilibrae.utils.cython.bridge cimport Bridge, log, aeq_format_string as f, DEBUG, msleep

from aequilibrae.utils.logging_utils import basic_config

import logging

logger = logging.getLogger(__name__)


def run_dijkstra_example(type_of_heap: str = "FourAryHeap"):

    basic_config(level=logging.DEBUG)
    cdef:
        size_t origin = 0
        size_t max_size = 3
        double costs[3]
        size_t csr[3]
        size_t fs[4]
        size_t predecessors[3]
        size_t found
        size_t i
        Bridge b

    # Tiny directed graph in CSR form:
    # 0 -> 1 (1.0), 0 -> 2 (4.0), 1 -> 2 (1.0)
    costs[0] = 1.0
    costs[1] = 4.0
    costs[2] = 1.0

    csr[0] = 1
    csr[1] = 2
    csr[2] = 2

    fs[0] = 0
    fs[1] = 2
    fs[2] = 3
    fs[3] = 3

    for i in range(max_size):
        predecessors[i] = max_size

    with Bridge(logger) as b:
        if type_of_heap == "FourAryHeap":
            found = dijkstra[FourAryHeap](origin, max_size, &costs[0], &csr[0], &fs[0], &predecessors[0], b.c)
        # elif type_of_heap == "PairingHeap":
        #     found = dijkstra[PairingHeap](origin, max_size, &costs[0], &csr[0], &fs[0], &predecessors[0])
        # elif type_of_heap == "StdPriorityQueueAdapter":
        #     found = dijkstra[StdPriorityQueueAdapter](origin, max_size, &costs[0], &csr[0], &fs[0], &predecessors[0])
        else:
            raise ValueError("Unknown heap type")

    return {
        "heap": type_of_heap,
        "found": int(found),
        "predecessors": [int(predecessors[0]), int(predecessors[1]), int(predecessors[2])],
    }
