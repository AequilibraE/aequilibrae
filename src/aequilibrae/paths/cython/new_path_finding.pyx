from libc.stddef cimport size_t
from libc.math cimport cos

from aequilibrae.paths.cython.pq_heap_types cimport FourAryHeap, PairingHeap, StdPriorityQueueAdapter
from aequilibrae.paths.cython.path_finding cimport (
    dijkstra as cpp_dijkstra,
    a_star as cpp_a_star,
    HeuristicFn,
    haversine_heuristic,
    equirectangular_heuristic,
    SENTINEL,
)
from aequilibrae.utils.cython.bridge cimport Bridge

from aequilibrae.utils.logging_utils import basic_config

import logging

logger = logging.getLogger(__name__)


def run_dijkstra_example(type_of_heap: str = "FourAryHeap",
                         destination: int | None = None):

    basic_config(level=logging.DEBUG)
    cdef:
        size_t origin = 0
        size_t max_size = 3
        double costs[3]
        size_t csr[3]
        size_t fs[4]
        size_t predecessors[3]
        size_t ids[3]
        size_t connectors[3]
        size_t reached_first[3]
        unsigned char destinations[3]
        long long dest_count
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

    # Link IDs (just sequential for demo)
    ids[0] = 0
    ids[1] = 1
    ids[2] = 2

    # Set up destinations / early exit. dijkstra() re-initialises predecessors to SENTINEL itself.
    for i in range(max_size):
        destinations[i] = 0
        predecessors[i] = SENTINEL

    if destination is not None:
        destinations[<size_t>destination] = 1
        dest_count = 1
    else:
        dest_count = -1  # disables early exit

    with Bridge(logger) as b:
        if type_of_heap == "FourAryHeap":
            found = cpp_dijkstra[FourAryHeap](origin, max_size, &costs[0], &csr[0], &fs[0],
                                              &predecessors[0], &ids[0], &connectors[0],
                                              &reached_first[0], &destinations[0], dest_count, b.c)
        elif type_of_heap == "PairingHeap":
            found = cpp_dijkstra[PairingHeap](origin, max_size, &costs[0], &csr[0], &fs[0],
                                              &predecessors[0], &ids[0], &connectors[0],
                                              &reached_first[0], &destinations[0], dest_count, b.c)
        elif type_of_heap == "StdPriorityQueueAdapter":
            found = cpp_dijkstra[StdPriorityQueueAdapter](origin, max_size, &costs[0], &csr[0], &fs[0],
                                                          &predecessors[0], &ids[0], &connectors[0],
                                                          &reached_first[0], &destinations[0], dest_count, b.c)
        else:
            raise ValueError("Unknown heap type")

    # Reconstruct path to destination if early exit was used. Unreachable
    # nodes keep the SENTINEL predecessor written by dijkstra().
    cdef size_t dnode
    path = None
    if destination is not None:
        dnode = <size_t>destination
        if predecessors[dnode] != SENTINEL:
            path = [int(dnode)]
            dnode = predecessors[dnode]
            while True:
                path.append(int(dnode))
                if dnode == origin:
                    break
                dnode = predecessors[dnode]
            path.reverse()

    return {
        "heap": type_of_heap,
        "found": int(found),
        "predecessors": [int(predecessors[i]) for i in range(max_size)],
        "connectors": [int(connectors[i]) for i in range(max_size)],
        "path": [int(n) for n in path] if path else None,
    }


def run_a_star_example(type_of_heap: str = "FourAryHeap", heuristic: str = "haversine"):

    basic_config(level=logging.DEBUG)
    cdef:
        size_t origin = 0
        size_t destination = 3
        size_t max_size = 4
        # Graph: 0 -> 1 (0.5), 0 -> 2 (1.5), 1 -> 3 (0.8), 2 -> 3 (0.6)
        # Optimal path from 0 to 3: 0->1->3 = 1.3 (vs 0->2->3 = 2.1)
        double costs[4]
        size_t csr[4]
        size_t fs[5]
        size_t predecessors[4]
        size_t ids[4]
        size_t connectors[4]
        # Coordinates in degrees (roughly 0.00001 deg ~ 1.1m)
        double lats[4]
        double lons[4]
        size_t nodes_to_indices[4]
        HeuristicFn heur_fn
        void* heur_data
        double cos_lat1
        size_t i
        Bridge b

    # Edge costs
    costs[0] = 0.5   # 0 -> 1
    costs[1] = 1.5   # 0 -> 2
    costs[2] = 0.8   # 1 -> 3
    costs[3] = 0.6   # 2 -> 3

    # CSR: head vertices
    csr[0] = 1
    csr[1] = 2
    csr[2] = 3
    csr[3] = 3

    # CSR: row pointers
    fs[0] = 0
    fs[1] = 2
    fs[2] = 3
    fs[3] = 4
    fs[4] = 4

    # Link IDs
    ids[0] = 0
    ids[1] = 1
    ids[2] = 2
    ids[3] = 3

    # Lat/lon coordinates (in degrees):
    # Node 0: (0.0, 0.0)
    # Node 1: (0.001, 0.001)  ~111m NE of origin
    # Node 2: (0.005, 0.000)  ~556m E of origin
    # Node 3: (0.002, 0.002)  ~222m NE of origin, ~157m from both 1 and 2
    lats[0] = 0.0
    lons[0] = 0.0
    lats[1] = 0.001
    lons[1] = 0.001
    lats[2] = 0.005
    lons[2] = 0.000
    lats[3] = 0.002
    lons[3] = 0.002

    # nodes_to_indices: identity for this simple example
    for i in range(max_size):
        nodes_to_indices[i] = i
        predecessors[i] = SENTINEL

    # Select heuristic
    if heuristic == "haversine":
        heur_fn = haversine_heuristic
        cos_lat1 = cos(lats[destination] * 3.14159265358979323846 / 180.0)
        heur_data = <void*>&cos_lat1
    elif heuristic == "equirectangular":
        heur_fn = equirectangular_heuristic
        heur_data = <void*>0
    else:
        raise ValueError(f"Unknown heuristic: {heuristic}")

    with Bridge(logger) as b:
        if type_of_heap == "FourAryHeap":
            cpp_a_star[FourAryHeap](origin, destination, max_size,
                                    &costs[0], &csr[0], &fs[0],
                                    &nodes_to_indices[0], &lats[0], &lons[0],
                                    &predecessors[0], &ids[0], &connectors[0],
                                    heur_fn, heur_data, b.c)
        elif type_of_heap == "PairingHeap":
            cpp_a_star[PairingHeap](origin, destination, max_size,
                                    &costs[0], &csr[0], &fs[0],
                                    &nodes_to_indices[0], &lats[0], &lons[0],
                                    &predecessors[0], &ids[0], &connectors[0],
                                    heur_fn, heur_data, b.c)
        elif type_of_heap == "StdPriorityQueueAdapter":
            cpp_a_star[StdPriorityQueueAdapter](origin, destination, max_size,
                                                &costs[0], &csr[0], &fs[0],
                                                &nodes_to_indices[0], &lats[0], &lons[0],
                                                &predecessors[0], &ids[0], &connectors[0],
                                                heur_fn, heur_data, b.c)
        else:
            raise ValueError("Unknown heap type")

    # Reconstruct path, guarding against an unreachable destination (SENTINEL predecessor)
    path = None
    cdef size_t node = destination
    if node == origin or predecessors[node] != SENTINEL:
        path = []
        while True:
            path.append(int(node))
            if node == origin:
                break
            node = predecessors[node]
        path.reverse()

    return {
        "heap": type_of_heap,
        "heuristic": heuristic,
        "path": path,
        "predecessors": [int(predecessors[i]) for i in range(max_size)],
        "connectors": [int(connectors[i]) for i in range(max_size)],
    }


def example():
    print('=== Dijkstra examples ===')
    for heap in ['FourAryHeap', 'PairingHeap', 'StdPriorityQueueAdapter']:
        for dest in [None, 2]:
            r = run_dijkstra_example(heap, destination=dest)
            label = f'dest={dest}' if dest else 'no dest'
            print(f'  {heap} ({label}): found={r["found"]}, path={r["path"]}, preds={r["predecessors"]}')
            assert r['found'] == 2  # 3 nodes total, found-1 = 2
            assert r['predecessors'][0] == 2**64 - 1  # SENTINEL
            if dest == 2:
                assert r['path'] == [0, 1, 2]
            else:
                assert r['predecessors'][2] == 1

    print()
    print('=== A* examples ===')
    for heap in ['FourAryHeap', 'PairingHeap', 'StdPriorityQueueAdapter']:
        for heur in ['haversine', 'equirectangular']:
            r = run_a_star_example(heap, heur)
            print(f'  {heap} + {heur}: path={r["path"]}')
            assert r['path'][0] == 0 and r['path'][-1] == 3
            assert r['predecessors'][3] == 1

    print()
    print('All assertions passed!')
