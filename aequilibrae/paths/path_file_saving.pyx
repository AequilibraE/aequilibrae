# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.string cimport string
from string_helper cimport to_string

import pyarrow as pa
cimport pyarrow as pa

import numpy as np
cimport numpy as np

import pyarrow.parquet as pq

# from pyarrow.lib cimport *


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False) # turn of bounds-checking for entire function
cpdef void save_path_file(long origin_index,
                          long num_links,
                          long zones,
                          long long [:] pred,
                          long long [:] conn): #nogil:

    cdef long long class_, node, predecessor, connector, ctr
    cdef string file_name
    cdef vector[long long] path_for_od_pair_and_class
    cdef long long* temp_data

    cdef np.npy_intp dims

    for node in range(zones):
        path_for_od_pair_and_class.clear()

        # tracing backwards from each destination for this one-to-all shortest path
        predecessor = pred[node]
        connector = conn[node]
        while predecessor >= 0:
            path_for_od_pair_and_class.push_back(connector)
            predecessor = pred[predecessor]
            connector = conn[predecessor]

        file_name = b'test_' + to_string(origin_index) + b'.parquet'

        # get a view on data underlying vector, then as numpy array. avoids copying.
        dims = <np.npy_intp> (path_for_od_pair_and_class.size() + 1)
        temp_data = path_for_od_pair_and_class.data()
        pq.write_table(pa.array(np.PyArray_SimpleNewFromData(1, &dims, np.NPY_LONGLONG, temp_data)), file_name)


