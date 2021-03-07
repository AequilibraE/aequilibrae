"""
Shortest path saving.
TODO cython:
 - all in gil land, need to get rid of python things below.

TODO python:
 - need iteration and class name in the path of the file (and pass to cython)
 - make saving directory user configurable
 - need to save compressed graph correspondence once
 -
"""

# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.string cimport string
from string_helper cimport to_string


import sys

import pyarrow as pa
cimport pyarrow as pa
import pyarrow.parquet as pq

import numpy as np
cimport numpy as np
np.import_array()

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

    cdef np.npy_intp dims[1]
    cdef np.ndarray[np.longlong_t, ndim=1] numpy_array

    for node in range(zones):

        # now part of test below
        #if node == origin_index:
        #    continue

        path_for_od_pair_and_class.clear()
        # tracing backwards from each destination for this one-to-all shortest path
        predecessor = pred[node]
        # need to check if disconnected, also makes sure o==d is not included
        if predecessor == -1:
            continue
        connector = conn[node]
        path_for_od_pair_and_class.push_back(connector)

        # print(f" (b) d={node},   pred = {predecessor}, connector = {connector}"); sys.stdout.flush
        while predecessor >= 0:
            # print(f"    d={node},   pred = {predecessor}, connector = {connector}"); sys.stdout.flush
            predecessor = pred[predecessor]
            if predecessor != -1:
                connector = conn[predecessor]
                # need this to avoid ading last element. Would it be faster to resize after loop?
                if connector != -1:
                    path_for_od_pair_and_class.push_back(connector)

        file_name = b'path_saving/test_' + to_string(origin_index) + b"_" + to_string(node) + b'.parquet'

        # print(f"size of path vec {path_for_od_pair_and_class.size()}")

        # get a view on data underlying vector, then as numpy array. avoids copying.
        dims[0] = <np.npy_intp> (path_for_od_pair_and_class.size())
        # print(f"dims = {dims}")

        temp_data = &path_for_od_pair_and_class[0] #.data()
        # print(f"temp[0] = {temp_data[0]}")

        numpy_array = np.PyArray_SimpleNewFromData(1, dims, np.NPY_LONGLONG, temp_data)
        # print(f"np array = {numpy_array}")
        pq.write_table(pa.table({"data": numpy_array}), file_name.decode('utf-8'))


