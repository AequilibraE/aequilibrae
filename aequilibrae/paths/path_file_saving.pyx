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

import pyarrow as pa
cimport pyarrow as pa

# need to decide or make optional which format we want
import pyarrow.parquet as pq
import pyarrow.feather as feather

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
                          long long [:] conn,
                          string path_file,
                          string index_file): #nogil:

    cdef long long class_, node, predecessor, connector, ctr
    cdef string file_name
    cdef vector[long long] path_data
    # could make this an ndarray and not do the conversion, we know the size of the index array is zones
    cdef vector[long long] size_of_path_arrays
    cdef long long* temp_data
    cdef long long* temp_data_ind

    cdef np.npy_intp dims[1]
    cdef np.ndarray[np.longlong_t, ndim=1] numpy_array
    cdef np.npy_intp dims_ind[1]
    cdef np.ndarray[np.longlong_t, ndim=1] numpy_array_ind

    for node in range(zones):
        predecessor = pred[node]
        # need to check if disconnected, also makes sure o==d is not included
        if predecessor == -1:
            continue
        connector = conn[node]
        path_data.push_back(connector)

        # print(f" (b) d={node},   pred = {predecessor}, connector = {connector}"); sys.stdout.flush
        while predecessor >= 0:``
            # print(f"    d={node},   pred = {predecessor}, connector = {connector}"); sys.stdout.flush
            predecessor = pred[predecessor]
            if predecessor != -1:
                connector = conn[predecessor]
                # need this to avoid ading last element. Would it be faster to resize after loop?
                if connector != -1:
                    path_data.push_back(connector)

        # print(f"size of path vec {path_for_od_pair_and_class.size()}")


        size_of_path_arrays.push_back(<np.longlong_t> path_data.size())

    # get a view on data underlying vector, then as numpy array. avoids copying.
    dims[0] = <np.npy_intp> (path_data.size())
    dims_ind[0] = <np.npy_intp> (size_of_path_arrays.size())
    # print(f"dims = {dims}")

    temp_data = &path_data[0]
    temp_data_ind = &size_of_path_arrays[0]
    # print(f"temp[0] = {temp_data[0]}")

    numpy_array = np.PyArray_SimpleNewFromData(1, dims, np.NPY_LONGLONG, temp_data)
    numpy_array_ind = np.PyArray_SimpleNewFromData(1, dims, np.NPY_LONGLONG, temp_data_ind)
    # print(f"np array = {numpy_array}")


    # parquet
    #file_name = path_file_base + to_string(node) + b'.parquet'
    #pq.write_table(pa.table({"data": numpy_array}), file_name.decode('utf-8'))
    # feather
    # file_name = path_file_base + to_string(node) + b'.feather'

    feather.write_feather(pa.table({"data": numpy_array}), path_file.decode('utf-8'))
    feather.write_feather(pa.table({"data": numpy_array_ind}), index_file.decode('utf-8'))

