# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.string cimport string
import importlib.util as iutil
spec = iutil.find_spec("pyarrow")

if spec is not None:
    import pyarrow as pa
    cimport pyarrow as pa
    # need to decide or make optional which format we want
    import pyarrow.parquet as pq
    import pyarrow.feather as feather

import numpy as np
cimport numpy as np
np.import_array()

@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False) # turn of bounds-checking for entire function
cpdef void save_path_file(long origin_index,
                          long num_links,
                          long zones,
                          long long [:] pred,
                          long long [:] conn,
                          string path_file,
                          string index_file,
                          bool write_feather) noexcept:

    cdef long long class_, node, predecessor, connector, ctr
    cdef vector[long long] path_data
    # could make this an ndarray and not do the conversion, we know the size of the index array is zones
    cdef vector[long long] size_of_path_arrays

    cdef np.npy_intp dims[1]
    cdef np.ndarray[np.longlong_t, ndim=1] numpy_array
    cdef np.npy_intp dims_ind[1]
    cdef np.ndarray[np.longlong_t, ndim=1] numpy_array_ind

    with nogil:
        for node in range(zones):
            predecessor = pred[node]
            # need to check if disconnected, also makes sure o==d is not included
            if predecessor == -1:
                size_of_path_arrays.push_back(<np.longlong_t> path_data.size())  # need to store index here
                continue
            connector = conn[node]
            path_data.push_back(connector)
            while predecessor != -1:
                connector = conn[predecessor] # connector has to be looked up BEFORE predecessor update
                predecessor = pred[predecessor]
                if (predecessor != -1) and (connector != -1):
                    path_data.push_back(connector)

            size_of_path_arrays.push_back(<np.longlong_t> path_data.size())

    # get a view on data underlying vector, then as numpy array. avoids copying.
    dims[0] = <np.npy_intp> (path_data.size())
    dims_ind[0] = <np.npy_intp> (size_of_path_arrays.size())
    numpy_array = np.PyArray_SimpleNewFromData(1, dims, np.NPY_LONGLONG, &path_data[0])
    numpy_array_ind = np.PyArray_SimpleNewFromData(1, dims_ind, np.NPY_LONGLONG, &size_of_path_arrays[0])

    if write_feather:
        feather.write_feather(pa.table({"data": numpy_array}), path_file.decode('utf-8'))
        feather.write_feather(pa.table({"data": numpy_array_ind}), index_file.decode('utf-8'))
    else:
        pq.write_table(pa.table({"data": numpy_array}), path_file.decode('utf-8'))
        pq.write_table(pa.table({"data": numpy_array_ind}), index_file.decode('utf-8'))
