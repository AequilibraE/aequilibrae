# distutils: language = c++

cimport cython
from libcpp.vector cimport vector
from libc.stdint cimport int64_t, uint32_t

import numpy as np
import pandas as pd


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)  # turn of bounds-checking for entire function
cpdef void save_path_file(
    long origin_index,
    long num_links,
    long zones,
    long long [:] pred,
    long long [:] conn,
    str path_file,
    str index_file,
    bint write_feather,
    const long long [:] arc_pred=None,
    const uint32_t [:] mapping_idx=None,
    const int64_t [:] mapping_data=None,
) noexcept:

    cdef long long node, predecessor, connector, cur_arc, link_idx, steps, max_steps
    cdef vector[long long] path_data
    # could make this an ndarray and not do the conversion, we know the size of the index array is zones
    cdef vector[long long] size_of_path_arrays
    cdef bint has_arc_pred = (arc_pred is not None and arc_pred.shape[0] > 0)
    cdef bint has_mapping = (mapping_idx is not None and mapping_data is not None and mapping_idx.shape[0] > 0)
    cdef uint32_t m_start, m_end

    with nogil:
        for node in range(zones):
            predecessor = pred[node]
            # need to check if disconnected, also makes sure o==d is not included
            if predecessor == -1:
                size_of_path_arrays.push_back(<long long> path_data.size())  # need to store index here
                continue

            if has_arc_pred:
                cur_arc = conn[node]
                steps = 0
                max_steps = <long long>arc_pred.shape[0]
                while cur_arc >= 0 and steps < max_steps:
                    steps += 1
                    if has_mapping:
                        m_start = mapping_idx[cur_arc]
                        m_end = mapping_idx[cur_arc + 1]
                        if m_end > m_start:
                            link_idx = <long long>(m_end - 1)
                            while link_idx >= <long long>m_start:
                                path_data.push_back(mapping_data[link_idx])
                                link_idx -= 1
                    else:
                        path_data.push_back(cur_arc)
                    cur_arc = arc_pred[cur_arc]
            else:
                connector = conn[node]
                path_data.push_back(connector)
                while predecessor != -1:
                    connector = conn[predecessor]  # connector has to be looked up BEFORE predecessor update
                    predecessor = pred[predecessor]
                    if (predecessor != -1) and (connector != -1):
                        path_data.push_back(connector)

            size_of_path_arrays.push_back(<long long> path_data.size())

    # get a view on data underlying vector, then as numpy array. avoids copying.
    numpy_array = np.asarray(<long long[:path_data.size()]>path_data.data())
    numpy_array_ind = np.asarray(<long long[:size_of_path_arrays.size()]>size_of_path_arrays.data())

    table1 = pd.DataFrame({"data": numpy_array})
    table2 = pd.DataFrame({"data": numpy_array_ind})

    if write_feather:
        table1.to_feather(path_file)
        table2.to_feather(index_file)
    else:
        table1.to_parquet(path_file)
        table2.to_parquet(index_file)
