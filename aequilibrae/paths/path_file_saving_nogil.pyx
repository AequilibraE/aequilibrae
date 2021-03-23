"""
Shortest path saving.

TODO python:
 - make saving directory user configurable
 - need to save compressed graph correspondence once
"""

# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdint cimport int64_t

cdef extern from "ParquetWriter.cpp":
    pass

cdef extern from "ParquetWriter.h":
    cdef cppclass ParquetWriter nogil:
        ParquetWriter() except +
        int write_parquet(vector[int64_t] vec, string filename)
        int write_feather(vector[int64_t] vec, string filename)


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
                          bool write_feather) nogil:

    cdef long long class_, node, predecessor, connector, ctr
    cdef string file_name
    cdef vector[int64_t] path_data
    cdef vector[int64_t] size_of_path_arrays

    for node in range(zones):
        predecessor = pred[node]
        # need to check if disconnected, also makes sure o==d is not included
        if predecessor == -1:
            continue
        connector = conn[node]
        path_data.push_back(connector)

        # print(f" (b) d={node},   pred = {predecessor}, connector = {connector}"); sys.stdout.flush
        while predecessor >= 0:
            # print(f"    d={node},   pred = {predecessor}, connector = {connector}"); sys.stdout.flush
            predecessor = pred[predecessor]
            if predecessor != -1:
                connector = conn[predecessor]
                # need this to avoid ading last element. Would it be faster to resize after loop?
                if connector != -1:
                    path_data.push_back(<int64_t> connector)

        # print(f"size of path vec {path_for_od_pair_and_class.size()}")


        # size_of_path_arrays.push_back(<np.longlong_t> path_data.size())
        size_of_path_arrays.push_back(<int64_t> path_data.size())


    cdef ParquetWriter* writer = new ParquetWriter()
    if write_feather:
        writer.write_feather(path_data, path_file)
        writer.write_feather(size_of_path_arrays, index_file)
    else:
        writer.write_parquet(path_data, path_file)
        writer.write_parquet(size_of_path_arrays, index_file)
    del writer
