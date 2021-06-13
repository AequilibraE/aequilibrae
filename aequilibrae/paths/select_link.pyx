# distutils: language = c++

from libcpp.vector cimport vector
from cython.operator cimport dereference as deref
from cython.parallel import prange

import pyarrow as pa
cimport pyarrow as pa

import pyarrow.parquet as pq
import pyarrow.feather as feather

import numpy as np
cimport numpy as np
np.import_array()

@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False) # turn of bounds-checking for entire function
cpdef void select_link_for_origin_cython(
                                  long long link_id,
                                  long long origin_index,
                                  long long [:] path_links,
                                  long long path_links_size,
                                  long long [:] path_index,
                                  long long path_index_size,
                                  double [:] demand,
                                  double weight,
                                  double [:] select_link,
                                  ) nogil:

    cdef long long i, min_dest, dest_val, val, j
    cdef vector[long long] path_index_to_look_up
    cdef vector[long long] dest_this_o

    for i in range(path_links_size):
        if path_links[i] == link_id:
            path_index_to_look_up.push_back(i)

    for i in range(path_index_to_look_up.size()):
        dest_val = path_index_to_look_up[i]
        min_dest = -1
        for j in range(path_index_size):
            val = path_index[j]
            if val > dest_val:
                if min_dest < 0:
                    min_dest = j
                else:
                    if min_dest > val:
                        min_dest = j
        if min_dest >= 0:
            dest_this_o.push_back(min_dest)

    for i in range(dest_this_o.size()):
        select_link[dest_this_o[i]] += weight * demand[dest_this_o[i]]