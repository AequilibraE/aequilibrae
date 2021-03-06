# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.string cimport string


#import numpy as np
#cimport numpy as np
#
#import pyarrow as pa
#cimport pyarrow as pa
#pa.get_include()
#import pyarrow.parquet as pq
## from pyarrow.lib cimport *


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False) # turn of bounds-checking for entire function
cpdef void save_path_file(long classes,
                          long origin_index,
                          long num_links,
                          long zones,
                          long long [:] pred,
                          long long [:] conn) nogil:

    cdef long long node, predecessor, connector, ctr

    cdef vector[long long] path_for_od_pair_and_class


    for class_ in range(classes):
        for node in range(zones):
            # tracing backwards from each destination for this one-to-all shortest path
            predecessor = pred[node]
            connector = conn[node]
            while predecessor >= 0:
                path_for_od_pair_and_class.push_back(connector)
                predecessor = pred[predecessor]
                connector = conn[predecessor]
                ctr += 1

            cdef string file_name = "test_" + to_string(origin_index) + "_" + to_string(node) + "_"
                                    + to_string(class_);

        #pq.write_table(pa.array(path_for_origin[0:ctr]), '~/test.parquet')


