from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport pair

cdef class GeneralisedCOODemand:
    cdef:
        public object df
        readonly object f64_names
        readonly object f32_names
        readonly object shape
        readonly object nodes_to_indices
        vector[pair[long long, long long]] ods
        vector[unique_ptr[vector[double]]] f64
        vector[unique_ptr[vector[float]]] f32
