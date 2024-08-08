from aequilibrae.matrix.coo_demand cimport GeneralisedCOODemand
from aequilibrae.paths.cython.route_choice_types cimport RouteVec_t
from aequilibrae.matrix.sparse_matrix cimport COO_f64_struct, COO_f32_struct

from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr, make_shared
from libcpp.unordered_set cimport unordered_set
from libcpp cimport bool

# I understand this isn't a great way to handle this but I'm rather unsure of a better method.  We need a method to
# store both float and double loads. Cython doesn't allow us to use fused types (Cython diet templates) in any manner
# that's useful. We can't store them on an object nor put them into a C++ structure, they can only be returned, used as
# a local variable, or passed to a function.
#
# Cython doesn't allow us to write a templated class, we'd have to write an actual C++ class for that. Even if we did
# that we'd still have to write a wrapper class.
cdef class LinkLoadingResults:
    cdef:
        GeneralisedCOODemand demand
        readonly object select_link_set_names
        size_t num_links
        bint sl_link_loading

        # Number of threads
        #               * number of demand cols
        #                 |               * number of links
        vector[unique_ptr[vector[unique_ptr[vector[double]]]]] f64_link_loading_threaded

        # Number of demand cols
        #               * number of links
        vector[unique_ptr[vector[double]]] f64_link_loading

        vector[unique_ptr[vector[unique_ptr[vector[float]]]]] f32_link_loading_threaded
        vector[unique_ptr[vector[float]]] f32_link_loading

        # Select link
        # A select link set is represented by a vector of unordered AND sets, OR'd together
        # Number of select link sets
        #               * number of select link OR sets
        #                 |               * number of AND sets
        vector[unique_ptr[vector[unique_ptr[unordered_set[long long]]]]] select_link_sets

        # Number of select link sets
        #               * number of select link OR sets
        vector[unique_ptr[vector[size_t]]] select_link_set_lengths

        # Number of threads
        #               * number of select link sets
        #                 |               * number of demand cols
        #                 |                 |               * number of links
        vector[unique_ptr[vector[unique_ptr[vector[unique_ptr[vector[double]]]]]]] f64_sl_link_loading_threaded

        # Number of select link sets
        #               * number of demand cols
        #                 |               * number of links
        vector[unique_ptr[vector[unique_ptr[vector[double]]]]]  f64_sl_link_loading

        vector[unique_ptr[vector[unique_ptr[vector[unique_ptr[vector[float]]]]]]] f32_sl_link_loading_threaded
        vector[unique_ptr[vector[unique_ptr[vector[float]]]]] f32_sl_link_loading

        # Number of threads
        #               * number of select link sets
        #                 |               * number of demand cols
        vector[unique_ptr[vector[unique_ptr[vector[COO_f64_struct]]]]] f64_sl_od_matrix_threaded

        # Number of select link sets
        #               * number of demand cols
        vector[unique_ptr[vector[COO_f64_struct]]] f64_sl_od_matrix

        vector[unique_ptr[vector[unique_ptr[vector[COO_f32_struct]]]]] f32_sl_od_matrix_threaded
        vector[unique_ptr[vector[COO_f32_struct]]] f32_sl_od_matrix

        readonly object link_loading_objects
        readonly object sl_link_loading_objects
        readonly object od_matrix_objects

    cdef void link_load_single_route_set(
        LinkLoadingResults self,
        const size_t od_idx,
        const RouteVec_t &route_set,
        const vector[double] &prob_vec,
        const size_t thread_id
    ) noexcept nogil

    cdef void reduce_link_loading(LinkLoadingResults self)
    cdef object apply_generic_link_loading(
        LinkLoadingResults self,
        vector[unique_ptr[vector[double]]] &f64_link_loading,
        vector[unique_ptr[vector[float]]] &f32_link_loading,
        long long[:] compressed_id_view,
        int cores
    )

    @staticmethod
    cdef bool is_in_select_link_set(
        vector[long long] &route,
        const vector[unique_ptr[unordered_set[long long]]] &select_link_set,
        const vector[size_t] &select_link_set_lengths
    ) noexcept nogil
    cdef void sl_link_load_single_route_set(
        LinkLoadingResults self,
        const size_t od_idx,
        const RouteVec_t &route_set,
        const vector[double] &prob_vec,
        const long long origin_idx,
        const long long dest_idx,
        const size_t thread_id
    ) noexcept nogil
    cdef void reduce_sl_link_loading(LinkLoadingResults self)
    cdef void reduce_sl_od_matrix(LinkLoadingResults self)
    cdef object link_loading_to_objects(self, long long[:] compressed_id_view, int cores)
    cdef object sl_link_loading_to_objects(self, long long[:] compressed_id_view, int cores)
    cdef object sl_od_matrices_structs_to_objects(LinkLoadingResults self)
