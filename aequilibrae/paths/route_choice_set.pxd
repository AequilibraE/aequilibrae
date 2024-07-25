# cython: language_level=3str
from aequilibrae.paths.results import PathResults
from aequilibrae.matrix.sparse_matrix cimport COO, COO_f64_struct, COO_f32_struct

from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair
from libcpp cimport bool

cimport numpy as np  # Numpy *must* be cimport'd BEFORE pyarrow.lib, there's nothing quite like Cython.
cimport pyarrow as pa
cimport pyarrow.lib as libpa
cimport pyarrow._dataset_parquet as pq
from libcpp.memory cimport shared_ptr, unique_ptr, make_shared, make_unique
from libcpp.utility cimport move

# std::linear_congruential_engine is not available in the Cython libcpp.random shim. We'll import it ourselves
# from libcpp.random cimport minstd_rand
from libc.stdint cimport *

cdef extern from "<random>" namespace "std" nogil:
    cdef cppclass random_device:
        ctypedef uint_fast32_t result_type
        random_device() except +
        result_type operator()() except +

    cdef cppclass minstd_rand:
        ctypedef uint_fast32_t result_type
        minstd_rand() except +
        minstd_rand(result_type seed) except +
        result_type operator()() except +
        result_type min() except +
        result_type max() except +
        void discard(size_t z) except +
        void seed(result_type seed) except +

# std::shuffle is not available in the Cython libcpp.algorithm shim. We'll import it ourselves
# from libcpp.algorithm cimport shuffle
cdef extern from "<algorithm>" namespace "std" nogil:
    void shuffle[RandomIt, URBG](RandomIt first, RandomIt last, URBG&& g) except +

# std::make_pair is not available in the Cython libcpp.utilities shim. We'll import it ourselves based on the C++11 til C++14
# definition because C++14 makes this signature weird
# See https://github.com/cython/cython/issues/2706
cdef extern from "<utility>" namespace "std" nogil:
    pair[T, U] make_pair[T, U](T&& t, U&& u)

# To define our own hashing functions we have to write a little C++. The string is inlined directly into route_choice.cpp
# To make Cython aware of our hash types we also must declare them with the right signatures
#
# OrderedVectorPointerHasher: This hash function is for hashing the routes, thus it should be order *DEPENDENT*.
# Potential performance improvements may come from https://en.wikipedia.org/wiki/Universal_hashing#Hashing_vectors
#
# UnorderedSetPointerHasher: This hash function is for hashing the banned route sets, thus it should be order *INDEPENDENT*.
# Potential performance improvements may come from:
# Mark N. Wegman, J.Lawrence Carter,
# New hash functions and their use in authentication and set equality
# https://doi.org/10.1016/0022-0000(81)90033-7
#
# PointerDereferenceEqualTo: Because we are storing and hashing the pointers to objects to avoid unnecessary copies we must
# define our own comparator to resolve hash collisions. Without this equality operator the bare pointers are compared.
cdef extern from * nogil:
    """
    // Source: https://stackoverflow.com/a/72073933
    // License: CC BY-SA 4.0 Deed, https://creativecommons.org/licenses/by-sa/4.0/
    struct OrderedVectorPointerHasher {
        size_t operator()(const std::vector<long long> *V) const {
            size_t seed = V->size();
            long long x;
            for(auto &i : *V) {
                x = ((i >> 16) ^ i) * 0x45d9f3b;
                x = ((x >> 16) ^ x) * 0x45d9f3b;
                x = (x >> 16) ^ x;
                seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    // Source: https://stackoverflow.com/a/1537189
    // License: CC BY-SA 2.5 Deed, https://creativecommons.org/licenses/by-sa/2.5/
    struct UnorderedSetPointerHasher {
        size_t operator()(const std::unordered_set<long long> *S) const {
            size_t hash = 1;
            for(auto &i : *S) {
                hash *= 1779033703 + 2 * i;
            }
            return hash / 2;
        }
    };

    template<class T>
    struct PointerDereferenceEqualTo {
        bool operator()(const T& lhs, const T& rhs) const {
            return *lhs == *rhs;
        }
    };
    """
    cppclass OrderedVectorPointerHasher:
        size_t operator()(const vector[long long] *V) const

    cppclass UnorderedSetPointerHasher:
        size_t operator()(const unordered_set[long long] *S) const

    cppclass PointerDereferenceEqualTo[T]:
        bool operator()(const T& lhs, const T& rhs) const


# For typing (haha) convenience, the types names are getting long
ctypedef unordered_set[vector[long long] *, OrderedVectorPointerHasher, PointerDereferenceEqualTo[vector[long long] *]] RouteSet_t
ctypedef unordered_set[unordered_set[long long] *, UnorderedSetPointerHasher, PointerDereferenceEqualTo[unordered_set[long long] *]] LinkSet_t
ctypedef vector[pair[unordered_set[long long] *, vector[long long] *]] RouteMap_t

ctypedef vector[unique_ptr[vector[long long]]] RouteVec_t



# A (known 2016) bug in the Cython compiler means it incorrectly parses the following type when used in a cdef
# https://github.com/cython/cython/issues/534
ctypedef vector[bool]* vector_bool_ptr

# Pyarrow's Cython API does not provide all the functions available in the C++ API, some of them are really useful.
# Here we redeclare the classes with the functions we want, these are available in the current namespace, *not* libarrow
cdef extern from "arrow/builder.h" namespace "arrow" nogil:

    cdef cppclass CUInt32Builder" arrow::UInt32Builder"(libpa.CArrayBuilder):
        CUInt32Builder(libpa.CMemoryPool* pool)
        libpa.CStatus Append(const uint32_t value)
        libpa.CStatus AppendValues(const vector[uint32_t] &values)
        libpa.CStatus AppendValues(vector[uint32_t].const_reverse_iterator values_begin, vector[uint32_t].const_reverse_iterator values_end)
        libpa.CStatus AppendValues(const uint32_t *values, int64_t length, const uint8_t *valid_bytes = nullptr)

    cdef cppclass CDoubleBuilder" arrow::DoubleBuilder"(libpa.CArrayBuilder):
        CDoubleBuilder(libpa.CMemoryPool* pool)
        libpa.CStatus Append(const double value)
        libpa.CStatus AppendValues(const vector[double] &values)

    cdef cppclass CBooleanBuilder" arrow::BooleanBuilder"(libpa.CArrayBuilder):
        CBooleanBuilder(libpa.CMemoryPool* pool)
        libpa.CStatus Append(const bool value)
        libpa.CStatus AppendValues(const vector[bool] &values)


cdef class RouteChoiceSet:
    cdef:
        double [:] cost_view
        long long [:] graph_fs_view
        long long [:] b_nodes_view
        long long [:] nodes_to_indices_view
        double [:] lat_view
        double [:] lon_view
        long long [:] ids_graph_view
        long long [:] graph_compressed_id_view
        long long [:] compressed_link_ids
        long long num_nodes
        long long num_links
        long long zones
        bint block_flows_through_centroids
        bint a_star

        unsigned int [:] mapping_idx
        unsigned int [:] mapping_data

        readonly RouteChoiceSetResults results
        readonly LinkLoadingResults ll_results

    cdef void path_find(
        RouteChoiceSet self,
        long origin_index,
        long dest_index,
        double [:] scratch_cost,
        long long [:] thread_predecessors,
        long long [:] thread_conn,
        long long [:] thread_b_nodes,
        long long [:] thread_reached_first
    ) noexcept nogil

    cdef void bfsle(
        RouteChoiceSet self,
        RouteSet_t &route_set,
        long origin_index,
        long dest_index,
        unsigned int max_routes,
        unsigned int max_depth,
        unsigned int max_misses,
        double [:] thread_cost,
        long long [:] thread_predecessors,
        long long [:] thread_conn,
        long long [:] thread_b_nodes,
        long long [:] _thread_reached_first,
        double penatly,
        unsigned int seed
    ) noexcept nogil

    cdef void link_penalisation(
        RouteChoiceSet self,
        RouteSet_t &route_set,
        long origin_index,
        long dest_index,
        unsigned int max_routes,
        unsigned int max_depth,
        unsigned int max_misses,
        double [:] thread_cost,
        long long [:] thread_predecessors,
        long long [:] thread_conn,
        long long [:] thread_b_nodes,
        long long [:] _thread_reached_first,
        double penatly,
        unsigned int seed
    ) noexcept nogil

    # @staticmethod
    # cdef vector[vector[double] *] *compute_path_files(
    #     vector[pair[long long, long long]] &ods,
    #     vector[RouteSet_t *] &results,
    #     vector[vector[long long] *] &link_union_set,
    #     vector[vector[double] *] &prob_set,
    #     unsigned int cores
    # ) noexcept nogil

    # cdef vector[double] *apply_link_loading(RouteChoiceSet self, double[:, :] matrix_view) noexcept nogil
    # cdef vector[double] *apply_link_loading_from_path_files(RouteChoiceSet self, double[:, :] matrix_view, vector[vector[double] *] &path_files) noexcept nogil
    # cdef apply_link_loading_func(RouteChoiceSet self, vector[double] *ll, int cores)

    # cdef vector[double] *apply_select_link_loading(
    #     RouteChoiceSet self,
    #     COO sparse_mat,
    #     double[:, :] matrix_view,
    #     unordered_set[long] &select_link_set
    # ) noexcept nogil


cdef class Checkpoint:
    cdef:
        public object where
        public object schema
        public object partition_cols


cdef class RouteChoiceSetResults:
    cdef:
        GeneralisedCOODemand demand
        bool store_results
        bool perform_assignment
        double cutoff_prob
        double beta
        double[:] cost_view
        unsigned int [:] mapping_idx
        unsigned int [:] mapping_data

        vector[shared_ptr[RouteVec_t]] __route_vecs
        vector[vector[long long] *] __link_union_set
        vector[shared_ptr[vector[double]]] __cost_set
        vector[shared_ptr[vector[bool]]] __mask_set
        vector[shared_ptr[vector[double]]] __path_overlap_set
        vector[shared_ptr[vector[double]]] __prob_set

        readonly object table

    @staticmethod
    cdef void route_set_to_route_vec(RouteVec_t &route_vec, RouteSet_t &route_set) noexcept nogil

    cdef shared_ptr[RouteVec_t] get_route_vec(RouteChoiceSetResults self, size_t i) noexcept nogil
    cdef shared_ptr[vector[double]] __get_cost_set(RouteChoiceSetResults self, size_t i) noexcept nogil
    cdef shared_ptr[vector[bool]] __get_mask_set(RouteChoiceSetResults self, size_t i) noexcept nogil
    cdef shared_ptr[vector[double]] __get_path_overlap_set(RouteChoiceSetResults self, size_t i) noexcept nogil
    cdef shared_ptr[vector[double]] __get_prob_set(RouteChoiceSetResults self, size_t i) noexcept nogil

    cdef shared_ptr[vector[double]] compute_result(
        RouteChoiceSetResults self,
        size_t i,
        RouteVec_t &route_set,
        size_t thread_id
    ) noexcept nogil

    cdef void compute_cost(
        RouteChoiceSetResults self,
        vector[double] &cost_vec,
        const RouteVec_t &route_set,
        const double[:] cost_view
    ) noexcept nogil

    cdef bool compute_mask(
        RouteChoiceSetResults self,
        vector[bool] &route_mask,
        const vector[double] &total_cost
    ) noexcept nogil

    cdef void compute_frequency(
        RouteChoiceSetResults self,
        vector[long long] &keys,
        vector[long long] &counts,
        const RouteVec_t &route_set,
        const vector[bool] &route_mask
    ) noexcept nogil

    cdef void compute_path_overlap(
        RouteChoiceSetResults self,
        vector[double] &path_overlap_vec,
        const RouteVec_t &route_set,
        const vector[long long] &keys,
        const vector[long long] &counts,
        const vector[double] &total_cost,
        const vector[bool] &route_mask,
        const double[:] cost_view
    ) noexcept nogil

    cdef void compute_prob(
        RouteChoiceSetResults self,
        vector[double] &prob_vec,
        const vector[double] &total_cost,
        const vector[double] &path_overlap_vec,
        const vector[bool] &route_mask
    ) noexcept nogil

    cdef object make_table_from_results(RouteChoiceSetResults self)

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
