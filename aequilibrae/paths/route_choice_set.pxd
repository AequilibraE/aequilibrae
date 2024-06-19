# cython: language_level=3str
from aequilibrae.paths.results import PathResults
from aequilibrae.matrix.sparse_matrix cimport COO

from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair
from libcpp cimport bool

cimport numpy as np  # Numpy *must* be cimport'd BEFORE pyarrow.lib, there's nothing quite like Cython.
cimport pyarrow as pa
cimport pyarrow.lib as libpa
cimport pyarrow._dataset_parquet as pq
from libcpp.memory cimport shared_ptr

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

        vector[pair[long long, long long]] *ods
        vector[RouteSet_t *] *results
        vector[vector[long long] *] *link_union_set
        vector[vector[double] *] *cost_set
        vector[vector_bool_ptr] *mask_set
        vector[vector[double] *] *path_overlap_set
        vector[vector[double] *] *prob_set

        unsigned int [:] mapping_idx
        unsigned int [:] mapping_data

    cdef void deallocate(RouteChoiceSet self) nogil

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

    cdef RouteSet_t *bfsle(
        RouteChoiceSet self,
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

    cdef RouteSet_t *link_penalisation(
        RouteChoiceSet self,
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

    @staticmethod
    cdef pair[vector[long long] *, vector[long long] *] compute_frequency(RouteSet_t *route_set, vector[bool] &route_mask) noexcept nogil

    @staticmethod
    cdef vector[double] *compute_cost(RouteSet_t *route_sets, double[:] cost_view) noexcept nogil

    @staticmethod
    cdef vector[bool] *compute_mask(double cutoff_prob, vector[double] &total_cost) noexcept nogil

    @staticmethod
    cdef vector[double] *compute_path_overlap(
        RouteSet_t *route_set,
        pair[vector[long long] *, vector[long long] *] &freq_set,
        vector[double] &total_cost,
        vector[bool] &route_mask,
        double[:] cost_view
    ) noexcept nogil

    @staticmethod
    cdef vector[double] *compute_prob(
        vector[double] &total_cost,
        vector[double] &path_overlap_vec,
        vector[bool] &route_mask,
        double beta
    ) noexcept nogil

    @staticmethod
    cdef vector[vector[double] *] *compute_path_files(
        vector[pair[long long, long long]] &ods,
        vector[RouteSet_t *] &results,
        vector[vector[long long] *] &link_union_set,
        vector[vector[double] *] &prob_set,
        unsigned int cores
    ) noexcept nogil

    cdef vector[double] *apply_link_loading(RouteChoiceSet self, double[:, :] matrix_view) noexcept nogil
    cdef vector[double] *apply_link_loading_from_path_files(RouteChoiceSet self, double[:, :] matrix_view, vector[vector[double] *] &path_files) noexcept nogil
    cdef apply_link_loading_func(RouteChoiceSet self, vector[double] *ll, int cores)

    cdef vector[double] *apply_select_link_loading(
        RouteChoiceSet self,
        COO sparse_mat,
        double[:, :] matrix_view,
        vector[unordered_set[long long] *] &select_link_set,
        vector[size_t] select_link_set_lengths
    ) noexcept nogil

    @staticmethod
    cdef bool is_in_select_link(
        vector[long long] &route,
        vector[unordered_set[long long] *] &select_link_set,
        vector[size_t] &select_link_set_lengths
    ) noexcept nogil

    cdef shared_ptr[libpa.CTable] make_table_from_results(
        RouteChoiceSet self,
        vector[pair[long long, long long]] &ods,
        vector[RouteSet_t *] &route_sets,
        vector[vector[double] *] *cost_set,
        vector[vector_bool_ptr] *mask_set,
        vector[vector[double] *] *path_overlap_set,
        vector[vector[double] *] *prob_set
    )

cdef class Checkpoint:
    cdef:
        public object where
        public object schema
        public object partition_cols
