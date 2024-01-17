# cython: language_level=3str
from aequilibrae.paths.results import PathResults
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair
from libcpp cimport bool

# std::linear_congruential_engine is not available in the Cython libcpp.random shim. We'll import it ourselves
# from libcpp.random cimport minstd_rand
from libc.stdint cimport uint_fast32_t, uint_fast64_t

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

# To define our own hashing functions we have to write a little cpp. The string is inlined directly into route_choice.cpp
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
# PointerDereferenceEqualTo: Because we are storing and hashing the pointers to objects to avoid unnessecary copies we must
# define our own comparitor to resolve hash collisions. Without this equaility operator the bare pointers are compared.
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


# For typing (haha) convenince, the types names are getting long
ctypedef unordered_set[vector[long long] *, OrderedVectorPointerHasher, PointerDereferenceEqualTo[vector[long long] *]] RouteSet_t
ctypedef unordered_set[unordered_set[long long] *, UnorderedSetPointerHasher, PointerDereferenceEqualTo[unordered_set[long long] *]] LinkSet_t
ctypedef vector[pair[unordered_set[long long] *, vector[long long] *]] RouteMap_t


cdef class RouteChoice:
    cdef:
        double [:] cost_view
        long long [:] graph_fs_view
        long long [:] b_nodes_view
        long long [:] nodes_to_indices_view
        double [:] lat_view
        double [:] lon_view
        long long [:] predecessors_view
        long long [:] ids_graph_view
        long long [:] conn_view
        long long [:] compressed_link_ids
    cdef void path_find(RouteChoice self, long origin_index, long dest_index, double [:] scratch_cost) noexcept nogil
    cdef RouteSet_t *generate_route_set(RouteChoice self, long origin_index, long dest_index, unsigned int max_routes, unsigned int max_depth, double [:] scratch_cost) noexcept nogil
