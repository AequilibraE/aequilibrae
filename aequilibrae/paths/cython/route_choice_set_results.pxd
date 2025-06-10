from aequilibrae.matrix.coo_demand cimport GeneralisedCOODemand
from aequilibrae.paths.cython.route_choice_types cimport (
    RouteVec_t,
    RouteSet_t,
)

from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libc.stdint cimport *


cdef class RouteChoiceSetResults:
    cdef:
        GeneralisedCOODemand demand
        bint store_results
        bint perform_assignment
        double cutoff_prob
        double beta
        double[:] cost_view
        unsigned int [:] mapping_idx
        int64_t [::] mapping_data

        vector[shared_ptr[RouteVec_t]] __route_vecs
        vector[vector[long long] *] __link_union_set
        vector[shared_ptr[vector[double]]] __cost_set
        vector[shared_ptr[vector[bint]]] __mask_set
        vector[shared_ptr[vector[double]]] __path_overlap_set
        vector[shared_ptr[vector[double]]] __prob_set

        readonly object table

    @staticmethod
    cdef void route_set_to_route_vec(RouteVec_t &route_vec, RouteSet_t &route_set) noexcept nogil

    cdef shared_ptr[RouteVec_t] get_route_vec(RouteChoiceSetResults self, size_t i) noexcept nogil
    cdef shared_ptr[vector[double]] __get_cost_set(RouteChoiceSetResults self, size_t i) noexcept nogil
    cdef shared_ptr[vector[bint]] __get_mask_set(RouteChoiceSetResults self, size_t i) noexcept nogil
    cdef shared_ptr[vector[double]] __get_path_overlap_set(RouteChoiceSetResults self, size_t i) noexcept nogil
    cdef shared_ptr[vector[double]] get_prob_vec(RouteChoiceSetResults self, size_t i) noexcept nogil

    cdef shared_ptr[vector[double]] compute_result(
        RouteChoiceSetResults self,
        size_t i,
        RouteVec_t &route_set,
        bint *found_zero_cost,
        size_t thread_id
    ) noexcept nogil

    cdef void compute_cost(
        RouteChoiceSetResults self,
        vector[double] &cost_vec,
        const RouteVec_t &route_set,
        const double[:] cost_view,
        bint *found_zero_cost
    ) noexcept nogil

    cdef void compute_mask(
        RouteChoiceSetResults self,
        vector[bint] &route_mask,
        const vector[double] &total_cost
    ) noexcept nogil

    cdef void compute_frequency(
        RouteChoiceSetResults self,
        vector[long long] &keys,
        vector[long long] &counts,
        const RouteVec_t &route_set,
        const vector[bint] &route_mask
    ) noexcept nogil

    cdef void compute_path_overlap(
        RouteChoiceSetResults self,
        vector[double] &path_overlap_vec,
        const RouteVec_t &route_set,
        const vector[long long] &keys,
        const vector[long long] &counts,
        const vector[double] &total_cost,
        const vector[bint] &route_mask,
        const double[:] cost_view
    ) noexcept nogil

    cdef void compute_prob(
        RouteChoiceSetResults self,
        vector[double] &prob_vec,
        const vector[double] &total_cost,
        const vector[double] &path_overlap_vec,
        const vector[bint] &route_mask
    ) noexcept nogil

    cdef object make_df_from_results(RouteChoiceSetResults self)

cdef double inverse_binary_logit(double prob, double beta0, double beta1) noexcept nogil
