from aequilibrae.matrix.coo_demand cimport GeneralisedCOODemand
from aequilibrae.paths.cython.route_choice_types cimport (
    RouteVec_t,
    RouteView_t,
    RouteTurnView_t,
    RouteCandidate,
    RouteCandidateSet_t,
)

from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.memory cimport shared_ptr
from libc.stdint cimport *


cdef class RouteChoiceSetResults:
    cdef:
        GeneralisedCOODemand demand
        bint store_results
        bint perform_assignment
        double cutoff_prob
        double beta
        const double[:] cost_view
        const unsigned int [:] mapping_idx
        const int64_t [::] mapping_data
        const int64_t [::] link_id_direction
        const int64_t[::1] full_link_ids
        const int8_t[::1] full_directions

        vector[shared_ptr[RouteVec_t]] __route_vecs
        vector[vector[long long] *] __link_union_set
        vector[shared_ptr[vector[double]]] __cost_set
        vector[shared_ptr[vector[bint]]] __mask_set
        vector[shared_ptr[vector[double]]] __path_overlap_set
        vector[shared_ptr[vector[double]]] __prob_set

        readonly object table

    @staticmethod
    cdef void route_set_to_route_vec(
        RouteVec_t &route_vec,
        vector[vector[double]] &route_turns,
        RouteCandidateSet_t &route_set,
        bint save_turns
    ) noexcept nogil

    cdef shared_ptr[RouteVec_t] get_route_vec(RouteChoiceSetResults self, size_t i) noexcept nogil
    cdef shared_ptr[vector[double]] __get_cost_set(RouteChoiceSetResults self, size_t i) noexcept nogil
    cdef shared_ptr[vector[bint]] __get_mask_set(RouteChoiceSetResults self, size_t i) noexcept nogil
    cdef shared_ptr[vector[double]] __get_path_overlap_set(RouteChoiceSetResults self, size_t i) noexcept nogil
    cdef shared_ptr[vector[double]] get_prob_vec(RouteChoiceSetResults self, size_t i) noexcept nogil

    cdef void store_imported_result(
        self,
        size_t i,
        RouteVec_t &routes,
        const vector[double] &costs, const
        vector[bint] &mask,
        const vector[double] &overlap,
        const vector[double] &probabilities,
        const vector[size_t] &positions
    ) noexcept nogil

    cdef shared_ptr[vector[double]] compute_result(
        RouteChoiceSetResults self,
        size_t i,
        RouteVec_t &route_set,
        const vector[vector[double]] &route_turns,
        bint *found_zero_cost,
        size_t thread_id
    ) noexcept nogil

    @staticmethod
    cdef void compute_psl(
        const RouteView_t &route_set,
        const RouteTurnView_t &route_turns,
        const vector[double] &cost_vec,
        vector[bint] &route_mask,
        vector[double] &path_overlap_vec,
        vector[double] &prob_vec,
        const double[:] cost_view,
        double beta,
        double cutoff_prob
    ) noexcept nogil

    cdef void compute_cost(
        RouteChoiceSetResults self,
        vector[double] &cost_vec,
        const RouteVec_t &route_set,
        const vector[vector[double]] &route_turns,
        const double[:] cost_view,
        bint *found_zero_cost
    ) noexcept nogil

    @staticmethod
    cdef void compute_mask(
        vector[bint] &route_mask,
        const vector[double] &total_cost,
        double cutoff_prob
    ) noexcept nogil

    @staticmethod
    cdef void compute_frequency(
        vector[long long] &keys,
        vector[long long] &counts,
        const RouteView_t &route_set,
        const vector[bint] &route_mask
    ) noexcept nogil

    @staticmethod
    cdef void compute_path_overlap(
        vector[double] &path_overlap_vec,
        const RouteView_t &route_set,
        const vector[long long] &keys,
        const vector[long long] &counts,
        const vector[pair[long long, long long]] &turns,
        const RouteTurnView_t &route_turns,
        const vector[double] &total_cost,
        const vector[bint] &route_mask,
        const double[:] cost_view
    ) noexcept nogil

    @staticmethod
    cdef void compute_prob(
        vector[double] &prob_vec,
        const vector[double] &total_cost,
        const vector[double] &path_overlap_vec,
        const vector[bint] &route_mask,
        double beta
    ) noexcept nogil

    cdef object make_df_from_results(RouteChoiceSetResults self)

cdef void recompute_route_probabilities(
    object df,
    const RouteVec_t &routes,
    const vector[vector[double]] &turns,
    const vector[double] &costs,
    vector[bint] &mask,
    vector[double] &overlap,
    vector[double] &probabilities,
    const double[:] link_costs,
    double beta,
    double cutoff_prob
)

cdef object imported_route_dataframe(
    object df,
    const vector[double] &costs,
    const vector[bint] &mask,
    const vector[double] &overlap,
    const vector[double] &probabilities
)

cdef double inverse_binary_logit(double prob, double beta0, double beta1) noexcept nogil
