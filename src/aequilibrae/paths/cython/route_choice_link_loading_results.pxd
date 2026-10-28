from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector


from aequilibrae.paths.cython.outputs cimport (
    LoadingOutputs,
    SelectLinkLoadingOutputs,
    SelectLinkODOutputs,
    CppLoadingOutputs,
    CppSelectLinkLoadingOutputsView,
    CppSelectLinkODOutputsView,
)
from aequilibrae.paths.cython.route_choice_types cimport RouteVec_t

cdef class LinkLoadingResults:
    cdef:
        object field_names
        object index
        object index_name
        readonly object select_link_set_names
        size_t num_links
        bint sl_link_loading
        list workers

        LoadingOutputs loading
        SelectLinkLoadingOutputs selected_loading
        SelectLinkODOutputs selected_od
        vector[CppLoadingOutputs[double]] loading_views
        vector[CppSelectLinkLoadingOutputsView[double]] selected_views
        vector[CppSelectLinkODOutputsView[double]] od_views

        # A select link set is represented by a vector of unordered AND sets, OR'd together.
        # Number of select link sets * number of OR sets * number of AND members.
        vector[unique_ptr[vector[unique_ptr[unordered_set[long long]]]]] select_link_sets
        vector[unique_ptr[vector[size_t]]] select_link_set_lengths

        readonly object link_loading_objects
        readonly object sl_link_loading_objects
        readonly object od_matrix_objects

    cdef void link_load_single_route_set(
        self,
        const double *values,
        size_t field_count,
        const RouteVec_t &route_set,
        const vector[double] &prob_vec,
        size_t thread_id,
    ) noexcept nogil
    cdef void reduce_link_loading(self)
    cdef object project_loads(self, object loads, const long long[::1] compressed_id_view, int cores)

    @staticmethod
    cdef bool is_in_select_link_set(
        vector[long long] &route,
        const vector[unique_ptr[unordered_set[long long]]] &select_link_set,
        const vector[size_t] &select_link_set_lengths,
    ) noexcept nogil

    cdef void sl_link_load_single_route_set(
        self,
        size_t origin,
        size_t destination,
        const double *values,
        size_t field_count,
        const RouteVec_t &route_set,
        const vector[double] &prob_vec,
        size_t thread_id,
    ) noexcept nogil

    cdef void reduce_sl_link_loading(self)
    cdef void reduce_sl_od_matrix(self)
    cdef object link_loading_to_objects(self, const long long[::1] compressed_id_view, int cores)
    cdef object sl_link_loading_to_objects(self, const long long[::1] compressed_id_view, int cores)
    cdef object sl_od_matrices_structs_to_objects(self)
