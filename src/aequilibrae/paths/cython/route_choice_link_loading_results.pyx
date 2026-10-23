cimport cython
from cython.operator cimport dereference as d
from cython.operator cimport predecrement as pre_dec

from aequilibrae.paths.cython.network_loading cimport cpp_reduce_loading_outputs
from aequilibrae.paths.cython.parallel_numpy cimport assign_link_loads
from aequilibrae.paths.cython.select_link_loading cimport cpp_reduce_select_link_loading
from aequilibrae.paths.cython.workspaces cimport SelectLinkODWorkspace

import numpy as np

from aequilibrae.matrix import AequilibraeSparseMatrix


cdef class LinkLoadingResults:
    def __init__(
        self,
        field_names,
        select_links,
        int num_links,
        bint sl_link_loading,
        int threads,
        *,
        index,
        index_name,
    ):
        if threads <= 0:
            raise ValueError(f"threads must be positive ({threads})")
        if num_links <= 0:
            raise ValueError(f"num_links must be positive ({num_links})")

        self.field_names = tuple(field_names)
        field_count, zone_count = len(self.field_names), len(index)
        self.index = np.array(index, copy=True)
        self.index_name = index_name
        self.num_links = num_links
        self.sl_link_loading = sl_link_loading
        self.select_link_set_names = tuple(select_links or {})

        # Coerce the select link sets to their cpp structures ahead of time. We'll
        # use these frequently and they don't change: a vector of OR alternatives,
        # each containing an unordered AND set of links.
        cdef vector[unique_ptr[unordered_set[long long]]] *selection
        cdef vector[size_t] *lengths

        for or_set in (select_links or {}).values():
            selection = new vector[unique_ptr[unordered_set[long long]]]()
            self.select_link_sets.emplace_back(selection)

            lengths = new vector[size_t]()
            self.select_link_set_lengths.emplace_back(lengths)

            for and_set in or_set:
                selection.emplace_back(new unordered_set[long long](and_set))
                lengths.push_back(len(and_set))

        self.loading = LoadingOutputs(num_links, field_count)
        self.selected_loading = SelectLinkLoadingOutputs(
            num_links, field_count, self.select_link_set_names if sl_link_loading else (),
        )
        self.selected_od = SelectLinkODOutputs(
            zone_count, zone_count, self.field_names, self.select_link_set_names,
        )

        self.workers = []
        cdef LoadingOutputs loading
        cdef SelectLinkLoadingOutputs selected_loading
        cdef SelectLinkODWorkspace selected_od
        for _ in range(threads):
            loading = LoadingOutputs(num_links, field_count)
            selected_loading = SelectLinkLoadingOutputs(
                num_links,
                field_count,
                self.select_link_set_names if sl_link_loading else (),
            )
            selected_od = SelectLinkODWorkspace(
                zone_count,
                zone_count,
                self.field_names,
                self.select_link_set_names,
            )
            self.workers.append((loading, selected_loading, selected_od))
            self.loading_views.push_back(loading.view())
            self.selected_views.push_back(selected_loading.view())
            self.od_views.push_back(selected_od.view())

    cdef object project_loads(self, object loads, const long long[::1] compressed_id_view, int cores):
        # The extra zero row handles links absent from the compressed graph.
        cdef double[:, ::1] compact = np.vstack((loads, np.zeros((1, len(self.field_names)))))
        cdef double[:, ::1] actual = np.zeros((compressed_id_view.shape[0], len(self.field_names)))

        assign_link_loads(actual, compact, compressed_id_view, cores)
        return {name: np.asarray(actual)[:, field].copy() for field, name in enumerate(self.field_names)}

    cdef object link_loading_to_objects(self, const long long[::1] compressed_id_view, int cores):
        if self.link_loading_objects is None:
            self.link_loading_objects = self.project_loads(self.loading.link_loads, compressed_id_view, cores)
        return self.link_loading_objects

    cdef object sl_link_loading_to_objects(self, const long long[::1] compressed_id_view, int cores):
        if not self.sl_link_loading:
            return {}

        if self.sl_link_loading_objects is None:
            self.sl_link_loading_objects = {
                name: self.project_loads(self.selected_loading.link_loads[index], compressed_id_view, cores)
                for index, name in enumerate(self.select_link_set_names)
            }
        return self.sl_link_loading_objects

    cdef void link_load_single_route_set(
        self,
        const double *values,
        size_t field_count,
        const RouteVec_t &route_set,
        const vector[double] &prob_vec,
        size_t thread_id,
    ) noexcept nogil:
        cdef size_t field, route, link
        cdef double load
        cdef double *loads = self.loading_views[thread_id].link_loads
        # For each demand column and route, apply its probability-weighted demand
        # to every link in the route.
        for field in range(field_count):
            for route in range(route_set.size()):
                load = prob_vec[route] * values[field]
                if load == 0:
                    continue
                for link in d(route_set[route]):
                    loads[link * field_count + field] += load

    cdef void reduce_link_loading(self):
        """Requires the GIL: do not call while workers are writing their accumulators."""
        cpp_reduce_loading_outputs[double](self.loading_views.data(), self.loading_views.size(), self.loading.view())

    @staticmethod
    cdef bool is_in_select_link_set(vector[long long] &route,
        const vector[unique_ptr[unordered_set[long long]]] &select_link_set,
        const vector[size_t] &select_link_set_lengths) noexcept nogil:
        """Confirm whether a route satisfies a select link set.

        Assumes the route contains unique links. Repeated links would be counted
        more than once within an AND set.
        """
        cdef vector[size_t] link_counts = select_link_set_lengths
        cdef long long link
        cdef size_t i
        # Iterate over an AND set first so a satisfied alternative short-circuits.
        for i in range(select_link_set.size()):
            for link in route:
                if d(select_link_set[i]).find(link) != d(select_link_set[i]).end():
                    if pre_dec(link_counts[i]) == 0:
                        return True
        return False

    cdef void sl_link_load_single_route_set(
        self,
        size_t origin,
        size_t destination,
        const double *values,
        size_t field_count,
        const RouteVec_t &route_set,
        const vector[double] &prob_vec,
        size_t thread_id,
    ) noexcept nogil:
        cdef size_t selection, route, field, link
        cdef double load
        cdef double *loads

        # For each select link set, load routes satisfying one of its alternatives.
        for selection in range(self.select_link_sets.size()):
            for route in range(route_set.size()):
                if prob_vec[route] == 0 or not LinkLoadingResults.is_in_select_link_set(
                    d(route_set[route]),
                    d(self.select_link_sets[selection]),
                    d(self.select_link_set_lengths[selection]),
                ):
                    continue

                self.od_views[thread_id].builders[selection].append_scaled(
                    origin, destination, values, prob_vec[route],
                )

                if self.sl_link_loading:
                    loads = self.selected_views[thread_id].selection(selection).link_loads

                    for field in range(field_count):
                        load = prob_vec[route] * values[field]
                        if load == 0:
                            continue
                        for link in d(route_set[route]):
                            loads[link * field_count + field] += load

    cdef void reduce_sl_link_loading(self):
        """Consolidate worker-local link loads and COO records after loading."""
        self.selected_od.reset()
        cpp_reduce_select_link_loading[double](
            self.selected_views.data(),
            self.selected_views.size(),
            self.selected_loading.view(),
            self.od_views.data(),
            self.od_views.size(),
            self.selected_od.view(),
        )

    cdef void reduce_sl_od_matrix(self):
        """Copy consolidated records to CSR arrays and discard intermediate results."""
        self.selected_od.update_matrices()
        self.loading_views.clear()
        self.selected_views.clear()
        self.od_views.clear()
        self.workers.clear()
        self.sl_od_matrices_structs_to_objects()
        self.selected_od = None

    cdef object sl_od_matrices_structs_to_objects(self):
        """Return labelled CSR snapshots, keeping their field and set order."""
        if self.od_matrix_objects is None:
            self.od_matrix_objects = {
                name: AequilibraeSparseMatrix._from_shared_arrays(values, self.index, self.index_name)
                for name, values in self.selected_od.matrices.items()
            }
        return self.od_matrix_objects
