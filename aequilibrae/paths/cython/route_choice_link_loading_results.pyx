from aequilibrae.matrix.sparse_matrix cimport COO

from cython.operator cimport predecrement as pre_dec
from cython.operator cimport dereference as d
cimport cython

import numpy as np
import itertools

from collections.abc import Hashable

include 'parallel_numpy.pyx'

# See note in route_choice_set.pxd
cdef class LinkLoadingResults:
    def __cinit__(
            self,
            demand: GeneralisedCOODemand,
            select_links: Dict[Hashable, FrozenSet[FrozenSet[int]]],
            num_links: int,
            sl_link_loading: bool,
            threads: int
    ):
        if threads <= 0:
            raise ValueError(f"threads must be positive ({threads})")
        elif num_links <= 0:
            raise ValueError(f"num_links must be positive ({num_links})")

        self.demand = demand
        self.num_links = num_links
        self.od_matrix_objects = None
        self.sl_link_loading = sl_link_loading

        cdef:
            vector[unique_ptr[vector[double]]] *f64_demand_cols
            vector[unique_ptr[vector[float]]] *f32_demand_cols

        # Link loading
        # Allocate the threaded f64 link loading.
        self.f64_link_loading_threaded.reserve(threads)
        for i in range(threads):
            f64_demand_cols = new vector[unique_ptr[vector[double]]]()
            f64_demand_cols.reserve(len(self.demand.f64_names))

            for j in range(len(self.demand.f64_names)):
                f64_demand_cols.emplace_back(new vector[double](self.num_links))

            self.f64_link_loading_threaded.emplace_back(f64_demand_cols)

        # Allocate the threaded f32 link loading.
        self.f32_link_loading_threaded.reserve(threads)
        for i in range(threads):
            f32_demand_cols = new vector[unique_ptr[vector[float]]]()
            f32_demand_cols.reserve(len(self.demand.f32_names))

            for j in range(len(self.demand.f32_names)):
                f32_demand_cols.emplace_back(new vector[float](self.num_links))

            self.f32_link_loading_threaded.emplace_back(f32_demand_cols)

        # self.f64_link_loading and self.f32_link_loading are not allocated here. The objects are initialised to empty
        # vectors but elements are created in self.reduce_link_loading

        # Select link loading
        cdef:
            vector[unique_ptr[unordered_set[long long]]] *select_link_set
            vector[size_t] *select_link_set_length

        # Select link loading sets
        # Coerce the select link sets to their cpp structures ahead of time. We'll be using these a lot and they don't
        # change. We allocate a vector of select link sets. These select link sets a vector representing an OR set,
        # containing a unordered_set of links representing the AND set.

        self.select_link_set_names = select_links.keys()

        self.select_link_sets.reserve(len(select_links))
        self.select_link_set_lengths.reserve(len(select_links))
        for or_set in select_links.values():
            select_link_set = new vector[unique_ptr[unordered_set[long long]]]()
            select_link_set_length = new vector[size_t]()

            select_link_set.reserve(len(or_set))
            select_link_set_length.reserve(len(or_set))

            for and_set in or_set:
                select_link_set.emplace_back(new unordered_set[long long](and_set))
                select_link_set_length.push_back(len(and_set))

            self.select_link_sets.emplace_back(select_link_set)
            self.select_link_set_lengths.emplace_back(select_link_set_length)

        # Select link loading link loads
        cdef:
            vector[unique_ptr[vector[unique_ptr[vector[double]]]]] *f64_sl_select_link_sets
            vector[unique_ptr[vector[unique_ptr[vector[float]]]]] *f32_sl_select_link_sets
            vector[unique_ptr[vector[double]]] *f64_sl_demand_cols
            vector[unique_ptr[vector[float]]] *f32_sl_demand_cols

        if self.sl_link_loading:
            # Allocate f64 thread storage for select link
            self.f64_sl_link_loading_threaded.reserve(threads)
            for i in range(threads):
                f64_sl_select_link_sets = new vector[unique_ptr[vector[unique_ptr[vector[double]]]]]()
                f64_sl_select_link_sets.reserve(self.select_link_sets.size())

                for j in range(self.select_link_sets.size()):
                    f64_sl_demand_cols = new vector[unique_ptr[vector[double]]]()
                    f64_sl_demand_cols.reserve(len(self.demand.f64_names))

                    for k in range(len(self.demand.f64_names)):
                        f64_sl_demand_cols.emplace_back(new vector[double](self.num_links))

                    f64_sl_select_link_sets.emplace_back(f64_sl_demand_cols)

                self.f64_sl_link_loading_threaded.emplace_back(f64_sl_select_link_sets)

            # Allocate f32 thread storage for select link
            self.f32_sl_link_loading_threaded.reserve(threads)
            for i in range(threads):
                f32_sl_select_link_sets = new vector[unique_ptr[vector[unique_ptr[vector[float]]]]]()
                f32_sl_select_link_sets.reserve(self.select_link_sets.size())

                for j in range(self.select_link_sets.size()):
                    f32_sl_demand_cols = new vector[unique_ptr[vector[float]]]()
                    f32_sl_demand_cols.reserve(len(self.demand.f32_names))

                    for k in range(len(self.demand.f32_names)):
                        f32_sl_demand_cols.emplace_back(new vector[float](self.num_links))

                    f32_sl_select_link_sets.emplace_back(f32_sl_demand_cols)

                self.f32_sl_link_loading_threaded.emplace_back(f32_sl_select_link_sets)

        # self.f64_sl_link_loading and self.f32_sl_link_loading are not allocated here. The objects are initialised to
        # empty vectors but elements are created in self.reduce_sl_link_loading

        # Select link loading od matrix
        cdef:
            vector[unique_ptr[vector[COO_f64_struct]]] *f64_sl_od_matrix_sets
            vector[unique_ptr[vector[COO_f32_struct]]] *f32_sl_od_matrix_sets
            vector[COO_f64_struct] *f64_sl_od_matrix_demand_cols
            vector[COO_f32_struct] *f32_sl_od_matrix_demand_cols

        # Allocate f64 thread storage for select link
        self.f64_sl_od_matrix_threaded.reserve(threads)
        for i in range(threads):
            f64_sl_od_matrix_sets = new vector[unique_ptr[vector[COO_f64_struct]]]()
            f64_sl_od_matrix_sets.reserve(self.select_link_sets.size())

            for j in range(self.select_link_sets.size()):
                f64_sl_od_matrix_demand_cols = new vector[COO_f64_struct](len(self.demand.f64_names))

                for k in range(len(self.demand.f64_names)):
                    COO.init_f64_struct(d(f64_sl_od_matrix_demand_cols)[k])

                f64_sl_od_matrix_sets.emplace_back(f64_sl_od_matrix_demand_cols)

            self.f64_sl_od_matrix_threaded.emplace_back(f64_sl_od_matrix_sets)

        # Allocate f32 thread storage for select link
        self.f32_sl_od_matrix_threaded.reserve(threads)
        for i in range(threads):
            f32_sl_od_matrix_sets = new vector[unique_ptr[vector[COO_f32_struct]]]()
            f32_sl_od_matrix_sets.reserve(self.select_link_sets.size())

            for j in range(self.select_link_sets.size()):
                f32_sl_od_matrix_demand_cols = new vector[COO_f32_struct](len(self.demand.f32_names))

                for k in range(len(self.demand.f32_names)):
                    COO.init_f32_struct(d(f32_sl_od_matrix_demand_cols)[k])

                f32_sl_od_matrix_sets.emplace_back(f32_sl_od_matrix_demand_cols)

            self.f32_sl_od_matrix_threaded.emplace_back(f32_sl_od_matrix_sets)

        # self.f64_sl_od_matrix and self.f32_sl_od_matrix are not allocated here. The objects are initialised to
        # empty vectors but elements are created in self.reduce_sl_link_loading

    cdef object link_loading_to_objects(self, long long[:] compressed_id_view, int cores):
        if self.link_loading_objects is None:
            self.link_loading_objects = dict(zip(*self.apply_generic_link_loading(
                self.f64_link_loading, self.f32_link_loading, compressed_id_view, cores
            )))
        return self.link_loading_objects

    cdef object sl_link_loading_to_objects(self, long long[:] compressed_id_view, int cores):
        if not self.sl_link_loading:
            return {}

        if self.sl_link_loading_objects is None:
            results = []
            for i in range(self.select_link_sets.size()):
                results.append(dict(zip(*self.apply_generic_link_loading(
                    d(self.f64_sl_link_loading[i]), d(self.f32_sl_link_loading[i]), compressed_id_view, cores
                ))))
            self.sl_link_loading_objects = dict(zip(self.select_link_set_names, results))
        return self.sl_link_loading_objects

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef void link_load_single_route_set(
        LinkLoadingResults self,
        const size_t od_idx,
        const RouteVec_t &route_set,
        const vector[double] &prob_vec,
        const size_t thread_id
    ) noexcept nogil:
        cdef:
            # Cython doesn't allow declaring references to objects outside of function signatures. So we get the raw
            # pointer instead. It is still owned by the unique_ptr.
            vector[unique_ptr[vector[double]]] *f64_ll_cols = self.f64_link_loading_threaded[thread_id].get()
            vector[unique_ptr[vector[float]]] *f32_ll_cols = self.f32_link_loading_threaded[thread_id].get()
            vector[double] *f64_ll
            vector[float] *f32_ll

            double f64_demand, f64_load
            float f32_demand, f32_load
            size_t i, j

        # For each demand column
        for i in range(self.demand.f64.size()):
            # we grab a pointer to the relevant link loading vector and demand value.
            f64_demand = d(self.demand.f64[i])[od_idx]
            f64_ll = d(f64_ll_cols)[i].get()

            # For each route in the route set
            for j in range(route_set.size()):
                # we compute out load,
                f64_load = prob_vec[j] * f64_demand
                if f64_load == 0.0:
                    continue

                # then apply that to every link in the route
                for link in d(route_set[j]):
                    d(f64_ll)[link] = d(f64_ll)[link] + f64_load

        # Then we do it all over again for the floats.
        for i in range(self.demand.f32.size()):
            f32_demand = d(self.demand.f32[i])[od_idx]
            f32_ll = d(f32_ll_cols)[i].get()

            for j in range(route_set.size()):
                f32_load = prob_vec[j] * f32_demand
                if f32_load == 0.0:
                    continue

                for link in d(route_set[j]):
                    d(f32_ll)[link] = d(f32_ll)[link] + f32_load

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef void reduce_link_loading(LinkLoadingResults self):
        """
        NOTE: doesn't require the GIL but should NOT be called in a multithreaded environment. Thus the function
        requires the GIL.
        """
        cdef:
            vector[unique_ptr[vector[double]]] *f64_ll_cols
            vector[unique_ptr[vector[float]]] *f32_ll_cols
            vector[double] *f64_ll_result
            vector[float] *f32_ll_result
            vector[double] *f64_ll
            vector[float] *f32_ll

            size_t thread_id, i, j

        # Allocate the result link loads
        self.f64_link_loading.reserve(self.demand.f64.size())
        for i in range(self.demand.f64.size()):
            self.f64_link_loading.emplace_back(new vector[double](self.num_links))

        self.f32_link_loading.reserve(self.demand.f32.size())
        for i in range(self.demand.f32.size()):
            self.f32_link_loading.emplace_back(new vector[float](self.num_links))

        # Here we sum all threads link loads into the results.
        # For each thread
        for thread_id in range(self.f64_link_loading_threaded.size()):
            # we grab the columns
            f64_ll_cols = self.f64_link_loading_threaded[thread_id].get()

            # for each column
            for i in range(d(f64_ll_cols).size()):
                # we get the link loading vector
                f64_ll_result = self.f64_link_loading[i].get()
                f64_ll = d(f64_ll_cols)[i].get()

                # for each link
                for j in range(d(f64_ll).size()):
                    d(f64_ll_result)[j] = d(f64_ll_result)[j] + d(f64_ll)[j]

        # Then we do it all over again for the floats.
        for thread_id in range(self.f32_link_loading_threaded.size()):
            f32_ll_cols = self.f32_link_loading_threaded[thread_id].get()

            for i in range(d(f32_ll_cols).size()):
                f32_ll_result = self.f32_link_loading[i].get()
                f32_ll = d(f32_ll_cols)[i].get()

                for j in range(d(f32_ll).size()):
                    d(f32_ll_result)[j] = d(f32_ll_result)[j] + d(f32_ll)[j]

        # Here we discard all the intermediate results
        self.f64_link_loading_threaded.clear()
        self.f32_link_loading_threaded.clear()
        self.f64_link_loading_threaded.shrink_to_fit()
        self.f32_link_loading_threaded.shrink_to_fit()

    cdef object apply_generic_link_loading(
        LinkLoadingResults self,
        vector[unique_ptr[vector[double]]] &f64_link_loading,
        vector[unique_ptr[vector[float]]] &f32_link_loading,
        long long[:] compressed_id_view,
        int cores
    ):
        cdef:
            vector[double] *f64_ll
            vector[float] *f32_ll
            double[:, :] f64_ll_view
            double[:, :] f64_actual
            float[:, :] f32_ll_view
            float[:, :] f32_actual

        f64_ll_vectors = []
        for i in range(f64_link_loading.size()):
            # Push a single element to the back to prevent an issue caused by links that are not present in the
            # compressed graph causing a OOB access
            f64_ll = f64_link_loading[i].get()
            f64_ll.push_back(0.0)

            # Cast the vector to a memory view
            f64_ll_view = <double [:f64_ll.size(), :1]>f64_ll.data()
            f64_actual = np.zeros((compressed_id_view.shape[0], 1), dtype=np.float64)

            # Assign the compressed link loads to the uncompressed graph
            assign_link_loads_cython(f64_actual, f64_ll_view, compressed_id_view, cores)

            # Delete the memory view object and pop that element off the end.
            del f64_ll_view
            f64_ll.pop_back()

            # Just return the actual link loads. If compressed are required then the memory view will have to be copied
            # before it is deleted. It does not own the vectors data.
            f64_ll_vectors.append(np.asarray(f64_actual).reshape(-1))

        # We do the same for the floats, again
        f32_ll_vectors = []
        for i in range(f32_link_loading.size()):
            f32_ll = f32_link_loading[i].get()
            f32_ll.push_back(0.0)

            f32_ll_view = <float [:f32_ll.size(), :1]>f32_ll.data()
            f32_actual = np.zeros((compressed_id_view.shape[0], 1), dtype=np.float32)

            assign_link_loads_cython(f32_actual, f32_ll_view, compressed_id_view, cores)

            del f32_ll_view
            f32_ll.pop_back()

            f32_ll_vectors.append(np.asarray(f32_actual).reshape(-1))

        return itertools.chain(self.demand.f64_names, self.demand.f32_names), \
            itertools.chain(f64_ll_vectors, f32_ll_vectors)

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @staticmethod
    cdef bool is_in_select_link_set(
        vector[long long] &route,  # const but cython doesn't allow iteration over const vectors unless an index is used
        const vector[unique_ptr[unordered_set[long long]]] &select_link_set,
        const vector[size_t] &select_link_set_lengths
    ) noexcept nogil:
        """
        Confirms if a given route satisfies the requirements of the select link set.
        **Assumes the route contains unique links**. If a link is present >1 and is in the select link set, this method
        will double count it.
        """

        cdef:
            vector[size_t] link_counts
            long long link

        # We count the number of links within the AND set. Assuming we only see unique links, then if that count goes to
        # 0 we have seen all links present within the AND set. A shortest-path will contain only unique links in
        # paths. This may not be not be true for heuristics, but if the heuristic is that bad that it's traversing a
        # link twice I don't think we should be using it.
        link_counts.insert(link_counts.begin(), select_link_set_lengths.cbegin(), select_link_set_lengths.cend())

        # We iterate over an AND set first in hopes that a whole set is satisfied before checking another. This let's
        # just "short circuit" in a sense
        for i in range(select_link_set.size()):
            for link in route:
                if d(select_link_set[i]).find(link) != d(select_link_set[i]).end():
                    if pre_dec(link_counts[i]) == 0:
                        return True
                else:
                    continue

        return False

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef void sl_link_load_single_route_set(
        LinkLoadingResults self,
        const size_t od_idx,
        const RouteVec_t &route_set,
        const vector[double] &prob_vec,
        const long long origin_idx,
        const long long dest_idx,
        const size_t thread_id
    ) noexcept nogil:
        cdef:
            # Cython doesn't allow declaring references to objects outside of function signatures. So we get the raw
            # pointer instead. It is still owned by the unique_ptr.
            vector[unique_ptr[vector[unique_ptr[vector[double]]]]] *f64_ll_sets_cols
            vector[unique_ptr[vector[unique_ptr[vector[float]]]]] *f32_ll_sets_cols
            vector[unique_ptr[vector[COO_f64_struct]]] *f64_od_sets_cols
            vector[unique_ptr[vector[COO_f32_struct]]] *f32_od_sets_cols
            vector[double] *f64_ll
            vector[float] *f32_ll

            double f64_load
            float f32_load
            size_t i, j, k

        if self.sl_link_loading:
            f64_ll_sets_cols = self.f64_sl_link_loading_threaded[thread_id].get()
            f32_ll_sets_cols = self.f32_sl_link_loading_threaded[thread_id].get()
        else:
            f64_ll_sets_cols = NULL
            f32_ll_sets_cols = NULL

        f64_od_sets_cols = self.f64_sl_od_matrix_threaded[thread_id].get()
        f32_od_sets_cols = self.f32_sl_od_matrix_threaded[thread_id].get()

        # For each select link set
        for i in range(self.select_link_sets.size()):
            # for each route in the route set
            for j in range(route_set.size()):
                # if this route satisfies this link set, then we link load
                if not LinkLoadingResults.is_in_select_link_set(
                        d(route_set[j]),
                        d(self.select_link_sets[i]),
                        d(self.select_link_set_lengths[i])
                ):
                    continue

                # For each demand column
                for k in range(self.demand.f64.size()):
                    # we grab a pointer to the relevant link loading vector and compute our load from our demand,
                    f64_load = prob_vec[j] * d(self.demand.f64[k])[od_idx]

                    if f64_load == 0.0:
                        continue

                    COO.f64_struct_append(d(d(f64_od_sets_cols)[i])[k], origin_idx, dest_idx, f64_load)

                    if self.sl_link_loading:
                        f64_ll = d(d(f64_ll_sets_cols)[i])[k].get()
                        # then apply that to every link in the route
                        for link in d(route_set[j]):
                            d(f64_ll)[link] = d(f64_ll)[link] + f64_load

                # then we do it again for f32
                for k in range(self.demand.f32.size()):
                    f32_load = prob_vec[j] * d(self.demand.f32[k])[od_idx]

                    if f32_load == 0.0:
                        continue

                    COO.f32_struct_append(d(d(f32_od_sets_cols)[i])[k], origin_idx, dest_idx, f32_load)

                    if self.sl_link_loading:
                        f32_ll = d(d(f32_ll_sets_cols)[i])[k].get()
                        for link in d(route_set[j]):
                            d(f32_ll)[link] = d(f32_ll)[link] + f32_load

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef void reduce_sl_link_loading(LinkLoadingResults self):
        """
        NOTE: doesn't require the GIL but should NOT be called in a multithreaded environment. Thus the function
        requires the GIL.
        """
        cdef:
            vector[unique_ptr[vector[unique_ptr[vector[double]]]]] *f64_sl_ll_sets_cols
            vector[unique_ptr[vector[unique_ptr[vector[float]]]]] *f32_sl_ll_sets_cols
            vector[unique_ptr[vector[double]]] *f64_sl_ll_cols
            vector[unique_ptr[vector[float]]] *f32_sl_ll_cols
            vector[double] *f64_sl_ll_result
            vector[float] *f32_sl_ll_result
            vector[double] *f64_sl_ll
            vector[float] *f32_sl_ll

            size_t thread_id, i, j

        if not self.sl_link_loading:
            return

        # Allocate the result link loads
        self.f64_sl_link_loading.reserve(self.select_link_sets.size())
        for i in range(self.select_link_sets.size()):
            f64_sl_ll_cols = new vector[unique_ptr[vector[double]]]()
            f64_sl_ll_cols.reserve(self.demand.f64.size())

            for j in range(self.demand.f64.size()):
                f64_sl_ll_cols.emplace_back(new vector[double](self.num_links))

            self.f64_sl_link_loading.emplace_back(f64_sl_ll_cols)

        self.f32_sl_link_loading.reserve(self.select_link_sets.size())
        for i in range(self.select_link_sets.size()):
            f32_sl_ll_cols = new vector[unique_ptr[vector[float]]]()
            f32_sl_ll_cols.reserve(self.demand.f32.size())

            for j in range(self.demand.f32.size()):
                f32_sl_ll_cols.emplace_back(new vector[float](self.num_links))

            self.f32_sl_link_loading.emplace_back(f32_sl_ll_cols)

        # Here we sum all threads link loads into the results.
        # For each thread
        for thread_id in range(self.f64_sl_link_loading_threaded.size()):
            # we grab the link set specific one
            f64_sl_ll_sets_cols = self.f64_sl_link_loading_threaded[thread_id].get()

            for i in range(d(f64_sl_ll_sets_cols).size()):
                # we grab the columns
                f64_sl_ll_cols = d(f64_sl_ll_sets_cols)[i].get()

                # for each column
                for j in range(d(f64_sl_ll_cols).size()):
                    # we get the link loading vector
                    f64_sl_ll_result = d(self.f64_sl_link_loading[i])[j].get()
                    f64_sl_ll = d(f64_sl_ll_cols)[j].get()

                    # for each link
                    for k in range(d(f64_sl_ll).size()):
                        d(f64_sl_ll_result)[k] = d(f64_sl_ll_result)[k] + d(f64_sl_ll)[k]

        # Then we do it all over again for the floats.
        for thread_id in range(self.f32_sl_link_loading_threaded.size()):
            f32_sl_ll_sets_cols = self.f32_sl_link_loading_threaded[thread_id].get()

            for i in range(d(f32_sl_ll_sets_cols).size()):
                f32_sl_ll_cols = d(f32_sl_ll_sets_cols)[i].get()

                for j in range(d(f32_sl_ll_cols).size()):
                    f32_sl_ll_result = d(self.f32_sl_link_loading[i])[j].get()
                    f32_sl_ll = d(f32_sl_ll_cols)[j].get()

                    for k in range(d(f32_sl_ll).size()):
                        d(f32_sl_ll_result)[k] = d(f32_sl_ll_result)[k] + d(f32_sl_ll)[k]

        # Here we discard all the intermediate results
        self.f64_sl_link_loading_threaded.clear()
        self.f32_sl_link_loading_threaded.clear()
        self.f64_sl_link_loading_threaded.shrink_to_fit()
        self.f32_sl_link_loading_threaded.shrink_to_fit()

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef void reduce_sl_od_matrix(LinkLoadingResults self):
        """
        NOTE: doesn't require the GIL but should NOT be called in a multithreaded environment. Thus the function
        requires the GIL.
        """
        cdef:
            vector[unique_ptr[vector[COO_f64_struct]]] *f64_sl_od_sets_cols
            vector[unique_ptr[vector[COO_f32_struct]]] *f32_sl_od_sets_cols
            vector[COO_f64_struct] *f64_sl_od_cols
            vector[COO_f32_struct] *f32_sl_od_cols
            COO_f64_struct *f64_sl_od_result
            COO_f32_struct *f32_sl_od_result
            COO_f64_struct *f64_sl_od
            COO_f32_struct *f32_sl_od

            size_t thread_id, i, j

        # Allocate f64 thread storage for select link
        self.f64_sl_od_matrix.reserve(self.select_link_sets.size())
        for i in range(self.select_link_sets.size()):
            f64_sl_od_cols = new vector[COO_f64_struct](self.demand.f64.size())

            for j in range(self.demand.f64.size()):
                COO.init_f64_struct(d(f64_sl_od_cols)[j])

            self.f64_sl_od_matrix.emplace_back(f64_sl_od_cols)

        # Allocate f32 thread storage for select link
        self.f32_sl_od_matrix.reserve(self.select_link_sets.size())
        for i in range(self.select_link_sets.size()):
            f32_sl_od_cols = new vector[COO_f32_struct](self.demand.f32.size())

            for j in range(self.demand.f32.size()):
                COO.init_f32_struct(d(f32_sl_od_cols)[j])

            self.f32_sl_od_matrix.emplace_back(f32_sl_od_cols)

        # Here we sum all threads link loads into the results.
        # For each thread
        for thread_id in range(self.f64_sl_od_matrix_threaded.size()):
            # we grab the link set specific one
            f64_sl_od_sets_cols = self.f64_sl_od_matrix_threaded[thread_id].get()

            for i in range(d(f64_sl_od_sets_cols).size()):
                # we grab the columns
                f64_sl_od_cols = d(f64_sl_od_sets_cols)[i].get()

                # for each column
                for j in range(d(f64_sl_od_cols).size()):
                    f64_sl_od_result = &d(self.f64_sl_od_matrix[i])[j]
                    f64_sl_od = &d(f64_sl_od_cols)[j]

                    d(f64_sl_od_result.row).insert(
                        d(f64_sl_od_result.row).end(),
                        d(f64_sl_od.row).cbegin(),
                        d(f64_sl_od.row).cend()
                    )
                    d(f64_sl_od_result.col).insert(
                        d(f64_sl_od_result.col).end(),
                        d(f64_sl_od.col).cbegin(),
                        d(f64_sl_od.col).cend()
                    )
                    d(f64_sl_od_result.f64_data).insert(
                        d(f64_sl_od_result.f64_data).end(),
                        d(f64_sl_od.f64_data).begin(),
                        d(f64_sl_od.f64_data).end()
                    )

        # Then we do it all over again for the floats.
        for thread_id in range(self.f32_sl_od_matrix_threaded.size()):
            # we grab the link set specific one
            f32_sl_od_sets_cols = self.f32_sl_od_matrix_threaded[thread_id].get()

            for i in range(d(f32_sl_od_sets_cols).size()):
                # we grab the columns
                f32_sl_od_cols = d(f32_sl_od_sets_cols)[i].get()

                # for each column
                for j in range(d(f32_sl_od_cols).size()):
                    f32_sl_od_result = &d(self.f32_sl_od_matrix[i])[j]
                    f32_sl_od = &d(f32_sl_od_cols)[j]

                    d(f32_sl_od_result.row).insert(
                        d(f32_sl_od_result.row).end(),
                        d(f32_sl_od.row).cbegin(),
                        d(f32_sl_od.row).cend()
                    )
                    d(f32_sl_od_result.col).insert(
                        d(f32_sl_od_result.col).end(),
                        d(f32_sl_od.col).cbegin(),
                        d(f32_sl_od.col).cend()
                    )
                    d(f32_sl_od_result.f32_data).insert(
                        d(f32_sl_od_result.f32_data).end(),
                        d(f32_sl_od.f32_data).begin(),
                        d(f32_sl_od.f32_data).end()
                    )

        # Here we discard all the intermediate results
        self.f64_sl_od_matrix_threaded.clear()
        self.f32_sl_od_matrix_threaded.clear()
        self.f64_sl_od_matrix_threaded.shrink_to_fit()
        self.f32_sl_od_matrix_threaded.shrink_to_fit()

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef object sl_od_matrices_structs_to_objects(LinkLoadingResults self):
        """
        Convert the stored structs to python COO objects. This transfers the ownership of the memory meaning we need to
        store the newly created objects as this is a one-time operation.
        """
        if self.od_matrix_objects is not None:
            return self.od_matrix_objects

        cdef size_t i, j

        od_matrix_objects = []
        for i in range(self.select_link_sets.size()):
            res = []
            for j in range(self.demand.f64.size()):
                coo = COO.from_f64_struct(d(self.f64_sl_od_matrix[i])[j])
                coo.shape = self.demand.shape
                res.append((self.demand.f64_names[j], coo))

            for j in range(self.demand.f32.size()):
                coo = COO.from_f32_struct(d(self.f32_sl_od_matrix[i])[j])
                coo.shape = self.demand.shape
                res.append((self.demand.f32_names[j], coo))

            od_matrix_objects.append(dict(res))

        self.f64_sl_od_matrix.clear()
        self.f32_sl_od_matrix.clear()
        self.f64_sl_od_matrix.shrink_to_fit()
        self.f32_sl_od_matrix.shrink_to_fit()

        self.od_matrix_objects = dict(zip(self.select_link_set_names, od_matrix_objects))
        return self.od_matrix_objects
