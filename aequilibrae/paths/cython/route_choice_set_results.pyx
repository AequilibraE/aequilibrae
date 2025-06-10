from libcpp.algorithm cimport min_element, sort, lower_bound
from libc.math cimport INFINITY, exp, pow, log
from libcpp cimport nullptr
from libcpp.memory cimport shared_ptr, make_shared

from cython.operator cimport dereference as d
from cython.operator cimport postincrement as inc

import pandas as pd
import numpy as np

import cython

from aequilibrae.log import logger


@cython.embedsignature(True)
cdef class RouteChoiceSetResults:
    """
    This class is supposed to help manage and compute the results of the route choice set generation. It also
    provides method to perform an assignment and link loading.
    """

    def __init__(
            self,
            demand: GeneralisedCOODemand,
            cutoff_prob: float,
            beta: float,
            num_links: int,
            double[:] cost_view,
            unsigned int [:] mapping_idx,
            int64_t [::] mapping_data,
            store_results: bool = True,
            perform_assignment: bool = True,
    ):
        """
        :Arguments:
            **demand** (`obj`: GeneralisedCOODemand): A GeneralisedCOODemand object stores the ODs pairs and various
              demand values in a COO form. No verification of these is performed here.

            **cutoff_prob** (`obj`: float): The cut-off probability for the inverse binary logit filter.

            **beta** (`obj`: float): The beta parameter for the path-sized logit.

            **store_results** (`obj`: bool): Whether or not to store the route set computation results. At a minimum
              stores the route sets per OD. If `perform_assignment` is True then the assignment results are stored as
              well.

            **perform_assignment** (`obj`: bool): Whether or not to perform a path-sized logit assignment.

        NOTE: This class makes no attempt to be thread safe when improperly accessed. Multithreaded accesses should be
        coordinated to not collide. Each index of `ods` should only ever be accessed by a single thread.

        NOTE: Depending on `store_results` the behaviour of accessing a single `ods` index multiple times will
        differ. When True the previous internal buffers will be reused. This will highly likely result incorrect
        results. When False some new internal buffers will used, link loading results will still be incorrect. Thus A
        SINGLE `ods` INDEX SHOULD NOT BE ACCESSED MULTIPLE TIMES.
        """

        if not store_results and not perform_assignment:
            raise ValueError("either `store_results` or `perform_assignment` must be True")

        self.demand = demand
        self.cutoff_prob = cutoff_prob
        self.beta = beta
        self.store_results = store_results
        self.perform_assignment = perform_assignment
        self.cost_view = cost_view
        self.mapping_idx = mapping_idx
        self.mapping_data = mapping_data
        self.table = None

        cdef size_t size = self.demand.ods.size()

        # As the objects are attribute of the extension class they will be allocated before the object is
        # initialised. This ensures that accessing them is always valid and that they are just empty. We resize the ones
        # we will be using here and allocate the objects they store for the same reasons.
        #
        # We can't know how big they will be so we'll need to resize later as well.
        if self.store_results:
            self.__route_vecs.resize(size)
            for i in range(size):
                self.__route_vecs[i] = make_shared[RouteVec_t]()

        if self.perform_assignment and self.store_results:
            self.__cost_set.resize(size)
            self.__mask_set.resize(size)
            self.__path_overlap_set.resize(size)
            self.__prob_set.resize(size)
            for i in range(size):
                self.__cost_set[i] = make_shared[vector[double]]()
                self.__mask_set[i] = make_shared[vector[bint]]()
                self.__path_overlap_set[i] = make_shared[vector[double]]()
                self.__prob_set[i] = make_shared[vector[double]]()

    def write(self, where):
        table = self.make_df_from_results()

        engine = pd.io.parquet.get_engine('auto').__class__
        if (engine.__module__, engine.__name__) == ("pandas.io.parquet", "PyArrowImpl"):
            kwargs = dict(
                # can't provide partitioning_flavor and partition_cols through the Pandas API
                use_threads=True,
                existing_data_behavior="overwrite_or_ignore",
                file_visitor=lambda written_file: logger.info(f"Wrote partition dataset at {written_file.path}")
            )
        elif (engine.__module__, engine.__name__) == ("pandas.io.parquet", "FastParquetImpl"):
            logger.info("FastParquet back-end doesn't support individual partition logging, writing table now...")
            kwargs = dict(
                file_scheme="hive",
                # no threads option
                append=False,
                # no visitor option
            )
            logger.warn(
                "FastParquet back-end doesn't support writing a NumPy arrays as Parquet list types, converting to Python lists. "
                "Watch out for memory consumption..."
            )
            table["route set"] = table["route set"].map(lambda x: x.tolist())
        else:
            raise RuntimeError(
                "encountered unknown Pandas parquet engine, please report this as a bug to the AequilibraE issues page"
            )

        table.to_parquet(
            path=where,
            compression="zstd",
            index=False,
            partition_cols=["origin id"],
            **kwargs,
        )

    @classmethod
    def read_dataset(cls, where):
        df = pd.read_parquet(where, partitioning="hive")
        df["origin id"] = df["origin id"].astype(df["destination id"].dtype)

        # FastParquet is stupid and encodes Parquet list objects as json strings!!!
        is_json_encoded = df["route set"].map(lambda x: isinstance(x, (str, bytes)))
        if is_json_encoded.any():
            logger.warn("Found JSON encoded route sets. Parsing into a NumPy array...")
            if not is_json_encoded.all():
                raise TypeError(
                    f"route sets must either be encoded properly as list[int64], or json lists (by FastParquet). The two cannot be mixed"
                )

            import json
            df["route set"] = df["route set"].map(lambda x: np.array(json.loads(x), dtype="int64"))

        return df

    @staticmethod
    cdef void route_set_to_route_vec(RouteVec_t &route_vec, RouteSet_t &route_set) noexcept nogil:
        """
        Transform a set of raw pointers to routes (vectors) into a vector of unique points to
        routes.
        """
        cdef vector[long long] *route

        route_vec.reserve(route_set.size())
        for route in route_set:
            route_vec.emplace_back(route)

        # We now drop all references to those raw pointers. The unique pointers now own those vectors.
        route_set.clear()

    cdef shared_ptr[RouteVec_t] get_route_vec(RouteChoiceSetResults self, size_t i) noexcept nogil:
        """
        Return either a new empty RouteSet_t, or the RouteSet_t (initially empty) corresponding to a OD pair index.

        If `self.store_results` is False no attempt is made to store the route set. The caller is responsible for
        maintaining a reference to it.

        Requires that 0 <= i < self.ods.size().
        """
        if self.store_results:
            # All elements of self.__route_vecs have been initialised in self.__init__.
            return self.__route_vecs[i]
        else:
            # We make a new empty RouteSet_t here, we don't attempt to store it.
            return make_shared[RouteVec_t]()

    cdef shared_ptr[vector[double]] __get_cost_set(RouteChoiceSetResults self, size_t i) noexcept nogil:
        return self.__cost_set[i] if self.store_results else make_shared[vector[double]]()

    cdef shared_ptr[vector[bint]] __get_mask_set(RouteChoiceSetResults self, size_t i) noexcept nogil:
        return self.__mask_set[i] if self.store_results else make_shared[vector[bint]]()

    cdef shared_ptr[vector[double]] __get_path_overlap_set(RouteChoiceSetResults self, size_t i) noexcept nogil:
        return self.__path_overlap_set[i] if self.store_results else make_shared[vector[double]]()

    cdef shared_ptr[vector[double]] get_prob_vec(RouteChoiceSetResults self, size_t i) noexcept nogil:
        return self.__prob_set[i] if self.store_results else make_shared[vector[double]]()

    cdef shared_ptr[vector[double]] compute_result(
        RouteChoiceSetResults self,
        size_t i,
        RouteVec_t &route_set,
        bint *found_zero_cost,
        size_t thread_id
    ) noexcept nogil:
        """
        Compute the desired results for the OD pair index with the provided route set. The route set is required as
        an argument here to facilitate not storing them. The route set should correspond to the provided OD pair index,
        however that is not enforced.

        Requires that 0 <= i < self.ods.size().

        Returns a shared pointer to the probability vector.
        """
        cdef:
            shared_ptr[vector[double]] cost_vec
            shared_ptr[vector[bint]] route_mask
            vector[long long] keys, counts
            shared_ptr[vector[double]] path_overlap_vec
            shared_ptr[vector[double]] prob_vec

        if not self.perform_assignment:
            # If we're not performing an assignment then we must be storing the routes and the routes most already be
            # stored when they were acquired, thus we don't need to do anything here.
            return make_shared[vector[double]]()

        cost_vec = self.__get_cost_set(i)
        route_mask = self.__get_mask_set(i)
        path_overlap_vec = self.__get_path_overlap_set(i)
        prob_vec = self.get_prob_vec(i)

        self.compute_cost(d(cost_vec), route_set, self.cost_view, found_zero_cost)
        self.compute_mask(d(route_mask), d(cost_vec))
        self.compute_frequency(keys, counts, route_set, d(route_mask))
        self.compute_path_overlap(
            d(path_overlap_vec),
            route_set,
            keys,
            counts,
            d(cost_vec),
            d(route_mask),
            self.cost_view
        )
        self.compute_prob(d(prob_vec), d(cost_vec), d(path_overlap_vec), d(route_mask))

        return prob_vec

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef void compute_cost(
        RouteChoiceSetResults self,
        vector[double] &cost_vec,
        const RouteVec_t &route_set,
        const double[:]
        cost_view,
        bint *found_zero_cost
    ) noexcept nogil:
        """Compute the cost each route."""
        cdef:
            # Scratch objects
            double cost
            long long link
            size_t i

        cost_vec.resize(route_set.size())

        found_zero_cost[0] = False
        for i in range(route_set.size()):
            cost = 0.0
            for link in d(route_set[i]):
                cost = cost + cost_view[link]

            cost_vec[i] = cost
            if cost == 0.0:
                found_zero_cost[0] = True

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef void compute_mask(
        RouteChoiceSetResults self,
        vector[bint] &route_mask,
        const vector[double] &total_cost
    ) noexcept nogil:
        """
        Computes a binary logit between the minimum cost path and each path, if the total cost is greater than the
        minimum + the difference in utilities required to produce the cut-off probability then the route is excluded
        from the route set.
        """
        cdef:
            bint found_zero_cost = False
            size_t i

            vector[double].const_iterator min = min_element(total_cost.cbegin(), total_cost.cend())
            double cutoff_cost

        if min == total_cost.cend():
            cutoff_cost = INFINITY
        else:
            cutoff_cost = d(min) + inverse_binary_logit(self.cutoff_prob, 0.0, 1.0)

        route_mask.resize(total_cost.size())

        # The route mask should be True for the routes we wish to include.
        for i in range(total_cost.size()):
            if total_cost[i] == 0.0:
                found_zero_cost = True
                break
            elif total_cost[i] <= cutoff_cost:
                route_mask[i] = True

        if found_zero_cost:
            # If we've found a zero cost path we must abandon the whole route set.
            for i in range(total_cost.size()):
                route_mask[i] = False
        elif min != total_cost.cend():
            # Always include the min element. It should already be but I don't trust floating math to do this correctly.
            # But only if there actually was a min element (i.e. empty route set)
            route_mask[min - total_cost.cbegin()] = True

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef void compute_frequency(
        RouteChoiceSetResults self,
        vector[long long] &keys,
        vector[long long] &counts,
        const RouteVec_t &route_set,
        const vector[bint] &route_mask
    ) noexcept nogil:
        """
        Compute a frequency map for each route with the route_mask applied.

        Each node at index i in the first returned vector has frequency at index i in the second vector.
        """
        cdef:
            vector[long long] link_union
            vector[long long].const_iterator union_iter

            # Scratch objects
            size_t length, count, i
            long long link

        # When calculating the frequency of routes, we need to exclude those not in the mask.
        length = 0
        for i in range(route_set.size()):
            # We do so here ...
            if not route_mask[i]:
                continue

            length = length + d(route_set[i]).size()
        link_union.reserve(length)

        for i in range(route_set.size()):
            # ... and here.
            if not route_mask[i]:
                continue

            link_union.insert(link_union.end(), d(route_set[i]).begin(), d(route_set[i]).end())

        sort(link_union.begin(), link_union.end())

        union_iter = link_union.cbegin()
        while union_iter != link_union.cend():
            count = 0
            link = d(union_iter)
            while union_iter != link_union.cend() and link == d(union_iter):
                count = count + 1
                inc(union_iter)

            keys.push_back(link)
            counts.push_back(count)

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef void compute_path_overlap(
        RouteChoiceSetResults self,
        vector[double] &path_overlap_vec,
        const RouteVec_t &route_set,
        const vector[long long] &keys,
        const vector[long long] &counts,
        const vector[double] &total_cost,
        const vector[bint] &route_mask,
        const double[:] cost_view
    ) noexcept nogil:
        """
        Compute the path overlap figure based on the route cost and frequency.

        Notation changes:
            a: link
            t_a: cost_view
            c_i: total_costs
            A_i: route
            sum_{k in R}: delta_{a,k}: freq_set
        """
        cdef:
            # Scratch objects
            vector[long long].const_iterator link_iter
            double path_overlap
            long long link
            size_t i

        path_overlap_vec.resize(route_set.size())

        for i in range(route_set.size()):
            # Skip masked routes
            if not route_mask[i]:
                continue

            path_overlap = 0.0
            for link in d(route_set[i]):
                # We know the frequency table is ordered and contains every link in the union of the routes.
                # We want to find the index of the link, and use that to look up it's frequency
                link_iter = lower_bound(keys.cbegin(), keys.cend(), link)

                # lower_bound returns keys.end() when no link is found.
                # This /should/ never happen.
                if link_iter == keys.cend():
                    continue
                path_overlap = path_overlap + cost_view[link] / counts[link_iter - keys.cbegin()]

            path_overlap_vec[i] = path_overlap / total_cost[i]

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef void compute_prob(
        RouteChoiceSetResults self,
        vector[double] &prob_vec,
        const vector[double] &total_cost,
        const vector[double] &path_overlap_vec,
        const vector[bint] &route_mask
    ) noexcept nogil:
        """Compute a probability for each route in the route set based on the path overlap."""
        cdef:
            # Scratch objects
            double inv_prob
            size_t i, j

        prob_vec.resize(total_cost.size())

        # Beware when refactoring the below, the scale of the costs may cause floating point errors. Large costs will
        # lead to NaN results
        for i in range(total_cost.size()):
            # The probability of choosing a route that has been masked out is 0.
            if not route_mask[i]:
                continue

            inv_prob = 0.0
            for j in range(total_cost.size()):
                # We must skip any other routes that are not included in the mask otherwise our probabilities won't
                # add up.
                if not route_mask[j]:
                    continue

                inv_prob = inv_prob + pow(path_overlap_vec[j] / path_overlap_vec[i], self.beta) \
                    * exp((total_cost[i] - total_cost[j]))  # Assuming theta=1.0

            prob_vec[i] = 1.0 / inv_prob

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef object make_df_from_results(RouteChoiceSetResults self):
        """
        Construct an pd.DataFrame from the C++ stdlib structures.

        Compressed link IDs are expanded to full network link IDs.
        """

        if self.table is not None:
            return self.table
        elif not self.store_results:
            raise RuntimeError("route set table construction requires `store_results` is True")

        cdef:
            size_t link, tmp, n_routes
            bint have_assignment_results = self.perform_assignment and self.store_results

        columns = {
            "origin id": [],
            "destination id": [],
        }
        route_set_col = []  # We treat this one differently when constructing it

        types = {"cost":"float64", "mask":"bool", "path overlap":"float64", "probability":"float64",
                 "origin id":"uint32", "destination id":"uint32"}

        if have_assignment_results:
            columns["cost"] = []
            columns["mask"] = []
            columns["path overlap"] = []
            columns["probability"] = []

        if have_assignment_results:
            for i in range(self.demand.ods.size()):
                n_routes = d(self.__route_vecs[i]).size()
                if not d(self.__route_vecs[i]).size():  # If there's no routes to add just skip these.
                    continue

                # When assigning from df, the cost, mask, and path overlap vectors may be empty
                tmp = d(self.__cost_set[i]).size()
                columns["cost"].append(
                    np.asarray(<double[:tmp]>d(self.__cost_set[i]).data()) if tmp else np.zeros(n_routes, dtype="float64")
                )

                tmp = d(self.__mask_set[i]).size()
                columns["mask"].append(
                    np.asarray(<bint[:tmp]>d(self.__mask_set[i]).data()) if tmp else np.ones(n_routes, dtype="bool")
                )

                tmp = d(self.__path_overlap_set[i]).size()
                columns["path overlap"].append(
                    np.asarray(<double[:tmp]>d(self.__path_overlap_set[i]).data()) if tmp else np.zeros(n_routes, dtype="float64")
                )

                tmp = d(self.__prob_set[i]).size()
                columns["probability"].append(
                    np.asarray(<double[:tmp]>d(self.__prob_set[i]).data())
                )

        for i in range(self.demand.ods.size()):
            route_set = self.__route_vecs[i]

            columns["origin id"].append(np.full(d(route_set).size(), self.demand.ods[i].first, "uint32"))
            columns["destination id"].append(np.full(d(route_set).size(), self.demand.ods[i].second, "uint32"))

            # Instead of constructing a "list of lists" style object for storing the route sets we instead will
            # construct one big array of link IDs with a corresponding offsets array that indicates where each new row
            # (path) starts.
            for j in range(d(route_set).size()):

                links = []
                for link in d(d(route_set)[j]):
                    # Translate the compressed link IDs in route to network link IDs, this is a 1:n mapping
                    links.append(np.asarray(self.mapping_data[self.mapping_idx[link]:self.mapping_idx[link + 1]]))

                route_set_col.append(np.hstack(links))

        columns = {k: np.hstack(v, casting="no") if len(v) else np.array([], dtype=types[k]) for k, v in columns.items()}
        columns["route set"] = route_set_col

        self.table = pd.DataFrame(columns)
        return self.table


@cython.wraparound(False)
@cython.embedsignature(True)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef double inverse_binary_logit(double prob, double beta0, double beta1) noexcept nogil:
    if prob == 1.0:
        return INFINITY
    elif prob == 0.0:
        return -INFINITY
    else:
        return (log(prob / (1.0 - prob)) - beta0) / beta1
