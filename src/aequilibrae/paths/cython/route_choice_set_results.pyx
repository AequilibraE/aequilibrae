from libcpp.algorithm cimport sort, lower_bound, upper_bound
from libcpp.utility cimport pair
from libc.math cimport INFINITY, exp, pow, log, isfinite
from libcpp.memory cimport shared_ptr, make_shared

from cython.operator cimport dereference as d
from cython.operator cimport postincrement as inc

import importlib.util
import pandas as pd
import numpy as np

import cython
import logging

logger = logging.getLogger(__name__)


@cython.embedsignature(True)
cdef class RouteChoiceSetResults:
    """
    This class is supposed to help manage and compute the results of the route choice set generation. It also
    provides method to perform an assignment and link loading.
    """

    def __init__(
            self,
            demand: GeneralisedCOODemand,
            disutility_cutoff_constant: float,
            disutility_cutoff_coefficient: float,
            beta: float,
            num_links: int,
            const double[:] cost_view,
            const unsigned int [:] mapping_idx,
            const int64_t [::] mapping_data,
            const int64_t [::] link_id_direction,
            store_results: bool = True,
            perform_assignment: bool = True,
            const int64_t[::1] full_link_ids = None,
            const int8_t[::1] full_directions = None,
    ):
        """
        :Arguments:
            **demand** (`obj`: GeneralisedCOODemand): A GeneralisedCOODemand object stores the ODs pairs and various
              demand values in a COO form. No verification of these is performed here.

            **disutility_cutoff_constant** (`obj`: float): The cutoff disutility for path filter is a linear function
            of the disutility of the route with the minimum disutility. This is the constant term.

            **disutility_cutoff_coefficient** (`obj`: float): The cutoff disutility for path filter is a linear function
            of the disutility of the route with the minimum disutility. This is the coefficient of the minimum 
            disutility.

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
        self.disutility_cutoff_constant = disutility_cutoff_constant
        self.disutility_cutoff_coefficient = disutility_cutoff_coefficient
        self.beta = beta
        self.store_results = store_results
        self.perform_assignment = perform_assignment
        self.cost_view = cost_view
        self.mapping_idx = mapping_idx
        self.mapping_data = mapping_data
        self.link_id_direction = link_id_direction
        # Imported routes keep full-link indices, including invalid sequences that cannot be expanded from compact
        # links.
        self.full_link_ids = full_link_ids
        self.full_directions = full_directions
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

    def write(self, where, to_parquet_kwargs):
        table = self.make_df_from_results()

        engine_name = pd.get_option("io.parquet.engine")
        if engine_name == "auto":
            if importlib.util.find_spec("pyarrow") is not None:
                engine_name = "pyarrow"
            elif importlib.util.find_spec("fastparquet") is not None:
                engine_name = "fastparquet"
            else:
                raise RuntimeError(
                    "No supported parquet engine (pyarrow or fastparquet) available. "
                    "Please install one with: pip install pyarrow OR pip install fastparquet"
                )

        kwargs = {
            "path": where,
            "compression": "zstd",
            "index": False,
            "partition_cols": ["origin id"],
        }

        if engine_name == "pyarrow":
            kwargs |= {
                # can't provide partitioning_flavor and partition_cols through the Pandas API
                "use_threads": True,
                "existing_data_behavior": "overwrite_or_ignore",
            }
        elif engine_name == "fastparquet":
            logger.info("FastParquet back-end doesn't support individual partition logging, writing table now...")
            kwargs |= {
                "file_scheme": "hive",
                # no threads option
                "append": False,
                # no visitor option
            }
            logger.warning(
                "FastParquet back-end doesn't support writing a NumPy arrays as Parquet list types, converting to "
                "Python lists. Watch out for memory consumption..."
            )
            # HACK: assign() rather than __setitem__: pandas 3's chained-assignment check can't see locals
            # of a compiled Cython frame, so plain df[col] = ... warns spuriously here
            table = table.assign(**{"route set": table["route set"].map(lambda x: x.tolist())})
        else:
            raise RuntimeError(
                "encountered unknown Pandas parquet engine, please report this as a bug to the AequilibraE issues page"
            )

        table.to_parquet(**(kwargs | to_parquet_kwargs))

    @classmethod
    def read_dataset(cls, where):
        df = pd.read_parquet(where, partitioning="hive")
        # HACK: assign() rather than __setitem__: pandas 3's chained-assignment check can't see locals
        # of a compiled Cython frame, so plain df[col] = ... warns spuriously here
        df = df.assign(**{"origin id": df["origin id"].astype(df["destination id"].dtype)})

        # FastParquet is stupid and encodes Parquet list objects as json strings!!!
        is_json_encoded = df["route set"].map(lambda x: isinstance(x, (str, bytes)))
        if is_json_encoded.any():
            logger.warning("Found JSON encoded route sets. Parsing into a NumPy array...")
            if not is_json_encoded.all():
                raise TypeError(
                    "route sets must either be encoded properly as list[int64], or json lists (by FastParquet). "
                    "The two cannot be mixed"
                )

            import json
            # HACK: assign() rather than __setitem__: pandas 3's chained-assignment check can't see locals
            # of a compiled Cython frame, so plain df[col] = ... warns spuriously here
            df = df.assign(**{"route set": df["route set"].map(lambda x: np.array(json.loads(x), dtype="int64"))})

        return df

    @staticmethod
    cdef void route_set_to_route_vec(
        RouteVec_t &route_vec,
        vector[vector[double]] &route_turns,
        RouteCandidateSet_t &route_set,
        bint save_turns
    ) noexcept nogil:
        """Move links and any needed turn steps into matching output positions."""
        cdef RouteCandidate *candidate
        cdef vector[RouteCandidate *] candidates

        route_vec.reserve(route_set.size())
        if save_turns:
            route_turns.reserve(route_set.size())

        for candidate in route_set:
            candidates.push_back(candidate)

        # Remove candidates from the hash set before moving their link keys.
        route_set.clear()
        for candidate in candidates:
            route_vec.emplace_back(new vector[long long]())
            d(route_vec.back()).swap(candidate.links)

            if save_turns:
                route_turns.emplace_back()
                route_turns.back().swap(candidate.turn_steps)

            del candidate

    cdef shared_ptr[RouteVec_t] get_route_vec(RouteChoiceSetResults self, size_t i) noexcept nogil:
        """
        Return either a new route vector or the stored route vector for this OD pair.

        If `self.store_results` is False no attempt is made to store the route set. The caller is responsible for
        maintaining a reference to it.

        Requires that 0 <= i < self.ods.size().
        """
        if self.store_results:
            # All elements of self.__route_vecs have been initialised in self.__init__.
            return self.__route_vecs[i]
        else:
            # Make a temporary route vector without storing it.
            return make_shared[RouteVec_t]()

    cdef shared_ptr[vector[double]] __get_cost_set(RouteChoiceSetResults self, size_t i) noexcept nogil:
        return self.__cost_set[i] if self.store_results else make_shared[vector[double]]()

    cdef shared_ptr[vector[bint]] __get_mask_set(RouteChoiceSetResults self, size_t i) noexcept nogil:
        return self.__mask_set[i] if self.store_results else make_shared[vector[bint]]()

    cdef shared_ptr[vector[double]] __get_path_overlap_set(RouteChoiceSetResults self, size_t i) noexcept nogil:
        return self.__path_overlap_set[i] if self.store_results else make_shared[vector[double]]()

    cdef shared_ptr[vector[double]] get_prob_vec(RouteChoiceSetResults self, size_t i) noexcept nogil:
        return self.__prob_set[i] if self.store_results else make_shared[vector[double]]()

    cdef void store_imported_result(
        self,
        size_t i,
        RouteVec_t &routes,
        const vector[double] &costs,
        const vector[bint] &mask,
        const vector[double] &overlap,
        const vector[double] &probabilities,
        const vector[size_t] &positions
    ) noexcept nogil:
        """Move imported paths into the result owner and store their matching numeric results."""
        cdef size_t position, j
        if not self.store_results:
            return

        for j in range(positions.size()):
            position = positions[j]
            d(self.__route_vecs[i]).emplace_back(routes[position].release())
            d(self.__cost_set[i]).push_back(costs[position])
            d(self.__mask_set[i]).push_back(mask[position])
            d(self.__path_overlap_set[i]).push_back(overlap[position])
            d(self.__prob_set[i]).push_back(probabilities[position])

    cdef shared_ptr[vector[double]] compute_result(
        RouteChoiceSetResults self,
        size_t i,
        RouteVec_t &route_set,
        const vector[vector[double]] &route_turns,
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
            shared_ptr[vector[double]] path_overlap_vec
            shared_ptr[vector[double]] prob_vec
            RouteView_t paths
            RouteTurnView_t turns
            size_t j

        if not self.perform_assignment:
            # If we're not performing an assignment then we must be storing the routes and the routes most already be
            # stored when they were acquired, thus we don't need to do anything here.
            return make_shared[vector[double]]()

        cost_vec = self.__get_cost_set(i)
        route_mask = self.__get_mask_set(i)
        path_overlap_vec = self.__get_path_overlap_set(i)
        prob_vec = self.get_prob_vec(i)

        self.compute_cost(d(cost_vec), route_set, route_turns, self.cost_view, found_zero_cost)

        for j in range(route_set.size()):
            paths.push_back(&d(route_set[j]))

        for j in range(route_turns.size()):
            turns.push_back(&route_turns[j])

        RouteChoiceSetResults.compute_psl(
            paths, turns,
            d(cost_vec),
            d(route_mask),
            d(path_overlap_vec),
            d(prob_vec),
            self.cost_view,
            self.beta,
            self.disutility_cutoff_constant,
            self.disutility_cutoff_coefficient
        )

        return prob_vec

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
        double disutility_cutoff_constant,
        double disutility_cutoff_coefficient,
    ) noexcept nogil:
        """Compute PSL from costs and links, preserving any supplied exclusions."""
        cdef vector[long long] keys, counts
        cdef vector[pair[long long, long long]] turns
        cdef pair[long long, long long] turn
        cdef size_t j, k
        cdef long long previous, link

        RouteChoiceSetResults.compute_mask(route_mask, cost_vec, disutility_cutoff_constant, disutility_cutoff_coefficient)
        RouteChoiceSetResults.compute_frequency(keys, counts, route_set, route_mask)

        if route_turns.size() and d(route_turns[0]).size():
            for j in range(route_set.size()):
                if not route_mask[j]:
                    continue

                previous = -1
                for k in range(d(route_set[j]).size()):
                    link = d(route_set[j])[k]
                    if previous != -1:
                        turn.first = previous
                        turn.second = link
                        turns.push_back(turn)
                    previous = link

            sort(turns.begin(), turns.end())

        RouteChoiceSetResults.compute_path_overlap(
            path_overlap_vec,
            route_set,
            keys,
            counts,
            turns,
            route_turns,
            cost_vec,
            route_mask,
            cost_view
        )
        RouteChoiceSetResults.compute_prob(prob_vec, cost_vec, path_overlap_vec, route_mask, beta)

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef void compute_cost(
        RouteChoiceSetResults self,
        vector[double] &cost_vec,
        const RouteVec_t &route_set,
        const vector[vector[double]] &route_turns,
        const double[:] cost_view,
        bint *found_zero_cost
    ) noexcept nogil:
        """Compute the cost each route."""
        cdef:
            # Scratch objects
            double cost
            size_t i, j

        cdef bint has_turn_steps = route_turns.size() and route_turns[0].size()
        cost_vec.resize(route_set.size())

        found_zero_cost[0] = False
        for i in range(route_set.size()):
            cost = 0.0
            for j in range(d(route_set[i]).size()):
                cost = cost + cost_view[d(route_set[i])[j]]
                if has_turn_steps:
                    cost = cost + route_turns[i][j]

            cost_vec[i] = cost
            if cost == 0.0:
                found_zero_cost[0] = True

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @staticmethod
    cdef void compute_mask(
        vector[bint] &route_mask,
        const vector[double] &total_cost,
        double disutility_cutoff_constant,
        double disutility_cutoff_coefficient,
    ) noexcept nogil:
        """
        Computes a binary logit between the minimum cost path and each path, if the total cost is greater than the
        minimum + the difference in utilities required to produce the cut-off probability then the route is excluded
        from the route set.
        """
        cdef:
            bint found_zero_cost = False
            size_t i

            size_t min_index = total_cost.size()
            double min_cost = INFINITY
            double cutoff_cost

        route_mask.resize(total_cost.size(), True)
        # An excluded route or infinite ban must not become the fallback minimum; also skip NaN costs.
        for i in range(total_cost.size()):
            if route_mask[i] and isfinite(total_cost[i]) and total_cost[i] < min_cost:
                min_cost = total_cost[i]
                min_index = i

        cutoff_cost = disutility_cutoff_constant + disutility_cutoff_coefficient * min_cost

        # The route mask should be True for the routes we wish to include.
        for i in range(total_cost.size()):
            if not route_mask[i]:
                continue
            route_mask[i] = False
            if not isfinite(total_cost[i]):
                continue
            if total_cost[i] == 0.0:
                found_zero_cost = True
            elif total_cost[i] <= cutoff_cost:
                route_mask[i] = True

        if found_zero_cost:
            # If we've found a zero cost path we must abandon the whole route set.
            for i in range(total_cost.size()):
                route_mask[i] = False
        elif min_index != total_cost.size():
            # Always include the min element. It should already be but I don't trust floating math to do this correctly.
            # But only if there actually was a finite minimum.
            route_mask[min_index] = True

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @staticmethod
    cdef void compute_frequency(
        vector[long long] &keys,
        vector[long long] &counts,
        const RouteView_t &route_set,
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
            vector[pair[long long, long long]].const_iterator turn_begin, turn_end
            pair[long long, long long] turn
            double path_overlap
            long long link, previous
            size_t i, j

        cdef bint has_turn_steps = route_turns.size() and d(route_turns[0]).size()
        path_overlap_vec.resize(route_set.size())

        for i in range(route_set.size()):
            # Skip masked routes
            if not route_mask[i]:
                path_overlap_vec[i] = 0.0
                continue

            path_overlap = 0.0
            previous = -1
            for j in range(d(route_set[i]).size()):
                link = d(route_set[i])[j]
                # We know the frequency table is ordered and contains every link in the union of the routes.
                # We want to find the index of the link, and use that to look up it's frequency
                link_iter = lower_bound(keys.cbegin(), keys.cend(), link)

                # lower_bound returns keys.end() when no link is found.
                # This /should/ never happen.
                if link_iter == keys.cend():
                    continue
                path_overlap = path_overlap + cost_view[link] / counts[link_iter - keys.cbegin()]

                if has_turn_steps and previous != -1:
                    turn.first = previous
                    turn.second = link
                    turn_begin = lower_bound(turns.cbegin(), turns.cend(), turn)
                    turn_end = upper_bound(turns.cbegin(), turns.cend(), turn)
                    path_overlap = path_overlap + d(route_turns[i])[j] / (turn_end - turn_begin)

                previous = link

            path_overlap_vec[i] = path_overlap / total_cost[i]

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    @staticmethod
    cdef void compute_prob(
        vector[double] &prob_vec,
        const vector[double] &total_cost,
        const vector[double] &path_overlap_vec,
        const vector[bint] &route_mask,
        double beta
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
                prob_vec[i] = 0.0
                continue

            inv_prob = 0.0
            for j in range(total_cost.size()):
                # We must skip any other routes that are not included in the mask otherwise our probabilities won't
                # add up.
                if not route_mask[j]:
                    continue

                inv_prob = inv_prob + pow(path_overlap_vec[j] / path_overlap_vec[i], beta) \
                    * exp((total_cost[i] - total_cost[j]))  # Assuming theta=1.0

            prob_vec[i] = 1.0 / inv_prob

    @cython.wraparound(False)
    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef object make_df_from_results(RouteChoiceSetResults self):
        """
        Construct an pd.DataFrame from the C++ stdlib structures.

        Generated compact link IDs are expanded to full network link IDs.
        Imported full-link indices are mapped directly to their original signed IDs.
        """

        if self.table is not None:
            return self.table
        elif not self.store_results:
            raise RuntimeError("route set table construction requires `store_results` is True")

        cdef:
            size_t link, tmp, n_routes, idx_min, idx_max, supernet_id
            bint have_assignment_results = self.perform_assignment and self.store_results
            const int64_t [::] supernet_ids

        columns = {
            "origin id": [],
            "destination id": [],
        }
        route_set_col = []  # We treat this one differently when constructing it

        types = {"cost": "float64", "mask": "bool", "path overlap": "float64", "probability": "float64",
                 "origin id": "uint32", "destination id": "uint32"}

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

                # Empty numeric vectors use defaults for results without these fields.
                tmp = d(self.__cost_set[i]).size()
                columns["cost"].append(
                    np.asarray(<double[:tmp]>d(self.__cost_set[i]).data())
                    if tmp
                    else np.zeros(n_routes, dtype="float64")
                )

                tmp = d(self.__mask_set[i]).size()
                columns["mask"].append(
                    np.asarray(<bint[:tmp]>d(self.__mask_set[i]).data()).astype(bool)
                    if tmp
                    else np.ones(n_routes, dtype="bool")
                )

                tmp = d(self.__path_overlap_set[i]).size()
                columns["path overlap"].append(
                    np.asarray(<double[:tmp]>d(self.__path_overlap_set[i]).data())
                    if tmp
                    else np.zeros(n_routes, dtype="float64")
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
            # construct one big array of link IDs (with direction as sign) with a corresponding offsets array that
            # indicates where each new row (path) starts.
            for j in range(d(route_set).size()):
                if self.full_link_ids is not None:
                    route_set_col.append(np.array(
                        [self.full_link_ids[link] * self.full_directions[link] for link in d(d(route_set)[j])],
                        dtype=np.int64
                    ))
                    continue

                links = []
                for link in d(d(route_set)[j]):
                    # Translate compressed link IDs to __supernet_id__ then to link_id * direction
                    idx_min = self.mapping_idx[link]
                    idx_max = self.mapping_idx[link + 1]

                    # If there's just one link we can do a little better but just adding that single link ID
                    if idx_max - idx_min == 1:
                        links.append(self.link_id_direction[self.mapping_data[idx_min]])
                    else:
                        # Otherwise we pull out the full range and translate that
                        supernet_ids = self.mapping_data[idx_min:idx_max]
                        for supernet_id in supernet_ids:
                            links.append(self.link_id_direction[supernet_id])

                route_set_col.append(np.hstack(links))

        columns = {
            k: np.hstack(v, casting="no") if len(v) else np.array([], dtype=types[k]) for k, v in columns.items()
        }
        columns["route set"] = route_set_col

        self.table = pd.DataFrame(columns)
        return self.table


cdef void recompute_route_probabilities(
    object df, const RouteVec_t &routes, const vector[vector[double]] &turn_steps,
    const vector[double] &costs, vector[bint] &route_mask, vector[double] &path_overlap,
    vector[double] &probabilities, const double[:] link_costs, double beta, 
    double disutility_cutoff_constant, double disutility_cutoff_coefficient
):
    """Apply the shared PSL kernel to borrowed native routes and turn steps, without demand."""
    cdef RouteView_t paths
    cdef RouteTurnView_t turns
    cdef vector[double] route_costs, overlap, probability
    cdef vector[bint] mask
    cdef vector[size_t] positions
    cdef size_t position, j

    if not isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and non-negative")
    if disutility_cutoff_constant != float('inf') and disutility_cutoff_coefficient == float('inf'):
        raise ValueError(
            "`disutility_cutoff_constant` is set while `disutility_cutoff_coefficient` is unset. "
            "Either both or neither should be specified"
        )
    elif disutility_cutoff_constant == float('inf') and disutility_cutoff_coefficient != float('inf'):
        raise ValueError(
            "`disutility_cutoff_coefficient` is set while `disutility_cutoff_constant` is unset. "
            "Either both or neither should be specified"
        )


    path_overlap.resize(costs.size(), 0.0)
    probabilities.resize(costs.size(), 0.0)
    # Group positions rather than labels: dataframe indices need not be unique.
    for group in df.groupby(["origin id", "destination id"], sort=False).indices.values():
        positions = group
        paths.clear()
        turns.clear()
        route_costs.clear()
        mask.clear()
        for position in positions:
            paths.push_back(&d(routes[position]))
            turns.push_back(&turn_steps[position])
            route_costs.push_back(costs[position])
            mask.push_back(route_mask[position])
        with nogil:
            RouteChoiceSetResults.compute_psl(
                paths, turns, route_costs, mask, overlap, probability, link_costs, beta, disutility_cutoff_constant, disutility_cutoff_coefficient
            )
        for j in range(positions.size()):
            position = positions[j]
            route_mask[position] = mask[j]
            path_overlap[position] = overlap[j]
            probabilities[position] = probability[j]


cdef object imported_route_dataframe(
    object df, const vector[double] &costs, const vector[bint] &mask,
    const vector[double] &overlap, const vector[double] &probabilities
):
    """Copy numeric results to public columns without exposing temporary native storage."""
    cdef size_t size = costs.size()
    cdef const double *cost_data = costs.data()
    cdef const bint *mask_data = mask.data()
    cdef const double *overlap_data = overlap.data()
    cdef const double *probability_data = probabilities.data()
    columns = {}
    columns["cost"] = np.asarray(<const double[:size]>cost_data).copy() if size else np.empty(0)
    columns["mask"] = np.asarray(<const bint[:size]>mask_data).astype(bool) if size else np.empty(0, dtype=bool)
    columns["path overlap"] = np.asarray(<const double[:size]>overlap_data).copy() if size else np.empty(0)
    columns["probability"] = np.asarray(<const double[:size]>probability_data).copy() if size else np.empty(0)
    return df.assign(**columns)


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
