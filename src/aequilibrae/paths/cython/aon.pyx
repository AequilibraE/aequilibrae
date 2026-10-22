# cython: language_level=3
"""
Assignment orchestration over independent routing and operation components.

The dispatch of all this code can be hard to follow due to the mixing of fused
types, C++ templates, ctypedefs, and conditional dispatch. I'll try to explain
it all here.

In the beginning User created the Python objects.

We have some data that describes a particular combination of:

Routing type:
  - Node based
  - Turn based

Demand type:
  - Dense
  - Sparse

On the C++ level we have instances of templated functions, one for every
combination we of types we use. We need a way to go from runtime types to the
correct C++ code. So we spell out all the combinations in an if-ladder in
``PreparedAoN.run``.

While we do have spell out the initial dispatch to each method, we don't want
to write a different function for each combination, so the ``PreparedAoN.run``
method calls the different instances of the 'fused type' functions. These are a
Cython feature that let's us do templating like C++, in Cython. I'll use
'templating' to refer to both C++ templating and Cython fused types because they
are functionally the same here.

``assign_origins`` is templated on the ``routing`` and ``query``
arguments. These are a ``RoutingContext`` or ``AoNOriginView`` respectively.

RoutingContext is either a:
  - NodeBasedContext, or
  - TurnBasedContext.

These are both instances of the ``GraphContext`` Cython extension type (Python
object). So we check the instance type and use that as a conditional.

AoNOriginView (fused type) is either a:
  - CppAoNOrigin[DenseDemandRow] or
  - CppAoNOrigin[SparseDemandRow].

CppAoNOrigin is a C++ template for the
aequilibrae::paths::cpp::routing::AoNOrigin struct. This struct holds a
SearchQuery (concrete type) 'search' and a DemandRow (templated type)
'loading'. For this fused type, the DemandRow is either a DenseDemandRow or
SparseDemandRow.

DenseDemandRow is a ctypedef for a CppDenseMatrixOriginView[const double].
SpraseDemandRow is a ctypedef for a CppSpraseMatrixOriginView[const double].

CppDenseMatrixOriginView is a C++ template for the
aequilibrae::matrix::cpp::DenseMatrixOriginView class, and
CppSpraseMatrixOriginView is a C++ template for the
aequilibrae::matrix::cpp::SpraseMatrixOriginView class.

These two templates are single origin views into the CppSparseMatrix
(aequilibrae::matrix::cpp::SparseMatrix) and CppDenseMatrix
(aequilibrae::matrix::cpp::DenseMatrix) templates.

These matrix templates are C++ views of the SparseMatrix and DenseMatrix Cython
extension types! We create them via the .view() cdef function. We then call
.origin() on these C++ classes and store the results in the
PreparedAoN.sparse_origin_queries and PreparedAoN.dense_origin_queries
vectors. We populate them in the __prepare_origins method.

Cython see these fused type functions and generates each combination for each
of the static types. It doesn't really care about the C++ templates other than
that they exist and are declared (in pxd files). The C++ compiler then sees
these and generates the correct C++ code for each instance. Tying the if-ladder
to the correct static types.
"""

import operator

import numpy as np
cimport cython

from cython.parallel cimport prange, threadid
from libcpp.vector cimport vector
from libcpp cimport bool as cpp_bool

from aequilibrae.matrix.cython.dense_matrix cimport DenseMatrix
from aequilibrae.matrix.cython.sparse_matrix cimport SparseMatrix
from aequilibrae.matrix.cython.matrix_base cimport _MatrixBase
from aequilibrae.matrix.cython.views cimport CppDenseMatrix, CppSparseMatrixOriginView, CppDenseMatrixOriginView, CppSparseMatrix

from aequilibrae.utils.cython.array_allocations import readonly_view
from aequilibrae.paths.cython.workspaces cimport AoNWorkspace
from aequilibrae.paths.cython.dijkstra cimport RoutingContext, cpp_dijkstra
from aequilibrae.paths.cython.dijkstra import validate_routing_heap
from aequilibrae.paths.cython.context cimport (
    GraphContext,
    NodeBasedContext,
    TurnBasedContext,
    SelectLinkContext,
    SkimmingContext,
)
from aequilibrae.paths.cython.search_results cimport SearchResults, CppSearchResults
from aequilibrae.paths.cython.demand cimport DemandRowView
from aequilibrae.paths.cython.demand import validate_demand, choose_origins, make_destination_masks
from aequilibrae.paths.cython.skimming cimport cpp_skimming
from aequilibrae.paths.cython.network_loading cimport (
    cpp_network_loading,
    cpp_reduce_loading_outputs,
    cpp_sum_weighted_turn_costs,
    cpp_sum_unassigned_demand,
)
from aequilibrae.paths.cython.outputs cimport (
    AoNOutputs,
    CppAoNOutputsView,
    LoadingOutputs,
    SelectLinkLoadingOutputs,
    CppSelectLinkODOriginView,
    CppSelectLinkODOutputsView,
)
from aequilibrae.paths.cython.select_link_loading cimport (
    cpp_select_link_loading,
    cpp_reduce_select_link_loading,
)


cdef class _AoNWorker:
    """Keep one worker's owners together for the lifetime of their cached views."""

    cdef SearchResults search
    cdef AoNWorkspace workspace
    cdef LoadingOutputs loading
    cdef SelectLinkLoadingOutputs selected_loading

    def __init__(
        self,
        GraphContext context,
        heap,
        class_count,
        skim_width,
        selection_names,
        load_selected_links,
        write_selected_od,
        zone_count,
        demand_names,
    ):
        self.search = SearchResults(context.node_count, context.state_count, context.link_count)
        self.workspace = AoNWorkspace(
            context.node_count,
            context.state_count,
            heap=heap,
            class_count=class_count,
            field_count=skim_width,
            select_links=load_selected_links or write_selected_od,
            od_zone_count=zone_count if write_selected_od else None,
            demand_names=demand_names,
            set_names=selection_names,
        )
        self.loading = LoadingOutputs(context.link_count, class_count)
        if load_selected_links:
            self.selected_loading = SelectLinkLoadingOutputs(
                context.link_count, class_count, selection_names,
            )

    cdef CppAoNWorkerView view(self) noexcept nogil:
        cdef CppAoNWorkerView worker

        worker.search = self.search.view()
        worker.workspace = self.workspace.view()
        worker.loading = self.loading.view()

        if self.selected_loading is not None:
            worker.selected_loading = self.selected_loading.view()

        return worker


cdef class PreparedAoN:
    """Reuse assignment workers with a caller-configured routing context.

    Packed DenseMatrix or SparseMatrix demand [zones, zones, classes], topology,
    restrictions, origins and targets stay fixed. Only routing costs change
    between iterations: update the supplied context, or write valid costs into
    its borrowed buffer, before run(). The driver retains that context directly
    and refreshes its view each run.

    Skimming and selection inputs are independent owners with fixed configuration
    and buffer bindings. Ordinary loading and weighted turn accounting always run.
    This is not a general multi-origin executor.
    Do not mutate inputs or use a worker or output concurrently with a run.
    """

    cdef GraphContext context
    cdef _MatrixBase demand
    cdef const cpp_bool[:, ::1] destination_masks_buffer
    cdef const size_t[::1] destination_counts_buffer, origins_buffer
    cdef SkimmingContext skim_context
    cdef SelectLinkContext selection_context
    cdef bint load_selected_links, write_selected_od
    cdef size_t zone_count, class_count
    cdef int cores
    cdef object heap_name

    cdef list workers
    cdef vector[CppAoNOrigin[DenseDemandRow]] dense_origin_queries
    cdef vector[CppAoNOrigin[SparseDemandRow]] sparse_origin_queries
    cdef vector[CppAoNWorkerView] worker_views
    cdef CppAoNInputs inputs

    # The standalone reductions require contiguous tables of component views
    cdef vector[CppLoadingOutputs[double]] loading_outputs
    cdef vector[CppSelectLinkLoadingOutputsView[double]] selected_outputs
    cdef vector[CppSelectLinkODOutputsView[double]] selected_od_outputs

    def __init__(
        self,
        context,
        demand,
        *,
        cores=1,
        SkimmingContext skimming=None,
        origins=None,
        SelectLinkContext selected_links=None,
        select_link_loads=True,
        select_link_od=True,
        heap="4ary",
    ):
        if self.context is not None:
            raise RuntimeError("PreparedAoN cannot be reinitialised")
        validate_routing_heap(heap)
        self.heap_name = heap
        if not isinstance(context, (NodeBasedContext, TurnBasedContext)):
            raise TypeError("context must be a node or turn routing context")

        cores = operator.index(cores)
        if not 1 <= cores <= np.iinfo(np.int32).max:
            raise ValueError("cores must be positive and fit an OpenMP thread count")

        shape = validate_demand(demand, context.node_count)

        if skimming is not None and skimming.link_count != context.link_count:
            raise ValueError("skimming link_count does not match routing context")
        if selected_links is not None and selected_links.link_count != context.link_count:
            raise ValueError("selection link_count does not match routing context")

        self.context = context
        self.cores = cores
        self.demand = demand
        self.zone_count = shape[0]
        self.class_count = shape[2]

        self.skim_context = skimming
        self.selection_context = selected_links

        if skimming is not None:
            self.inputs.skimming = skimming.view()
        if selected_links is not None:
            self.inputs.selection = selected_links.view()

        self.load_selected_links = self.inputs.selection.set_count > 0 and bool(select_link_loads)
        self.write_selected_od = self.inputs.selection.set_count > 0 and bool(select_link_od)

        self.origins_buffer = choose_origins(origins, demand, self.inputs.skimming.field_count > 0)
        self.destination_masks_buffer, self.destination_counts_buffer = make_destination_masks(
            demand, context.node_count, self.inputs.skimming.field_count > 0,
        )

        self.__prepare_origins()
        self.__prepare_workers()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void __prepare_origins(self) except *:
        """Bind each origin's demand and targets once, without Python row owners."""
        cdef size_t origin, mask_row
        cdef CppSearchQuery search

        cdef CppAoNOrigin[SparseDemandRow] sparse_query
        cdef CppAoNOrigin[DenseDemandRow] dense_query

        cdef CppSparseMatrix[double] sparse_view
        cdef CppDenseMatrix[double] dense_view

        cdef CppSparseMatrixOriginView[double] sparse_row
        cdef CppDenseMatrixOriginView[double] dense_row
        cdef bint sparse = isinstance(self.demand, SparseMatrix)
        if sparse:
            sparse_view = (<SparseMatrix>self.demand).view[double]()
        else:
            dense_view = (<DenseMatrix>self.demand).view[double]()

        for origin in self.origins_buffer:
            search = CppSearchQuery()
            mask_row = 0 if self.inputs.skimming.field_count else origin
            search.node_count = self.context.node_count
            search.origin = origin
            search.target_count = self.destination_counts_buffer[mask_row]
            if search.target_count:
                search.target_mask = &self.destination_masks_buffer[mask_row, 0]

            # An explicitly requested origin with no demand keeps the full-search
            # query. Automatic origin selection omits these rows without skims.
            if sparse:
                sparse_row = sparse_view.origin(origin)
                sparse_query.search = search
                sparse_query.loading.destination_count = sparse_row.destination_count
                sparse_query.loading.field_count = sparse_row.field_count
                sparse_query.loading.data = sparse_row.data
                self.sparse_origin_queries.push_back(sparse_query)
            else:
                dense_row = dense_view.origin(origin)
                dense_query.search = search
                dense_query.loading.destination_count = dense_row.destination_count
                dense_query.loading.field_count = dense_row.field_count
                dense_query.loading.data = dense_row.data
                self.dense_origin_queries.push_back(dense_query)

    cdef void __prepare_workers(self) except *:
        """Allocate only the scratch and accumulators required by each worker."""
        cdef _AoNWorker worker
        cdef CppAoNWorkerView view

        # Label-only skimming reads search labels and needs no state-sum buffer.
        skim_width = None
        if self.inputs.skimming.needs_state_sums():
            skim_width = self.inputs.skimming.additive_field_count

        selection_names = ()
        if self.selection_context is not None:
            selection_names = self.selection_context.set_names

        self.workers = []
        for _ in range(self.cores):
            worker = _AoNWorker(
                self.context,
                self.heap_name,
                self.class_count,
                skim_width,
                selection_names,
                self.load_selected_links,
                self.write_selected_od,
                self.zone_count,
                self.demand.field_names,
            )
            self.workers.append(worker)

            view = worker.view()
            self.worker_views.push_back(view)
            self.loading_outputs.push_back(view.loading)
            if self.load_selected_links:
                self.selected_outputs.push_back(view.selected_loading)
            if self.write_selected_od:
                self.selected_od_outputs.push_back(view.workspace.selected_od)

    @property
    def origins(self):
        return readonly_view(self.origins_buffer)

    @property
    def destination_masks(self):
        return readonly_view(self.destination_masks_buffer)

    @property
    def destination_counts(self):
        return readonly_view(self.destination_counts_buffer)

    def make_outputs(self):
        """Allocate a matching group of outputs, caller must hold on to it."""
        skim_names = () if self.skim_context is None else self.skim_context.field_names
        selection_names = () if self.selection_context is None else self.selection_context.set_names

        return AoNOutputs(
            self.context.link_count,
            self.zone_count,
            self.class_count,
            skim_names=skim_names,
            select_link_names=selection_names,
            select_link_loads=self.load_selected_links,
            select_link_od=self.write_selected_od,
            demand_names=self.demand.field_names,
        )

    cdef void validate_outputs(self, AoNOutputs out) except *:
        """Check actual components before clearing caller output or worker totals."""
        if (
            out.loading.link_count != self.context.link_count
            or out.loading.class_count != self.class_count
        ):
            raise ValueError("loading output dimensions must match assignment")

        if (out.skimming is not None) != (self.inputs.skimming.field_count > 0):
            raise ValueError("skimming output presence must match assignment")

        if out.skimming is not None:
            if (
                out.skimming.origin_count != self.zone_count
                or out.skimming.destination_count != self.zone_count
                or out.skimming.field_names != self.skim_context.field_names
            ):
                raise ValueError("skimming output dimensions and ordered names must match assignment")

        if (out.select_link is not None) != (self.load_selected_links or self.write_selected_od):
            raise ValueError("selected output components must match assignment configuration")

        if out.select_link is None:
            return

        if (
            (out.select_link.loading is not None) != self.load_selected_links
            or (out.select_link.od is not None) != self.write_selected_od
        ):
            raise ValueError("selected output components must match assignment configuration")

        if self.load_selected_links:
            if (
                out.select_link.loading.link_count != self.context.link_count
                or out.select_link.loading.class_count != self.class_count
                or out.select_link.loading.set_names != self.selection_context.set_names
            ):
                raise ValueError("selected loading dimensions and ordered names must match assignment")

        if self.write_selected_od:
            if (
                out.select_link.od.origin_count != self.zone_count
                or out.select_link.od.destination_count != self.zone_count
                or out.select_link.od.class_count != self.class_count
                or out.select_link.od.set_names != self.selection_context.set_names
                or out.select_link.od.field_names != self.demand.field_names
            ):
                raise ValueError("selected OD dimensions and ordered names must match assignment")

    def run(self, AoNOutputs out not None):
        """Replace this iteration's outputs, reusing all prepared worker storage."""
        cdef size_t worker
        cdef CppAoNOutputsView output
        cdef double turn_cost_total = 0
        cdef double unassigned_demand = 0
        cdef bint node_based = isinstance(self.context, NodeBasedContext)
        cdef bint sparse = isinstance(self.demand, SparseMatrix)

        self.validate_outputs(out)
        if self.write_selected_od:
            out.select_link.od.reset()
        output = out.view()

        with nogil:
            for worker in range(self.worker_views.size()):
                self.worker_views[worker].reset()

            # Clear skipped skim rows as well. Link accumulators and sparse OD
            # results are replaced by the reductions below.
            output.skimming.reset()

            if sparse:
                if node_based:
                    assign_origins(
                        <NodeBasedContext>self.context, self.sparse_origin_queries.data(),
                        self.sparse_origin_queries.size(), self.inputs,
                        self.worker_views.data(), self.cores, output,
                    )
                else:
                    assign_origins(
                        <TurnBasedContext>self.context, self.sparse_origin_queries.data(),
                        self.sparse_origin_queries.size(), self.inputs,
                        self.worker_views.data(), self.cores, output,
                    )
            else:
                if node_based:
                    assign_origins(
                        <NodeBasedContext>self.context, self.dense_origin_queries.data(),
                        self.dense_origin_queries.size(), self.inputs,
                        self.worker_views.data(), self.cores, output,
                    )
                else:
                    assign_origins(
                        <TurnBasedContext>self.context, self.dense_origin_queries.data(),
                        self.dense_origin_queries.size(), self.inputs,
                        self.worker_views.data(), self.cores, output,
                    )

            cpp_reduce_loading_outputs[double](
                self.loading_outputs.data(),
                self.loading_outputs.size(),
                output.loading,
            )
            if self.load_selected_links or self.write_selected_od:
                cpp_reduce_select_link_loading[double](
                    self.selected_outputs.data(),
                    self.selected_outputs.size(),
                    output.selected_loading,
                    self.selected_od_outputs.data(),
                    self.selected_od_outputs.size(),
                    output.selected_od,
                )

            for worker in range(self.worker_views.size()):
                turn_cost_total += self.worker_views[worker].turn_cost_total
                unassigned_demand += self.worker_views[worker].unassigned_demand

        if self.write_selected_od:
            out.select_link.od.update_matrices()
        out.turn_cost_total = turn_cost_total
        out.unassigned_demand = unassigned_demand
        return out


cdef void assign_origin(
    RoutingContext routing,
    const AoNOriginView &query,
    const CppAoNInputs &inputs,
    CppAoNWorkerView &worker,
    const CppAoNOutputsView &output,
) noexcept nogil:
    """Compose operations on one search, without consulting Python owners."""
    cdef CppSearchResults search = worker.search.read_view()
    cdef CppSelectLinkODOriginView[double] selected_od
    cdef size_t origin = query.search.origin

    cpp_dijkstra(routing.view(), query.search, worker.search, worker.workspace.search)

    if inputs.skimming.field_count:
        cpp_skimming[double](
            search,
            inputs.skimming,
            worker.workspace.skimming,
            output.skimming.origin(origin),
        )

    if worker.workspace.selected_od.set_count:
        selected_od = worker.workspace.selected_od.origin(origin)
    load_origin(search, query.loading, inputs, worker, selected_od)


cdef void load_origin(
    const CppSearchResults &search,
    const DemandRowView &demand,
    const CppAoNInputs &inputs,
    CppAoNWorkerView &worker,
    const CppSelectLinkODOriginView[double] &selected_od,
) noexcept nogil:
    # OD builders and selected link accumulators remain worker-local until
    # reduction, and can reuse the ordinary loading scratch sequentially.
    cpp_network_loading(search, demand, worker.workspace.loading, worker.loading)
    cpp_select_link_loading(
        search, demand, inputs.selection, worker.workspace.select_link,
        worker.workspace.loading, worker.selected_loading, selected_od,
    )
    worker.turn_cost_total += cpp_sum_weighted_turn_costs(search, demand)
    worker.unassigned_demand += cpp_sum_unassigned_demand(search, demand)


cdef void assign_origins(
    RoutingContext routing,
    const AoNOriginView *origins,
    size_t origin_count,
    const CppAoNInputs &inputs,
    CppAoNWorkerView *workers,
    int cores,
    const CppAoNOutputsView &output,
) noexcept nogil:
    cdef Py_ssize_t index

    # MSVC requires the OpenMP loop counter generated by prange to be signed.
    # origin_count cannot exceed the Python-backed demand dimension (Py_ssize_t).
    for index in prange(<Py_ssize_t>origin_count, num_threads=cores, schedule="guided"):
        assign_origin(routing, origins[index], inputs, workers[threadid()], output)
