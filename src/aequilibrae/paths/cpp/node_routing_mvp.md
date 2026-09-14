# Internal routing building blocks

This interface is separate from the public `Graph`, `PathResults` and assignment
APIs. It is being simplified before integration. There is no compatibility layer
for the earlier routing-context MVP.

## First slice: context, query, results

A search has three independent owners:

| Object | Owns or retains | Does not hold |
| --- | --- | --- |
| `NodeBasedContext` / `TurnBasedContext` | Fixed topology, borrowed current costs, routing restrictions | Query, results, demand, skims or operation scratch |
| `SearchQuery` | Origin and an optional borrowed target mask, with its count | Graph or results |
| `SearchResults` | Fixed-size path buffers and completion metadata | Context, query, node-path mapping, loading or skimming scratch |

Each Cython object creates a C++ `view()` that borrows its storage. A view does
not allocate or extend the owner's lifetime. Keep owners alive for the call.
Use separate results for concurrent workers; contexts and fixed target masks can
be shared. Inputs must be used consistently by internal callers. There are no
locks or runtime checks for concurrent changes or buffer overlap.

### Search example

```python
import numpy as np

from aequilibrae.paths.cython.graph_context import NodeBasedContext, TurnBasedContext
from aequilibrae.paths.cython.queries import SearchQuery
from aequilibrae.paths.cython.search_results import SearchResults
from aequilibrae.paths.cython.dijkstra import dijkstra

# Links: 0: 0->1, 1: 0->2, 2: 1->3, 3: 2->1.
costs = np.ones(4, dtype=np.float64)
context = TurnBasedContext(
    [0, 2, 3, 4, 4], [1, 2, 3, 1], costs,
    turn_fs=[0, 1, 1, 1, 1],
    turn_to_links=[2],
    turn_penalties=[10.0],
)
query = SearchQuery(4, origin=0, target_mask=np.array([False, True, False, True]))
results = SearchResults(context.node_count, context.state_count, context.link_count)

dijkstra(context, query, results)
assert results.path_links_to(1).tolist() == [0]
assert results.path_links_to(3).tolist() == [1, 3, 2]
assert results.path_cost_to(3) == 3.0
assert results.path_turn_cost_to(3) == 0.0
assert results.all_targets_reached

# Reuse the same buffers for another origin and the same target set.
query.origin = 2
dijkstra(context, query, results)

# Or search the entire reachable state space. No mask is needed.
full_query = SearchQuery(context.node_count, origin=0)
dijkstra(context, full_query, results)
assert results.exhausted
```

One-shot use allocates these same objects and calls the same function. There is
no optional result allocation inside `dijkstra`, and no context result factory.

### Routing contexts

Topology is copied once into contiguous snapshots. Index arrays use `np.uintp`;
node and link indices are local to the context. Connectors are CSR link positions,
so there is no separate identity link-ID allocation. External IDs and compressed
network mappings belong outside this layer.

Routing costs are borrowed through a const memoryview. Supply an aligned,
contiguous `float64` buffer with one cost per link. Negative values and NaN are
rejected; positive infinity blocks a link. Lists, strided costs and dtype
conversion are not supported at this internal boundary. Construction does not
change the caller's writeability flags.

- `context.update_costs(costs)` validates and rebinds the objective without copying.
  A failed update preserves the old binding.
- `context.with_costs(costs)` creates a context of the same type sharing topology
  and turn buffers, with an independent cost binding.

Cost changes do not change results already computed. Results store their labels,
not a pointer to the objective. To observe new costs, run another search.

Both contexts accept `blocked_centroid_count=0`. When nonzero, nodes in that prefix
may start or end paths but cannot be used as intermediate nodes, except for the
query origin. This rule is applied in routing, not by rewriting graph heads in
the assignment loop. Turn restrictions remain effective alongside it.

Turn tables retain the existing representation:

- `turn_fs` has `link_count + 1` entries, grouping turns by incoming link.
- `turn_to_links` is strictly increasing within each row. Each pair must connect
  consecutive directed links.
- Penalties are nonnegative; positive infinity is a ban.
- Missing turns have zero penalty, subject to `allow_uturns`.
- Explicit finite turns override the default U-turn ban. Explicit infinite turns
  remain banned. The first link leaving the source pays no turn cost.

### Search queries

`queries.hpp`, `queries.pxd` and `queries.pyx` group the search and loading
query owners. `outputs.hpp`, `outputs.pxd` and `outputs.pyx` group operation
outputs as they are extracted.

`SearchQuery(node_count, origin, target_mask=None)` separates inputs that vary
between origins from the graph data.

`None` is the only full-search representation. Otherwise supply a contiguous
boolean buffer with one entry per physical node and at least one target. The
mask is borrowed and counted at construction. Do not change its contents while
using that query, because its cached count must continue to match. Empty masks
are rejected rather than carrying a second meaning for full search.

`query.origin` can change between searches without allocating or recounting
anything. The query owns no graph reference. A caller needing a different target
set constructs another query; assignment can instead build a local C++ query
view of an already prepared mask row and count.

The C++ kernel reads the mask without scanning or modifying it. It counts each
physical node only at its first settled arrival, even in turn routing.

### Results and path queries

| Buffer | Meaning |
| --- | --- |
| `predecessors[s]` | Parent state along the selected path |
| `connectors[s]` | Directed link used from the parent into this state |
| `settlement_order[:settled_count]` | Finalized states, parents before children |
| `terminal_states[d]` | Chosen state for a path ending at physical node `d` |
| `distances[s]` | Total routing objective, including turn costs |
| `turn_costs[s]` | Turn-cost component already included in the distance |

Node routing has one state per node and uses the origin as its root. Turn routing
has one state per incoming link plus a virtual root at `link_count`.

The cheapest arrival at an intermediate node need not be the arrival used on a
path through that node. In the example, `terminal_states[1]` is link state 0, but
the parent of `terminal_states[3]` is link state 3. Consumers must follow state
predecessors, not replace them with the intermediate node's terminal.

The root has zero labels and no parent or connector. Unfinalized states have
infinite labels and sentinel indices (`np.iinfo(np.uintp).max`). A missing
terminal can mean unreachable or simply not finalized by a partial search.

Results expose only destination-specific path queries:

- `reachable_to(destination)` reports whether a finalized path is available.
- `path_links_to(destination)` reconstructs local directed links.
- `path_cost_to(destination)` returns the routing objective.
- `path_turn_cost_to(destination)` returns its turn-cost component.

A missing path has an empty link array and infinite costs. An origin-to-itself
path has no links and zero costs. There is no node-path reconstruction method;
callers needing physical nodes use their graph's link heads. Results do not store
an additional node per state or retain a graph for this purpose.

Completion metadata includes `origin`, `root`, `settled_count`, `target_count`,
`reached_target_count` and `exhausted`. `all_targets_reached` is false before a
search; afterwards it compares the two target counts. A full search has no
targets. `exhausted` becomes true only when the reachable state space has been
searched. Stopping at the last target is not exhaustion: its outgoing links
have not been explored, even if the heap happens to be empty at that point.

Results can be reused with any context whose node, state and link counts match.
This checks storage dimensions, not network identity. The caller remains
responsible for matching downstream link fields to the searched network.

Public buffer properties are read-only, zero-copy NumPy views. They keep storage
alive after the results wrapper is deleted. A later search overwrites retained
views; use `.copy()` for a snapshot. Result dimensions cannot change.

### Cython and C++ boundary

Routing accepts `context.view()`, `query.view()` and `results.view()` without the
GIL. Downstream kernels use `results.read_view()`, which has const buffer pointers.
Neither result view retains inputs. The metadata record belongs to the Cython
results object; views point to it rather than copying its counters. This ensures
that searches performed through local view copies still update the owner.

The routing kernels live in `dijkstra.hpp`. Legacy production algorithms remain
in `path_finding.hpp`. The new kernels still allocate their own four-ary heap on
each search; persistent heap storage and other heap choices are deferred.

## Second slice: workspaces and network loading

### Fixed-size operation workspaces

All workspace types live together in `workspaces.hpp`, `workspaces.pxd` and
`workspaces.pyx`. They remain independent objects:

| Object | Allocation | Used by |
| --- | --- | --- |
| `LoadingWorkspace(states, classes)` | State demand totals `[states, classes]` | Ordinary and selected loading |
| `SkimmingWorkspace(states, fields)` | Additive field sums `[states, fields]` | Field skimming |
| `SelectLinkWorkspace(states)` | One boolean path-membership flag per state | Select-link analysis |

None retains a graph, query, search results or output. Dimensions are fixed at
construction. There are no `prepare_*` or resize methods. Kernels replace their
own scratch before using it, so callers need not clear scratch between origins.
Read-only buffer views keep their allocations alive and reflect the last operation
that wrote them. Zero class and field widths produce empty buffers.

`AoNWorkspace` allocates an optional group of these objects:

```python
from aequilibrae.paths.cython.workspaces import AoNWorkspace

workspace = AoNWorkspace(
    context.state_count, class_count=2, field_count=3, select_links=True,
)
loading_scratch = workspace.loading
skim_scratch = workspace.skimming
selection_scratch = workspace.select_link
```

Omitting a width leaves that component `None`; `select_links=False` leaves flags
unallocated. Components can be used on their own and outlive the group. Each
C++ kernel takes only its required component views, never `AoNWorkspace`.
Select-link loading takes both selection flags and loading scratch explicitly,
so it can reuse the same cascade allocation as ordinary loading.

### Loading query and output

`LoadingQuery(demand)` borrows one origin's aligned, contiguous `float64` demand
buffer `[destinations, classes]`. It exposes a read-only `demand` view without
changing the caller's writeability flags. Values may change between calls; the
shape and allocation stay fixed. Lists, strided buffers and dtype conversion
are not accepted. The caller selects the row corresponding to the search origin;
the query does not retain or identify a search.

`LoadingOutputs(links, classes)` owns a zero-initialized link accumulation buffer.
It exposes read-only `link_loads`, supports explicit `reset()`, and holds no input
or scratch references. Both axes may be zero. Reset clears existing storage,
not retained historical snapshots or any workspace. Copy a NumPy view when a
snapshot is needed; output objects have no separate snapshot API.

```python
from aequilibrae.paths.cython.queries import LoadingQuery
from aequilibrae.paths.cython.outputs import LoadingOutputs
from aequilibrae.paths.cython.workspaces import LoadingWorkspace
from aequilibrae.paths.cython.network_loading import network_loading

# Use the finalized origin-0 paths from the search example above.
demand = np.ones((context.node_count, 2), dtype=np.float64)
loading_query = LoadingQuery(demand)
loading_scratch = LoadingWorkspace(results.state_count, 2)
loads = LoadingOutputs(results.link_count, 2)

network_loading(results, loading_query, loading_scratch, loads)
link_loads = loads.link_loads  # Read-only, zero-copy view.
loads.reset()                # Clear once before the next iteration's origins.
```

`network_loading(results, query, workspace, output)` checks dimensions before
writing and returns the supplied output. It releases the GIL and calls the same
C++ kernel used by assignment. The kernel accepts the four typed views and does
not allocate, change the result tree, retain inputs or clear link output.

Demand destinations are physical nodes `0..destination_count-1`; paths may use
states outside that prefix. Only finalized, non-intrazonal demand is loaded.
Missing terminals are ignored, even in partial searches; loading never extends
a search. Before a search it loads nothing. Empty demand replaces scratch with
zero totals but adds no link demand. Negative and nonfinite values use ordinary
floating-point addition along their selected paths.

The kernel seeds demand at each node's chosen terminal, then cascades it in reverse
settlement order. This follows turn histories without a routing-mode branch.
At completion, `workspace.state_loads` contains subtree demand, the root contains
all reachable non-intrazonal demand, and unfinalized states are zero. Ordinary
and selected loading share this reverse pass; selected loading only changes how
terminal demand is seeded.

### Worker reduction and assignment use

Each worker accumulates into its own `LoadingOutputs`. After workers finish,
`reduce_loading_outputs(workers, output)` replaces a distinct output with their
sum. All dimensions must match; checks happen before resetting the target.
Workers remain unchanged. An empty worker list resets output to zero.

`PreparedAoN` now retains one `LoadingOutputs` per worker rather than a separate
thread-axis load cube. It prepares a table of views once, resets the worker
accumulators before each iteration, and reduces them using the same C++ reduction
as the standalone entry point. Queries borrow already prepared origin rows, so
the origin loop builds no Python query objects or numeric buffers.

`AoNOutputs.loading` is an independent `LoadingOutputs`; `AoNOutputs.link_loads`
forwards its read-only view. Other output components have not yet been separated.
The old aggregate `copy()` / `copy_to()` and thread-load cube accessor are removed.
Output rotation remains ordinary reference rotation.

## Remaining downstream work

`SearchResults` has no loading, skimming, select-link or preparation methods.
Standalone loading is available now. Skimming and select-link kernels have been
adapted to the small workspaces; their Python input/output interfaces remain for
the next slices. The old `SkimmingContext` still awaits its split into inputs and
outputs, with one ordered skim array and named zero-copy matrix views.

`PreparedAoN` remains a testable consumer rather than the final integration API.
It borrows demand, routing costs and skim fields, and uses context-owned cost
bindings and centroid blocking. Target masks and origin selection are prepared
once: changing demand magnitudes is possible, but introducing new search targets
requires new preparation. Further work will separate its remaining outputs and
skim configuration, then reduce the driver to composition of those objects.

`aon_graph.py`, its adapter exports and legacy integration tests have been removed.
Production dispatch is unchanged.

## Validation

```sh
meson compile -C build && \
ASAN_OPTIONS=detect_leaks=1 \
LSAN_OPTIONS=suppressions=$(readlink -f .github/workflows/asan_suppressions.txt) \
LD_PRELOAD=$(gcc -print-file-name=libasan.so) \
AEQ_SHOW_PROGRESS=0 \
python -m pytest -q -s \
    tests/aequilibrae/paths/test_node_routing_mvp.py \
    tests/aequilibrae/paths/test_turn_routing_mvp.py \
    tests/aequilibrae/paths/test_context_skimming.py \
    tests/aequilibrae/paths/test_context_network_loading.py \
    tests/aequilibrae/paths/test_aon_context.py \
    tests/aequilibrae/utils/test_array_allocations.py
```

The new routing tests check independent NetworkX distances and expanded turn
states, partial-search cleanup, zero-cost cycles, U-turn rules, borrowed inputs,
context-independent reuse, metadata updates, view lifetimes and separate workers.
Standalone loading tests check partial and empty queries, demand borrowing,
fixed buffer reuse, read-only views, independent lifetimes, dimension validation,
worker-local accumulation and reduction. Driver tests compare skimming, ordinary
loading and select-link loading against OD-by-OD path walks.
