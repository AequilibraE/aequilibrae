# Standalone routing-context MVP

This remains separate from `Graph`, `PathResults`, assignment, production
skimming, select-link analysis, and the public `aequilibrae.paths` exports.
Single-origin state-tree skimming and network loading are available as standalone utilities.

## Python API

Both contexts use the same entry point and results class. The entry point is exposed by the dedicated Dijkstra module.

```python
from aequilibrae.paths.cython.graph_context import NodeBasedContext
from aequilibrae.paths.cython.graph_context import TurnBasedContext
from aequilibrae.paths.cython.dijkstra import dijkstra

# Links: 0: 0->1 (1), 1: 0->2 (1), 2: 1->3 (1), 3: 2->1 (1).
fs = [0, 2, 3, 4, 4]
heads = [1, 2, 3, 1]
costs = [1.0, 1.0, 1.0, 1.0]
node_context = NodeBasedContext(fs, heads, costs)
turn_context = TurnBasedContext(
    fs, heads, costs,
    turn_fs=[0, 1, 1, 1, 1],
    turn_to_links=[2],
    turn_penalties=[10.0],  # Arriving via link 0, continuing via link 2.
)

assert dijkstra(node_context, 0, 3).path_nodes.tolist() == [0, 1, 3]
results = turn_context.make_results()
dijkstra(turn_context, 0, [1, 3], results)
assert results.destinations.tolist() == [1, 3]
assert results.path_nodes_to(3).tolist() == [0, 2, 1, 3]
assert results.path_links_to(3).tolist() == [1, 3, 2]
assert results.path_cost_to(3) == 3.0
assert results.path_turn_cost_to(3) == 0.0

# The cheapest arrival at node 1 is NOT the arrival used en route to node 3.
assert results.terminal_states[1] == 0
assert results.predecessors[results.terminal_states[3]] == 3

predecessors = results.predecessors  # Read-only, zero-copy NumPy view.
snapshot = predecessors.copy()      # Independent historical result.
dijkstra(turn_context, 1, 3, results)  # Reuses allocations; predecessors changes.
```

The destinations argument can be one node, an iterable of nodes, a boolean mask
with `node_count` entries, or `None` for every node. Duplicate node indices are
ignored. An empty iterable or all-false mask disables early exit. Otherwise a
search ends when every requested physical node has settled or the reachable
state space is exhausted. `reachable` reports whether all requested
nodes were reached; the `*_to(destination)` methods inspect individual paths.
The singular path properties remain available when exactly one node was
requested.

`SearchResults(context)` also constructs compatible results directly. Results
from a different context are rejected, even if the buffer shapes match.

## Single-origin skimming

### Prepared fields and OD results

For repeated origins and iterations, prepare inputs and OD arrays once:

```python
import numpy as np
from aequilibrae.paths.cython.skimming_context import SkimmingContext

z = 2  # Centroids are context nodes 0 to z-1.
link_distance = np.ones(turn_context.link_count, dtype=np.float64)
skims = SkimmingContext(
    turn_context, [link_distance], centroid_count=z,
    include_costs=True, include_turn_costs=True,
)
results = turn_context.make_results()
results.prepare_skims(skims)  # Optional: allocate this worker's scratch up front.

for iteration in range(100):
    for origin in range(z):
        dijkstra(turn_context, origin, range(z), results)
        results.skim_fields(skims)

# Read-only, zero-copy views; retained views change when their rows are rewritten.
distance_od = skims.od_skims[:, :, 0]
cost_od = skims.od_costs
turn_od = skims.od_turn_costs
```

`SkimmingContext(context, fields, centroid_count=None, *, include_costs=False,
include_turn_costs=False)` validates its fields once. Each field must be an
aligned, contiguous `float64` NumPy vector with `context.link_count` entries.
The arrays are **not copied**: the object retains them, marks them read-only,
and builds a `vector<const double *>` once. Cython reads them using
`const double[::1]` memoryviews. Other aliases to the same memory may still be
writable; do not use them to change the fields, make the inputs writable again,
or resize them while the prepared object is in use. `skims.fields` returns
read-only views of the retained arrays in input order.

The object also owns these C-contiguous output allocations:

- `od_skims`: `(z, z, field_count)`, ordered by origin, destination, field.
  These are link sums only, with no added penalties.
- `od_costs`: `(z, z)` routing costs **including** turn penalties, or `None`
  unless `include_costs=True`.
- `od_turn_costs`: `(z, z)` cumulative turn penalties only, or `None`
  unless `include_turn_costs=True`. These are already included in `od_costs`.

`centroid_count=None` uses every context node. Otherwise it must be an integer
between zero and `context.node_count`. Zero produces empty OD arrays. Empty
`fields` is supported, including cost-only and penalty-only preparation.
All output entries start at infinity, including diagonals, until their origin
is searched and skimmed. A processed origin has zero diagonal entries.

`results.skim_fields(skims)` checks the context and current origin, then writes
that origin's row in every requested output. It returns a read-only view of
`od_skims[origin]`. Do not pass `out` or `destination_count` with prepared inputs;
the prepared object already supplies both. A search must have run, and its
origin must be in `[0, z)`. Partial searches write infinity for unfinalized
destinations; skimming does not run or extend the search.

Only the current origin's row is overwritten. Other rows keep their last values,
so finish all origins before reading a complete iteration. Use `.copy()` to keep
a previous iteration. Read-only views pin their NumPy allocations after the
wrappers are deleted. The prepared object cannot be reinitialized.

`results.prepare_skims(skims)` prepares state scratch under the GIL and checks
that it does not overlap the inputs. `skim_fields(skims)` calls this automatically.
The check is cached for the prepared object and scratch allocation; repeated
calls do not rebuild or revalidate inputs, allocate a pointer table, or allocate
numeric output/scratch buffers. The result retains its most recently prepared
object. Switching field counts may replace scratch, but not the OD allocations.

Workers may share one `SkimmingContext`, but must use separate `SearchResults` and
workspaces and write **different origin rows**. Do not read a row while it is
being written. Input arrays remain constant for all workers and iterations.

### One-off skims

The original input-sequence and caller-owned output methods remain available.
After searching, `SearchResults` exposes three methods that release the GIL:

- `skim_fields(fields, out=None, *, destination_count=None)` sums additional link
  fields along each finalized path. `fields` is a sequence of contiguous
  `float64` vectors, each of length `context.link_count`, ordered exactly like
  `context.costs`. Read-only inputs are accepted. Output has shape
  `(D, len(fields))`, indexed by physical node then field.
- `skim_costs(out=None, *, destination_count=None)` copies costs **including turn
  penalties** from `distances`, projected through `terminal_states`. Output
  has shape `(D, 1)`.
- `skim_turn_costs(out=None, *, destination_count=None)` copies **only** turn
  penalties through the same projection, also shaped `(D, 1)`. These penalties
  are already included in `skim_costs`; do not add the two together.

`D` is `destination_count`, or `context.node_count` when it is `None`. Output
rows are the first `D` physical nodes. Pass `destination_count=z` when centroids
are nodes `0` to `z-1`. The count must be an integer between zero and
`context.node_count`. It is independent of the search's destination mask and
is not inferred from `out.shape`.

A finalized origin has zero skims if included, and unreachable/unfinalized nodes
have infinity. Before the first search all outputs are infinite. Skimming does
not search or extend a partial search: request every required destination when
building OD matrices. Every output entry is overwritten, including unreachable
entries. Zero destinations gives empty output; an empty sequence of fields gives
an `(D, 0)` array. State scratch still covers the full search, even with no output
rows.

All additional fields are plain additive attributes, **without penalties**.
Passing `context.costs` to `skim_fields` therefore sums link costs only; explicitly
call `skim_costs` for the routing objective. Negative attributes are allowed;
NaN and infinity propagate via ordinary floating-point addition. Attributes do
not change the selected routes or break ties.

`out` may be omitted to allocate an array, or supplied as a writable, aligned,
C-contiguous `float64` NumPy array with the exact output shape. It is returned
unchanged by identity. Neither input fields nor outputs support strided memory;
no implicit contiguous copies are made. For example:

```python
import numpy as np

N = turn_context.node_count
fields = [np.ones(turn_context.link_count), turn_context.costs]
cube = np.empty((N, N, len(fields)))  # origin, destination, field
cost_matrix = np.empty((N, N))
turn_matrix = np.empty((N, N))
results = turn_context.make_results()
for origin in range(N):
    dijkstra(turn_context, origin, None, results)
    results.skim_fields(fields, out=cube[origin])
    results.skim_costs(out=cost_matrix[origin].reshape(N, 1))
    results.skim_turn_costs(out=turn_matrix[origin].reshape(N, 1))

# Also available after the most recent field skim:
node_skims = cube[-1]                   # cumulative values at physical nodes
state_skims = results.workspace.state_skims  # read-only cumulative values at states
```

For a centroid OD matrix, with centroids at the first `z` nodes:

```python
z = 2  # Centroid count; must not exceed context.node_count.
cube = np.empty((z, z, len(fields)))
cost_matrix = np.empty((z, z))
turn_matrix = np.empty((z, z))
for origin in range(z):
    dijkstra(turn_context, origin, range(z), results)
    results.skim_fields(fields, out=cube[origin], destination_count=z)
    results.skim_costs(out=cost_matrix[origin].reshape(z, 1), destination_count=z)
    results.skim_turn_costs(out=turn_matrix[origin].reshape(z, 1), destination_count=z)
```

This writes directly into centroid-sized output without an all-node temporary
array or a separate copy. Paths still use intermediate network nodes. This does
not block paths through other centroids; centroid-flow blocking remains separate
and is not implemented here.

The general kernel first accumulates over the settled **state** tree in
parent-before-child order, then projects selected terminals to physical nodes.
This is essential for turn routing: a destination's optimal path may pass
through a nonterminal arrival at an intermediate node. A node-only accumulation
would silently produce incorrect skims. Runtime is
`O((state_count + D) * field_count)` and scratch space is
`O(state_count * field_count)`. The output count does not limit which states
are summed: paths to centroids may use states outside the output rows.
Cost-only and penalty-only skims copy existing labels in `O(D)` time with no
scratch allocation or summation.

### Workspace and Cython entry points

`results.workspace` is an `AoNWorkspace`, with a separate borrowed-pointer
C++ struct in `aon_workspace.hpp` and a NumPy-owning Cython wrapper in
`aon_workspace.pyx`. Its `prepare_skims(field_count)` method allocates under
the GIL, reusing the existing allocation when the field count is unchanged.
The Python `skim_fields` method calls this automatically. The workspace is
separate from finalized search results so future heap/routing scratch can be
added without changing the search-result contract.

`workspace.state_skims` has shape `(state_count, field_count)`. Root values are
zero; unfinalized states have infinity. This is scratch from the **last field
skim**, not necessarily the last search: searches and cost-only skims do not
refresh it. Retained read-only views pin the allocation. Same-width field skims
overwrite those views; resizing replaces the allocation without invalidating
old views. Use `.copy()` for snapshots. Inputs and output must not overlap the
active workspace scratch.

Cython callers can use the prepared object directly:

```cython
from aequilibrae.paths.cython.search_results cimport SearchResults
from aequilibrae.paths.cython.skimming_context cimport SkimmingContext

# results and skims must be typed SearchResults and SkimmingContext variables.
# Under the GIL, before the search/skim loop:
results.prepare_skims(skims)
# After a search from a centroid, with no change to this workspace's allocation:
with nogil:
    results.skim_prepared_nogil(skims)
```

This unchecked method uses the current origin to select an OD row. The caller
must ensure a compatible context, a completed search, `origin < centroid_count`,
prepared scratch, and exclusive access to that row and the worker's results.
It calls the same C++ field/cost/penalty kernels without allocating or checking
Python arrays. No field pointer table needs to be built in the calling loop.

The raw-pointer `cdef ... noexcept nogil` methods also remain available:

```cython
# Under the GIL, once per required width:
results.workspace.prepare_skims(field_count)
# fields: const double *const *; field_count contiguous arrays of link_count values
# destination_count: size_t; first D nodes, with 0 <= D <= context.node_count
# output: double *; D * field_count packed row-major entries
# costs, turns: double *; D entries each (packed [D, 1])
# Empty outputs may use NULL. This count does not change the search's target set.
with nogil:
    results.skim_fields_nogil(fields, field_count, destination_count, output)
    results.skim_costs_nogil(destination_count, costs)
    results.skim_turn_costs_nogil(destination_count, turns)
```

The caller must ensure correct sizes, allocation lifetimes, disjoint scratch,
and exclusive access to results/workspace, with no concurrent mutation of
input fields. Do not overwrite search/context buffers with outputs. The C++
kernels live in `skimming.hpp` and are templated on the floating-point numeric
type; the Cython interface currently specializes them for `double` only.
The kernels do not allocate, call Python, or branch on the routing mode.

## Single-origin network loading

`results.network_loading(demand, link_loads)` accumulates the current search's
flows into a **required caller-owned** buffer and returns it by identity:

- `demand`: aligned, C-contiguous `float64` NumPy array shaped `[D, classes]`,
  normally `matrix[origin]` from an `[O, D, classes]` cube. Read-only inputs are
  accepted, without copying. Rows are physical nodes `0` through `D-1`, with
  `0 <= D <= context.node_count`; intermediate states need not be in this range.
- `link_loads`: writable, aligned, C-contiguous `float64` NumPy array shaped
  `[context.link_count, classes]`, in context-local directed-link order. This
  buffer is **accumulated into, never cleared or allocated by loading**.

Unreachable/unfinalized destinations and intrazonal demand are ignored. Loading
neither runs nor extends a search: request every destination whose demand you
want loaded. Before a search it loads nothing. Empty destination/class axes and
edgeless contexts are supported. Negative/NaN/infinite demand uses ordinary
floating-point addition along the selected paths; no turn penalties are added
to demand. Inputs, output and active loading scratch must not overlap. Do not
force internal search/context/workspace arrays writable or overwrite them.

Each worker keeps its own `SearchResults` and `AoNWorkspace`. An iteration-level
caller allocates `[threads, links, classes]` once, hands each worker its slice,
and reduces after all workers finish:

```python
thread_loads = np.zeros((threads, context.link_count, classes), dtype=np.float64)
workers = [context.make_results() for _ in range(threads)]
for results in workers:
    results.workspace.prepare_loading(classes)  # Optional up-front scratch allocation.

# Each worker executes this with its own tid and assigned origins:
# for origin in assigned_origins:
#     dijkstra(context, origin, range(D), workers[tid])
#     workers[tid].network_loading(matrix[origin], thread_loads[tid])

# Only after all workers have finished:
link_loads = thread_loads.sum(axis=0)
thread_loads.fill(0)  # Before starting the next iteration.
```

No thread index is passed to the kernel, and no shared mutable workspace or
atomics are needed. The caller owns the output allocation and its lifetime;
workspaces do not retain it. Do not read or clear an active worker's slice.
Input demand may be shared but must not change while loading.

`workspace.prepare_loading(class_count)` allocates `[state_count, class_count]`
cascade scratch under the GIL, reusing it at the same width. Python loading
calls this automatically. `workspace.state_loads` is `None` before preparation,
otherwise a read-only, zero-copy view. Every loading call resets this scratch
and seeds demand at `terminal_states[d]`, excluding the root. It then traverses
`reached_first[:settled_count]` in reverse, adding each state's demand to its
connector link and parent state. This handles nonterminal arrival histories in
turn routing without a mode branch. At completion each state holds its subtree's
demand; the root holds total reachable non-intrazonal demand, and unfinalized
states are zero. Skims and searches do not refresh loading scratch. Retained
views pin allocations; same-width loading overwrites them, resizing preserves
old views. Use `.copy()` for snapshots.

The allocation-free kernel lives in `network_loading.hpp`, templated on the
floating-point type. Cython currently specializes it for `double`, like skimming.
Runtime is `O((state_count + D) * classes)`, with `O(state_count * classes)` scratch.
Cython callers can avoid Python validation in the origin loop:

```cython
# results must be typed SearchResults. Prepare once under the GIL:
results.workspace.prepare_loading(class_count)
# demand: const double *, packed [D, class_count]
# link_loads: double *, packed [context.link_count, class_count]
with nogil:
    results.network_loading_nogil(demand, D, class_count, link_loads)
```

The unchecked caller must guarantee the shapes above, prepared scratch, buffer
lifetimes/non-overlap, and exclusive access to results/workspace/output. Zero
classes permits null pointers; zero destinations permits null demand; zero
links permits null output. No select-link loading or iteration scheduler is
implemented here.

## Turn representation

`turn_fs` has `link_count + 1` entries and indexes explicit turns by incoming
link. `turn_to_links` must be strictly increasing within each row, without
duplicates. Each explicit pair must connect consecutive directed links.
`turn_penalties` contains nonnegative costs, or positive infinity for a ban.
NaN and negative values are rejected.

Missing turns have zero penalty. With `allow_uturns=False`, a transition back to
the incoming link's tail is banned by default. Explicit entries take precedence:
an explicit finite penalty (including zero) can permit that U-turn, while an
explicit infinity prohibits it regardless of the default policy. First links
leaving the source incur no turn cost.

All three turn arrays can be omitted for an empty explicit turn table. These are
local directed-link indices, not external IDs or raw node triples; translating
user node triples to link pairs remains outside this MVP.

## Common state-tree contract

|                    |NodeBasedContext|TurnBasedContext                      |
|--------------------|----------------|--------------------------------------|
|State               |Physical node   |Incoming directed link                |
|`state_count`       |`node_count`    |`link_count + 1`                      |
|Root state          |Origin node     |Virtual source at `link_count`        |
|`terminal_states[d]`|`d`, if settled |First settled state ending at node `d`|

Both contexts populate the same `SearchResults`:

- `predecessors[s]`: parent **state**.
- `connectors[s]`: local link traversed from the parent into state `s`.
- `distances[s]`: routing cost, including penalties, to finalized state `s`.
- `turn_costs[s]`: cumulative turn penalty to finalized state `s`; zero for
  settled node-based states.
- `reached_first[:settled_count]`: finalized states in parent-before-child order,
  including the root. In turn routing, several states may end at the same node.
- `terminal_states[d]`: physical-node-to-selected-state projection. The origin
  maps to the root. Unfinalized/unreachable destinations have no terminal.
- `destination_mask[d]`: nonzero when physical node `d` was requested.
- `destination_count` and `reached_destination_count`: requested and settled
  target counts, allowing an O(1) all-targets-reached check.

Unfinalized states have sentinel predecessors/connectors and infinite cost
labels. The sentinel is `np.iinfo(np.uintp).max`. The root has no predecessor or
connector and has zero costs. The last requested node's first settled arrival
ends the search; tentative paths are cleared rather than exposed as finalized
paths.

Path reconstruction follows `terminal_states[d]` through state predecessors to
`root`. It has no node-versus-turn branch. Physical nodes are obtained from the
traversed links. The `path_nodes_to(d)`, `path_links_to(d)`, `path_cost_to(d)` and
`path_turn_cost_to(d)` accessors inspect a particular finalized node. An
unreachable node has empty paths and infinite path costs. An origin-to-itself
search returns a one-node path, no links, and zero costs.

## Ownership and implementation

The shared Cython `GraphContext` parent owns and validates the initialized
NumPy graph arrays; the node- and turn-based subclasses add their respective
embedded C++ structs and turn buffers. C++ structs borrow these pointers, and
results retain their context. Graph inputs are copied into contiguous snapshots
and exposed read-only. Prepared skim fields are instead retained without copying
and marked read-only.
Index arrays use `np.uintp` (`size_t`), and costs use `np.float64`.

Retained views pin their NumPy allocations after wrappers are deleted. Reusing a
result overwrites those same buffers; use `.copy()` for a snapshot. Do not
reinitialize contexts while results refer to them, resize the underlying buffers,
or force them writable.

Allocation/validation runs under the GIL, and searching releases the GIL. Share
contexts across workers, but give each worker separate results. Access to the
same results must be externally serialized, including reads through retained
array views. Both kernels still allocate/free their own four-ary heap per call.

Cython's `RoutingContext` fused type specializes only the Dijkstra entry point;
`SearchResults.context` is an ordinary object reference, not a fused member.
C++ overloads accept concrete contexts and populate a common results struct.
The node kernel explores physical-node states directly, while the turn kernel
explores the implicit incoming-link state space. Both use the existing heap and
neither calls Python in its search loop or materializes additional graph edges.

The target set uses one byte per physical node rather than `std::vector<bool>`.
Dense node indices make this an allocation-free O(1) membership test in the hot
settlement loop, and the O(node_count) target count is folded into result-array
initialization that every search already performs. An all-zero mask means that
no early-exit condition is active. A sorted sparse target list
would require O(log targets) checks; a hash set would add allocation and poor
locality. The mask is immutable during search. A monotonic reached-target count
is exposed in the result instead of destructively clearing mask bits or exposing
a decrementing implementation counter.

No select-link loading, centroid blocking, graph compression, A*, or heap
selection is implemented here. The persistent workspace holds skim and cascade
loading scratch; heaps are still allocated by each search.

## Validation

```sh
meson compile -C build
ASAN_OPTIONS=detect_leaks=1 \
LSAN_OPTIONS=suppressions=$(readlink -f .github/workflows/asan_suppressions.txt) \
LD_PRELOAD=$(gcc -print-file-name=libasan.so) \
python -m pytest --durations=50 --color=yes \
    tests/aequilibrae/paths/test_node_routing_mvp.py \
    tests/aequilibrae/paths/test_turn_routing_mvp.py \
    tests/aequilibrae/paths/test_context_skimming.py \
    tests/aequilibrae/paths/test_context_network_loading.py
```

Tests cover both state layouts, different arrival histories, turn penalties and
bans, U-turn overrides, parallel links, zero-cost cycles, partial searches,
reused buffers, view lifetimes, and shared-context concurrent searches. Random
multigraph searches are checked against an independently expanded NetworkX
state graph. Skimming tests cover all OD pairs, arbitrary/zero field counts,
nonterminal arrival histories, explicit routing-cost and penalty projection,
unreachable/unfinalized states, workspace reuse and retained views, buffer
validation, shared-context workers, and random multigraph path-sum comparisons.
Prepared skim tests check no-copy inputs, read-only views, OD buffer reuse,
optional cost/penalty outputs, cached preparation, input lifetimes, and workers
sharing one prepared object while writing separate origin rows. Loading tests
cover both state layouts, turn arrival histories, partial/unreachable paths,
centroid-sized demand, empty axes, scratch reuse/lifetimes, buffer validation,
random multigraph comparisons against path walks, and worker-local accumulation
with thread-axis reduction and iteration resets.
