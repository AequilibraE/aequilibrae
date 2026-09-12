.. _turn_aware_path_computation:

Turn-aware path computation
===========================

Classic shortest-path algorithms treat a road network as a graph whose states are *nodes*: the
cost of arriving at a node is all that a label needs to carry, because what a path may do next
depends only on where it is. Real road networks violate that assumption. A left turn may be
banned, a movement through a signalised intersection may cost twenty seconds, and a U-turn may be
illegal everywhere except at one designated median opening. Whether a movement is allowed, and
what it costs, depends on **how the path arrived** — not just on where it is.

AequilibraE supports this through *turn restrictions*: directed movement triples
``from_node -> via_node -> to_node``, each carrying a penalty in the same unit as the graph's cost
field, with ``+inf`` (or ``None``/``NaN``) denoting a prohibited movement.

.. code-block:: python

    >>> turns = pd.DataFrame(
    ...     [
    ...         {"from_node": 12, "via_node": 13, "to_node": 27, "penalty": np.inf},  # banned
    ...         {"from_node": 12, "via_node": 13, "to_node": 41, "penalty": 15.0},    # 15 s delay
    ...     ]
    ... )
    >>> graph.set_turn_restrictions(turns, allow_path_uturns=False)  # doctest: +SKIP

This page describes the algorithm AequilibraE uses to honour those restrictions, why it is
correct, and what it costs.

The state-space problem
-----------------------

The textbook way to handle turn costs is to change what a label represents. Instead of labelling
nodes, label **arcs** (directed links): a state is "the path has just traversed arc *a*", and the
transition from *a* to *a'* — legal only when ``head(a) == tail(a')`` — costs
:math:`c(a') + p(a, a')`, where :math:`p` is the turn penalty. This is the *line graph* (or
*dual graph*) construction, introduced for exactly this purpose by Caldwell (1961) and developed
by Kirby & Potts (1969); Winter (2002) gives the modern treatment. Dijkstra's algorithm
(Dijkstra, 1959) runs unchanged on that expanded graph, and the result is exact.

It is also expensive. The number of labels goes from :math:`|V|` to :math:`|A|`, and on a dense
urban network the arcs-per-node ratio is typically between 3 and 4. Every heap operation, every
label array, and every scan grows by that factor — even when the network carries a single turn
restriction, or none that actually bind.

That cliff is not hypothetical. On the Arkansas statewide model, installing **one** turn entry
with a penalty of ``0.0`` — semantically a no-op, it changes no path and no cost — made skimming
**4.34x** slower, purely because the presence of a turn table switched the engine from the
node-based kernel to the arc-based one. The ratio tracked the network's arcs-per-node ratio of
3.96 almost exactly.

The hybrid node/arc-state kernel
--------------------------------

The observation that removes the cliff is that **arc state is only informative where a movement
control exists**. If no turn entry is keyed on any arc entering node *v*, and no neighbour of *v*
carries one either, then a path arriving at *v* can do exactly the same things regardless of how
it got there — so one label per node suffices.

AequilibraE therefore partitions the nodes. Let

* :math:`R \subseteq V` be the set of **via nodes**: nodes at which at least one explicit turn
  entry is keyed on an incoming arc, and
* :math:`S = R \cup N(R)`, where :math:`N(R)` is the set of nodes adjacent to :math:`R` (both in-
  and out-neighbours).

Nodes in :math:`S` are **stateful**; every other node is **plain**. The kernel
(:func:`path_finding_hybrid` in ``basic_path_finding.pyx``) keeps labels in the arc index space,
so the heap is sized exactly as the arc-based kernel's, but:

* at a **stateful** node, each incoming arc keeps its own label — full arc-based behaviour;
* at a **plain** node, every incoming arc collapses onto that node's *representative arc*
  ``rep_arc[v]``, so the node holds one live label no matter how many arcs enter it. The arc
  actually taken is recorded in ``connectors[v]`` by the winning relaxation, which is what path
  reconstruction and the U-turn test read back.

The turn-table scan is likewise skipped at plain nodes: the ``turn_fs`` slice is taken only when
``stateful[cur_node]`` is set, so an unrestricted intersection costs exactly what it costs in the
node-based kernel.

The consequence is that the engine degrades gracefully rather than discontinuously. With no turn
restrictions at all, :math:`S = \emptyset` and the kernel *is* node-based Dijkstra. With a handful
of restricted intersections, :math:`|S| \ll |V|` and the extra work is proportional to the number
of restrictions, not to the size of the network. Only when a large fraction of intersections
carry controls does the cost approach that of the full arc-based kernel — which is the price the
model has genuinely asked for.

Why the collapse is safe
------------------------

Collapsing labels discards information, so it needs an argument. The only thing a plain node's
collapsed label can forbid that the full arc-based kernel would allow is a **U-turn back along an
arrival arc that was not the cheapest one** — because every other movement at a plain node is
unrestricted and free, and therefore available from any label.

So the claim to establish is:

    When link costs and turn penalties are non-negative, an optimal walk never *needs* to make a
    U-turn at a node whose entire neighbourhood is free of explicit movement controls.

**Argument.** Let :math:`W` be an optimal origin-destination walk that traverses arc :math:`a`
into node :math:`v \notin S` and immediately returns along :math:`\mathrm{rev}(a)` to
:math:`u = \mathrm{tail}(a)`. Excise the pair :math:`(a, \mathrm{rev}(a))` from :math:`W`. The
result :math:`W'` is still a walk from the same origin to the same destination — it simply stays
at :math:`u` and continues from there.

*Feasibility.* The excision creates one new movement at :math:`u`, from the arc :math:`b` that
entered :math:`u` to the arc :math:`b'` that leaves it. Because :math:`v \notin S` we know
:math:`v \notin N(R)`, hence :math:`u \notin R`, hence **no explicit prohibition or penalty is
keyed at** :math:`u`. The only way the new movement can be illegal is if it is itself a U-turn
(:math:`b' = \mathrm{rev}(b)`), in which case the same excision applies at :math:`u`. Each
excision removes two arcs from a finite walk, so the process terminates in a feasible walk.

*Optimality.* Each excision removes :math:`c(a) + c(\mathrm{rev}(a))` plus the two turn penalties
it spanned, all of which are non-negative, and introduces one movement at an uncontrolled node,
whose penalty is zero. Therefore :math:`\mathrm{cost}(W') \le \mathrm{cost}(W)`.

Hence an optimal walk exists that makes no U-turn at any plain node, and the collapsed label — which
retains the cheapest arrival and its arc — loses nothing. Note where the definition of :math:`S`
earns its keep: it is precisely the inclusion of :math:`N(R)` that guarantees the splice target
:math:`u` carries no explicit control. Restricting :math:`S` to :math:`R` alone would break the
feasibility half of the argument.

Two corollaries are worth stating explicitly:

* When U-turns are permitted globally (``allow_path_uturns=True``), nothing is forbidden at a plain
  node at all, so the collapse is trivially lossless.
* At a stateful node no collapse happens, so restricted intersections retain exact arc-based
  semantics — including the case where an optimal path genuinely must U-turn because a prohibition
  leaves it no alternative.

U-turn semantics on a compressed graph
--------------------------------------

Path computation for assignment and skimming runs on the **compressed** graph, where chains of
degree-two nodes have been contracted into single arcs (see :ref:`aequilibrae-graphs`). That
creates a subtlety for U-turns: a U-turn is a reversal at a *physical* node, but a compressed arc
may span many physical nodes, so the arc's own end nodes are the wrong thing to compare.

AequilibraE therefore carries **boundary contexts** with each compressed arc, built per direction
during compression:

* ``_compact_first_node[a]`` — the first physical node the arc enters after leaving its tail;
* ``_compact_last_node[a]`` — the last physical node the arc occupies before reaching its head.

Leaving node *v* along arc *a'* is a reversal of the arrival arc *a* exactly when
``first_ctx[a'] == last_ctx[a]``, which is the test the kernel applies. Comparing compact end
nodes instead would misclassify both ways: it would miss reversals into a different shortcut that
retraces the same physical link, and it would flag legitimate through movements at a node where
two shortcuts happen to share an endpoint.

Two further rules complete the semantics:

* **An explicit turn entry overrides the global U-turn ban.** The kernel skips the U-turn test
  when the movement has an explicit entry in the turn table (``has_explicit_entry``), so a
  modeller can permit — and price — one legal U-turn at a designated median opening while U-turns
  remain banned everywhere else. A finite penalty on such a movement is honoured; ``+inf`` still
  prohibits it.
* **Effective via nodes are protected from contraction.** A turn restriction has nowhere to attach
  if its via node has been absorbed into a shortcut, so contraction preserves the via nodes that
  actually map onto arcs of the graph. Restrictions naming nodes or legs that do not exist in the
  graph ("phantom" restrictions, common when a restriction table is shared across traffic classes
  whose mode does not serve every link) are filtered out first: they neither protect nodes nor
  perturb the graph topology, so a class that cannot use a restricted movement pays nothing for
  its existence.

Centroid-flow blocking is handled differently from the classic node-based kernel, which patches
the b-node array to sever outgoing edges at centroids. The hybrid kernel reads the *unpatched*
b-nodes and enforces blocking itself, by refusing to expand out of any centroid that is not the
origin of the search. Blocking is therefore a run-time flag: it is never baked into the graph
topology or into the turn table, which is why toggling it invalidates cached results but requires
no rebuild.

Complexity
----------

Let :math:`d` be the average out-degree and :math:`t` the average number of explicit turn entries
per restricted arc.

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * - Kernel
     - Distinct labels
     - Heap capacity
     - Per-relaxation work
   * - Node-based
     - :math:`|V|`
     - :math:`|V|`
     - :math:`O(1)`
   * - Arc-based (line graph)
     - :math:`|A|`
     - :math:`|A|`
     - :math:`O(t)`
   * - Hybrid
     - :math:`|V| + \sum_{v \in S} (\deg^-(v) - 1)`
     - :math:`|A|`
     - :math:`O(t)` at stateful nodes, :math:`O(1)` elsewhere

The hybrid kernel allocates the arc-sized heap unconditionally — the label index space is the arc
index space — but the number of labels that are ever *live* is what drives the
:math:`O(m + n \log n)` term, and that is :math:`|V|` plus one extra label per additional incoming
arc at a stateful node. The heap itself is a 4-ary heap (Johnson, 1975), which is the standard
choice for the arc-density of road networks.

Measured performance
--------------------

The figures below were measured on the Arkansas statewide model (approximately 265,000 directed
graph links) against the reference implementation, with the same compact graph, the same cost
field and identical results.

.. list-table::
   :header-rows: 1
   :widths: 46 18 18 18

   * - Workload
     - Reference
     - Turn-aware
     - Speed-up
   * - BFW assignment, 50 iterations, 16 threads
     - 404.44 s
     - 175.20 s
     - 2.31x

The comparison was run with the turn-aware branch configured exactly as the reference is — one
label space, the same compressed graph — so the difference is attributable to the path-finding
change set rather than to any difference in graph preparation.

.. note::

    Both sides of that comparison include a fix to ``Graph.set_skimming`` that was a far larger
    effect than anything measured here: testing skim field membership against a
    ``DataFrameGroupBy`` object iterates its *groups*, which on Arkansas materialised 264,562
    DataFrame slices (72 s) on every call and always answered "missing", so the cached groupby was
    never reused. Graph preparation fell from 79.7 s to 6.4 s once that was corrected. Benchmarks
    taken before that fix are not comparable with these.

Limitations
-----------

* Turn penalties are expressed in the unit of the cost field, and AequilibraE does not convert
  between units. A penalty table built for a travel-time field is meaningless against a distance
  field; keeping them consistent is the modeller's responsibility.
* Which skim fields accumulate turn penalties is controlled by ``Graph.turn_skim_fields``. When it
  is left empty, penalties accumulate into the cost field's skim only.
* Turn penalties must be non-negative. The kernel relies on non-negativity both for Dijkstra's own
  correctness and for the excision argument above.

References
----------

* Caldwell, T. (1961). On finding minimum routes in a network with turn penalties.
  *Communications of the ACM*, 4(2), 107-108.
* Delling, D., Goldberg, A. V., Pajor, T., & Werneck, R. F. (2017). Customizable route planning in
  road networks. *Transportation Science*, 51(2), 566-591.
* Dijkstra, E. W. (1959). A note on two problems in connexion with graphs.
  *Numerische Mathematik*, 1, 269-271.
* Geisberger, R., Sanders, P., Schultes, D., & Delling, D. (2008). Contraction hierarchies: faster
  and simpler hierarchical routing in road networks. *Experimental Algorithms (WEA 2008)*, LNCS
  5038, 319-333.
* Geisberger, R., & Vetter, C. (2011). Efficient routing in road networks with turn costs.
  *Experimental Algorithms (SEA 2011)*, LNCS 6630, 100-111.
* Gutierrez, E., & Medaglia, A. L. (2008). Labeling algorithm for the shortest path problem with
  turn prohibitions with application to large-scale road networks. *Annals of Operations
  Research*, 157(1), 169-182.
* Johnson, D. B. (1975). Priority queues with update and finding minimum spanning trees.
  *Information Processing Letters*, 4(3), 53-57.
* Kirby, R. F., & Potts, R. B. (1969). The minimum route problem for networks with turn penalties
  and prohibitions. *Transportation Research*, 3(3), 397-408.
* Villeneuve, D., & Desaulniers, G. (2005). The shortest path problem with forbidden paths.
  *European Journal of Operational Research*, 165(1), 97-107.
* Winter, S. (2002). Modeling costs of turns in route planning. *GeoInformatica*, 6(4), 345-361.

.. seealso::

    * :func:`aequilibrae.paths.graph.Graph.set_turn_restrictions`
        Class documentation
    * :ref:`aequilibrae-graphs`
        Graph compression, which turn restrictions interact with
