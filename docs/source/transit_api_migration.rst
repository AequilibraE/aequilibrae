Migrating transit models to AequilibraE 2.0
===========================================

AequilibraE 2.0 repairs a series of defects in public transport graph building, assignment and
skimming. The workflow keeps its shape (import a GTFS feed, build a graph per period, assign and
skim), but several results change and some of the API changes break existing code:

* graph edges now run from ``a_node`` to ``b_node``, as road links do;
* saved graphs hold physical attributes only, and each ``TransitClass`` turns them into the costs it
  assigns with, through a ``TransitCosting``;
* skims report true waits and transfers, and ``inf`` where no strategy exists;
* periods count seconds from the start of the GTFS service day, and trips belong to a period by a
  single trip-level rule shared with ``build_pt_preload``.

This page lists every change and how to adapt code written for AequilibraE 1.x. The results that
change are listed under "Changes in Behaviour" in :doc:`aequilibrae2_0`.

Upgrading existing projects
---------------------------

Copy the project folder before upgrading: database downgrades are not supported, and the copy is
the only way back. Then upgrade it as usual:

.. code-block:: python

    project.upgrade()

``project.upgrade()`` now migrates the root scenario and every other scenario of the project, each
with its own ``project_database.sqlite`` and ``public_transport.sqlite``.

Two migrations reach transit models:

* **Transit migration 007** drops every saved transit graph (the ``nodes`` and ``links`` tables of
  ``public_transport.sqlite``, and their rows in ``transit_graph_configs``) and recreates ``links``
  with the new columns. Saved graphs cannot be converted, because the costs they were built with
  were never saved. The GTFS supply tables (stops, routes, trips, schedules, fares, map matching) are
  untouched. Each dropped graph's configuration is logged as a warning, in JSON, so it can be
  rebuilt with the same settings.
* **Network migration 010** extends the default period (``period_id = 1``) to the whole service
  day, from ``0`` to ``172800`` seconds (``48:00:00``).

A dropped graph is rebuilt from its logged configuration. Configurations saved by 1.x name
``EPSG:3857`` as their ``projected_crs``; leave it out to use the new default (see below):

.. code-block:: python

    import json

    from aequilibrae.transit import Transit

    # The JSON that transit migration 007 logged for period 1
    config = json.loads('{"period_id": 1, "projected_crs": "EPSG:3857", "with_walking_edges": false}')
    config.pop("projected_crs")

    transit = Transit(project)
    builder = transit.create_graph(**config)
    transit.save_graphs()

``Transit.load()`` skips configurations whose graph has no saved nodes and links, since the two
databases of a scenario are upgraded one after the other.

Edge orientation
----------------

``a_node`` is now the tail of every edge and ``b_node`` its head, on ``TransitGraphBuilder.edges``,
on the ``links`` table of ``public_transport.sqlite`` and on ``TransitGraph``. This matches road
links. In 1.x, ``a_node`` was the head, so ``create_line_geometry`` drew every edge backwards and
``connector project match`` routed connectors backwards on one-way links.

Code that reads edges directly must swap the two columns:

.. code-block:: python

    # AequilibraE 1.x: a_node was the head of the edge
    boardings = builder.edges.query("link_type == 'boarding'")
    stops, boarding_vertices = boardings.b_node, boardings.a_node

    # AequilibraE 2.0: a_node is the tail
    stops, boarding_vertices = boardings.a_node, boardings.b_node

Vertex geometries are also looked up by ``node_id`` now, so zones whose ``zone_id`` values have gaps,
do not start at ``1`` or are not stored in order build correctly.

Physical graphs and class costing
---------------------------------

Graphs no longer store costs. ``TransitGraphBuilder.edges`` and the saved ``links`` table hold
unweighted physical attributes, and the ``trav_time`` and ``freq`` columns are gone from both:

+-------------------+--------------------------------------------------------------------------------+
| Column            | Meaning                                                                        |
+===================+================================================================================+
| ``run_time``      | Mean scheduled run time of an on-board edge, in seconds                        |
+-------------------+--------------------------------------------------------------------------------+
| ``dwell_time``    | Mean scheduled dwell at the stop of a dwell, boarding or alighting edge        |
+-------------------+--------------------------------------------------------------------------------+
| ``walk_distance`` | Geodesic length of a connector or of a walk between stops, in metres           |
+-------------------+--------------------------------------------------------------------------------+
| ``walk_time``     | ``walk_distance`` over the builder's ``walking_speed``                         |
+-------------------+--------------------------------------------------------------------------------+
| ``headway``       | Scheduled headway between a line's departures, on boarding edges               |
+-------------------+--------------------------------------------------------------------------------+
| ``trip_count``    | Trips of the line in the period, on boarding edges                             |
+-------------------+--------------------------------------------------------------------------------+
| ``line_freq``     | Distinct departures over the period length on boarding edges, ``inf``          |
|                   | elsewhere                                                                      |
+-------------------+--------------------------------------------------------------------------------+
| ``capacity``      | Capacity of all the period's trips, on on-board edges                          |
+-------------------+--------------------------------------------------------------------------------+
| ``segment_id``    | Physical element of the edge: ``pattern_id:seq`` for line edges, the stop for  |
|                   | connectors and ``from_stop:to_stop`` for walks between stops                   |
+-------------------+--------------------------------------------------------------------------------+

The ``o_line_id``, ``d_line_id``, ``transfer_id`` and ``distance`` columns of ``links`` are gone too.

How a class perceives the graph is a ``TransitCosting`` passed to ``TransitClass``. The class
evaluates it into edge costs and frequencies when it is created and before each assignment, and
writes them to the ``trav_time`` and ``freq`` columns of the in-memory ``TransitGraph``, so
``set_time_field("trav_time")`` and ``set_frequency_field("freq")`` keep working. Classes that share
a graph each keep their own costing.

.. code-block:: python

    # AequilibraE 1.x: costs were attributes of the builder, baked into the saved graph
    builder = transit.create_graph(with_walking_edges=False)
    builder.wait_time_factor = 0.5
    builder.alighting_penalty = 300
    transit_graph = builder.to_transit_graph()
    pt = TransitClass("pt", transit_graph, demand)

    # AequilibraE 2.0: costs belong to the class
    from aequilibrae.paths import TransitClass, TransitCosting

    builder = transit.create_graph(with_walking_edges=False)
    transit_graph = builder.to_transit_graph()
    pt = TransitClass("pt", transit_graph, demand, costing=TransitCosting(headway_fraction=0.5, alighting_penalty=300))

The builder attributes map onto ``TransitCosting`` as follows:

+--------------------------------------+------------------------------+-------------------------------+
| ``TransitGraphBuilder`` attribute    | ``TransitCosting`` field     | Default                       |
+======================================+==============================+===============================+
| ``wait_time_factor``                 | ``headway_fraction``         | ``0.5`` (was ``1.0``)         |
+--------------------------------------+------------------------------+-------------------------------+
|                                      | ``wait_weight``              | ``1.0`` (new)                 |
+--------------------------------------+------------------------------+-------------------------------+
| ``alighting_penalty``                | ``alighting_penalty``        | ``480``                       |
+--------------------------------------+------------------------------+-------------------------------+
| half of ``uniform_dwell_time`` on    | ``boarding_dwell_share``,    | ``0.5`` of the stop's         |
| boarding and alighting edges         | ``alighting_dwell_share``    | scheduled dwell               |
+--------------------------------------+------------------------------+-------------------------------+
| ``access_time_factor``               | ``access_weight``            | ``1.0``                       |
+--------------------------------------+------------------------------+-------------------------------+
| ``egress_time_factor``               | ``egress_weight``            | ``1.0``                       |
+--------------------------------------+------------------------------+-------------------------------+
| ``walk_time_factor``                 | ``walk_weight``              | ``1.0``                       |
+--------------------------------------+------------------------------+-------------------------------+

The defaults reproduce the 1.x costing except the headway fraction. ``wait_time_factor = 1.0`` made
the expected wait a full headway, which assumes random arrivals at a regular service; the new
default is half a headway. ``headway_fraction`` scales the reported waits, while ``wait_weight`` only
scales how much the class dislikes waiting, and never enters the reported waits.

Until 2.1, setting the old attributes on a builder still works and raises a
``DeprecationWarning``: classes on that builder's graphs then default to the corresponding costing.
``uniform_dwell_time`` and ``a_tiny_time_duration`` are retired and setting them has no effect:

* dwell edges carry each stop's scheduled dwell (``departure - arrival``) instead of a uniform 30 s,
  so dwell is no longer counted twice for feeds whose stop-to-stop times already include it;
* the assignment raises every edge cost to at least ``1e-9`` s itself, which is what the tiny
  durations used to guarantee. Negative costs now raise an error.

``walking_speed`` stays on the builder, as a constructor argument, because it shapes the physical
``walk_time``.

Transfers between stops
-----------------------

Transfers alight to the stop vertex and board again from it, so every boarding uses a boarding edge
and boardings are counted correctly:

* ``with_inner_stop_transfers`` is deprecated and has no effect: a transfer at a stop already runs
  ``alighting → stop → boarding``;
* ``with_outer_stop_transfers`` creates ``outer_transfer`` edges from one stop to another stop of
  the same parent station, with infinite frequency, instead of edges from an alighting vertex to a
  boarding vertex with the boarded line's frequency;
* ``with_walking_edges`` creates ``walking`` edges between the same stop pairs. A pair gets one edge,
  an outer transfer when both options are set. A feed without parent stations builds without either
  kind of edge, which used to fail with the default ``with_walking_edges=True``.

Connectors and distances
------------------------

``projected_crs`` now defaults to the UTM zone of the centroid of the stops, rather than
``EPSG:3857``. It is only used to search connector candidates: every reported distance (connectors,
walking edges, outer transfers and connectors matched to the road network) is geodesic. Web Mercator
stretched connector distances by :math:`1 / \cos(\text{latitude})`, about 15% at 30° and 50% or more
at 50°. The CRS actually used is saved with the graph's configuration.

Every zone now gets a connector to at least its nearest stop, in addition to each stop's nearest
zone. ``distance_upper_bound`` filters connectors under both connector methods, and a zone with no
stop within the bound is logged and left without connectors. ``max_connectors_per_zone`` keeps each
zone's nearest stops.

Periods and trips
-----------------

Times count seconds from the start of the GTFS service day, so a period may run past ``86400``
(a ``24:00–27:00`` late-night period is valid), and periods never wrap around midnight. The default
period is ``[0, 172800)``, so it keeps the trips that run past midnight, which 1.x dropped.

Trips belong to a period by one trip-level rule, the new ``trip_inclusion`` argument of
``TransitGraphBuilder`` (default ``"start"``), which is the same rule as ``inclusion_cond`` of
``Transit.build_pt_preload``:

* ``"start"``, ``"end"`` and ``"midpoint"`` use the trip's first departure, last arrival or their
  mean, and assign each trip to exactly one of a set of consecutive periods;
* ``"any"`` includes a trip with any stop event inside the period.

Windows are half-open, ``[start, end)``, so a trip leaving exactly at the end of one period belongs
to the next. A graph then takes complete trips: run times, dwell, trip counts and headways come from
the trips the period includes, rather than from the stop events inside the window.

Each line's frequency is its distinct departures in the period over the period's length, capped at
``86400`` s, so the whole-day default period gives daily averages. The scheduled headway, the spread
of departures over the gaps between them, is kept on boarding edges for reference. Trips that
duplicate another trip's departure add capacity but not frequency.

``build_pt_preload`` also accepts a period:

.. code-block:: python

    preload = transit.build_pt_preload(period_id=2)

Skims
-----

Waits and transfers are computed inside the assignment kernel rather than by subtraction. The
predefined skims mean:

+----------------------------+------------------------------------------------------------------------+
| Skim                       | Meaning                                                                |
+============================+========================================================================+
| ``trav_time``              | Expected perceived cost of the strategy, which includes the class's    |
|                            | weights and penalties                                                  |
+----------------------------+------------------------------------------------------------------------+
| ``waiting_time``           | Expected wait: the headway fraction over the combined frequency of the |
|                            | lines boarded at each stop, not weighted by ``wait_weight``. In 1.x it |
|                            | also held alighting penalties, dwell and walking times                 |
+----------------------------+------------------------------------------------------------------------+
| ``first_waiting_time``     | Expected wait at the first boarding (new)                              |
+----------------------------+------------------------------------------------------------------------+
| ``transfer_waiting_time``  | ``waiting_time`` minus ``first_waiting_time`` (new)                    |
+----------------------------+------------------------------------------------------------------------+
| ``pt_probability``         | Probability that the strategy boards a vehicle at all (new)            |
+----------------------------+------------------------------------------------------------------------+
| ``transfers``              | Expected boardings minus ``pt_probability``. In 1.x it counted         |
|                            | transfer edges, and was zero with the default settings                 |
+----------------------------+------------------------------------------------------------------------+
| ``boardings``,             | Expected traversals of boarding and alighting edges                    |
| ``alightings``             |                                                                        |
+----------------------------+------------------------------------------------------------------------+
| ``inner_transfers``,       | Expected traversals of ``inner_transfer`` and ``outer_transfer``       |
| ``outer_transfers``        | edges. Built graphs no longer have inner transfer edges, so prefer     |
|                            | ``transfers``                                                          |
+----------------------------+------------------------------------------------------------------------+
| ``in_vehicle_trav_time``,  | Physical times on the edges of each type, from ``run_time``,           |
| ``on_board_trav_time``,    | ``dwell_time`` and ``walk_time``, never weighted                       |
| ``dwelling_time``, ...     |                                                                        |
+----------------------------+------------------------------------------------------------------------+

Pairs that no strategy connects skim as ``inf`` instead of ``0.0``, in every skim, so mode choice
cannot mistake them for free trips. Demand between them is not loaded: it is reported in the
results' ``unassigned_demand`` (total) and ``unassigned_od`` (per zone pair).

Skims and demand follow the graph's centroid order. In 1.x, when the ``zones`` table was not stored
in ``zone_id`` order, demand from one zone was loaded at another zone's vertices.

The skimming example's ``freq`` skim, which summed the ``1e20`` that stood for infinite frequency,
is gone from the documentation: infinite frequencies are now exactly ``inf``.

Classes and modules
-------------------

* ``NewTransitGraph`` is now ``TransitGraph``. ``NewTransitGraph`` remains as a deprecated alias
  until 2.1.
* The legacy ``aequilibrae.paths.TransitGraph(GraphBase)`` class is removed, and
  ``aequilibrae.paths.TransitGraph`` is the transit graph above.
* The ``aequilibrae.paths.public_transport`` module alias is removed. Import ``HyperpathGenerating``
  from ``aequilibrae.paths`` instead.
* The ``seed``, ``geometry_noise`` and ``noise_coef`` arguments of ``TransitGraphBuilder``, which had
  no effect since 1.x, are removed. Arguments after ``num_threads`` are keyword-only.
* ``Transit``, ``TransitGraphBuilder`` and ``GTFSRouteSystemBuilder`` belong to a scenario. Given a
  project, they bind to its active scenario when created, and keep using that scenario if another one
  is activated later. ``Transit.graphs`` belongs to each ``Transit``, rather than being shared by all
  of them. ``GTFSRouteSystemBuilder`` takes the scenario as its ``scenario`` keyword argument;
  ``Transit.new_gtfs_builder`` passes it.

New tools
---------

* ``TransitCosting`` holds the costing of a transit class (see above).
* ``TransitClass.strategy(origin, destination)`` returns the optimal strategy of one trip between
  two zones: every edge it may use, by ``link_id``, with its expected volume and its split, the share
  of the flow through its tail that takes it. When the graph's edges have geometry (call
  ``create_line_geometry`` before ``to_transit_graph``), it is a GeoDataFrame ready for QGIS.
* ``TransitAssignment.save_skims(matrix_name)`` saves each class's skims as an OMX matrix and
  registers it in the project, as ``TrafficAssignment.save_skims`` does.
* Saved results and skims record where they came from: the period, the graph configuration and its
  hash, the scenario and the class costing, in the procedure report and in the skim matrix's
  ``provenance`` metadata. ``TransitAssignment.results()`` includes the ``period_id``, since transit
  link IDs are unique within a period only.
* ``TransitAssignmentResults`` reports ``unassigned_demand``, ``unassigned_od`` and
  ``weighted_waiting_time``: edge costs times volumes plus ``weighted_waiting_time`` equal demand times
  expected cost.
* The Spiess & Florian kernel now runs in C++. ``HyperpathGenerating`` keeps its interface and adds
  ``update_costs(trav_time, freq, headway_fraction=..., wait_weight=...)``, which replaces costs
  without rebuilding the graph, and ``strategy(origin, destination)`` for one trip between two
  vertices. Results are bitwise identical between runs with the same number of threads.
