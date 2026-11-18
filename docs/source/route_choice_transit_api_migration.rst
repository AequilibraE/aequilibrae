Migrating route choice and transit assignment
=============================================

Route choice
------------

Import ``RouteChoice`` and ``SubAreaAnalysis`` from
``aequilibrae.route_choice`` instead of ``aequilibrae.paths``. Users should
build a road graph using ``network.build_graph(mode, cost=...)`` or
``project.road.build_graph(mode, cost)`` and provide that to ``RouteChoice``.
If you use contraction, include the nodes you want to route between as
centroids so the graph always retains them. See :doc:`aequilibrae2_0` for the
project and graph changes.

Replace ``add_demand()`` calls with ``set_demand()``. It takes one complete
dense or sparse matrix collection rather than adding demand incrementally. If
your demand is an OD table, convert it with ``aequilibrae.matrix.from_df()``
first. Demand zone IDs must be graph centroids. By default, route choice
generates routes only for OD pairs with non-zero demand in at least one
matrix. To generate routes for explicit OD pairs without demand, call
``prepare([(origin, destination), ...])`` instead of ``set_demand()`` and run
``execute(perform_assignment=False)``.

Replace ``cutoff_prob`` with ``disutility_cutoff_constant`` and
``disutility_cutoff_coefficient`` in ``set_choice_set_generation()``. The
path-size logit calculation masks routes that cost more than ``constant +
coefficient * cheapest_route_cost`` for their OD pair. Set both options to
apply a cutoff, or omit both for no cutoff.

.. code-block:: python

    from aequilibrae.route_choice import RouteChoice

    route_choice = RouteChoice(graph, project=project)
    route_choice.set_choice_set_generation(
        "bfsle",
        max_routes=5,
        max_depth=5,
        disutility_cutoff_constant=0.0,
        disutility_cutoff_coefficient=1.5,
    )
    route_choice.set_demand(demand)
    route_choice.execute()
    routes = route_choice.get_results()

See :doc:`route_choice/choice_set_generation` for route generation options,
including A*.

Update saved route sets
~~~~~~~~~~~~~~~~~~~~~~~

``execute_single()`` now returns routes as ``(link_ids, directions)`` array
pairs. In result tables and Parquet route sets, ``route set`` contains unsigned
physical link IDs and ``route directions`` contains matching ``+1`` (a-to-b)
or ``-1`` (b-to-a) values in traversal order.

For a 1.x export that encodes direction in the sign of the link ID, split that
sign from the ID. Keep the ``origin id`` and ``destination id`` columns:

.. code-block:: python

    routes = legacy_routes.copy()
    routes["route directions"] = routes["route set"].map(
        lambda links: [1 if link > 0 else -1 for link in links]
    )
    routes["route set"] = routes["route set"].map(
        lambda links: [abs(link) for link in links]
    )
    route_choice.execute_from_pandas(routes, recompute_psl=True)

This conversion assumes non-zero signed IDs. If a legacy export contains a zero
ID, recover its direction from the source data before loading the route.
Routes must be valid on the current graph, including complete expansions of
contracted links, permitted turns and matching endpoints. With
``recompute_psl=True``, route choice recalculates costs, overlap and
probabilities. Otherwise, the input needs a ``probability`` column. For a
Parquet dataset, regenerate it with the new schema or load and convert it as
above before passing it back to route choice.

Transit assignment
------------------

Import ``TransitClass``, ``TransitAssignment`` and ``TransitGraph`` from
``aequilibrae.transit``. Obtain the builder through ``project.transit`` rather
than constructing or importing it directly. For an existing saved transit
graph, load the builder for its period and convert it with
``to_transit_graph()``. For a new graph, use ``project.transit.create_graph()``
to obtain the builder instead.

``TransitClass`` now takes keyword arguments and requires exactly one demand
matrix. Replace ``set_demand_matrix_core("pt")`` with ``matrix.subset(["pt"])``
before constructing the class. Dense and sparse demand are supported; values
must be finite, non-negative ``float64`` and the index must identify the
transit graph's centroids. Transit skims are still dense.

.. code-block:: python

    from aequilibrae.transit import TransitAssignment, TransitClass

    project.transit.load(period_ids=[period_id])
    builder = project.transit.graphs[period_id]
    transit_graph = builder.to_transit_graph()
    transit_class = TransitClass(name="pt", graph=transit_graph, matrix=matrix.subset(["pt"]))

    assignment = TransitAssignment(project=project)
    assignment.add_class(transit_class)
    assignment.set_time_field("trav_time")
    assignment.set_frequency_field("freq")
    assignment.set_algorithm("os")
    assignment.execute()

Unlike road assignment, transit assignment still uses setters to configure the
assignment. See :doc:`public_transport/hyperpath_routing` and
:doc:`public_transport/transit_skimming` for complete examples. If you are
migrating a GTFS-based project, also review the import changes in
:doc:`aequilibrae2_0` before deciding whether to rebuild the transit graph.

Transit is slated for an overhaul in an upcoming release.
