from venv import createTransit skimming
================

Transit skimming in AequilibraE is incredibly flexible, but more sophisticated use requires a good
understanding of the structure of the :ref:`transit_graph`, so we recommend reading that section first.

For typical use cases, the method `set_skimming_fields` accepts a set of predefined fields which are
defined based on the auto-generated link types. These include:

  - discrete: `'boardings'`, `'alightings'`, `'inner_transfers'`, `'outer_transfers'`, and `'transfers'`.
  - continuous: `'trav_time'`, `'on_board_trav_time'`, `'dwelling_time'`, `'egress_trav_time'`, `'access_trav_time'`,
    `'walking_trav_time'`, `'transfer_time'`, `'in_vehicle_trav_time'`, and `'waiting_time'`.


.. doctest::

    >>> from aequilibrae.paths import TransitAssignment, TransitClass

    >>> project = create_example(project_path, "coquimbo")
    >>> data = Transit(project)

    >>> graph = data.create_graph(
    ...     with_outer_stop_transfers=False,
    ...     with_walking_edges=False,
    ...     blocking_centroid_flows=False,
    ...     connector_method="overlapping_regions",
    ... )
    >>> project.network.build_graphs()

    >>> graph.create_line_geometry(method="direct", graph="c")

    >>> transit_graph = graph.to_transit_graph()

    >>> # We mock a demand matrix
    >>> num_zones = len(transit_graph.centroids)

    >>> mat = AequilibraeMatrix()
    >>> mat.create_empty(zones=num_zones, matrix_names=["pt"], memory_only=True)
    >>> mat.index = transit_graph.centroids[:]
    >>> mat.matrices[:, :, 0] = np.full((num_zones, num_zones), 1.0)
    >>> mat.computational_view()

    >>> # We can now execute the assignment, and we will use some of the default skimming fields
    >>> skim_cols = ["trav_time", "boardings", "in_vehicle_trav_time", "egress_trav_time", "access_trav_time"]

    >>> assigclass = TransitClass(name="pt", graph=transit_graph, matrix=mat)

    >>> assig = TransitAssignment()
    >>> assig.add_class(assigclass)
    >>> assig.set_time_field("trav_time")
    >>> assig.set_frequency_field("freq")

    >>> assig.set_skimming_fields(skim_cols) # Skimming must be set after a transit assignment class is added

    >>> assig.set_algorithm("os")
    >>> assigclass.set_demand_matrix_core("pt")

    >>> assig.execute() # doctest: +SKIP

    >>> project.close()

More sophisticated skimming is also possible, such as skimming related to specific routes and/or modes.
As it is the case with traffic graphs, this type of exercise consists basically of defining fields in
the graph that represent the desired skimming metrics.

One example is skimming travel time in rail only.

.. doctest::

    >>> from aequilibrae.paths import TransitAssignment, TransitClass

    >>> project = create_example(f"{project_path}v2", "coquimbo")
    >>> data = Transit(project)

    >>> graph = data.create_graph(
    ...     with_outer_stop_transfers=False,
    ...     with_walking_edges=False,
    ...     blocking_centroid_flows=False,
    ...     connector_method="overlapping_regions",
    ... )
    >>> project.network.build_graphs()

    >>> graph.create_line_geometry(method="direct", graph="c")

    >>> transit_graph = graph.to_transit_graph()

    >>> # We now define a new field in the graph that will be used for skimming
    >>> transit_graph.graph["rail_trav_time"] = np.where(
    ...      transit_graph.graph["link_type"].isin(["on-board", "dwell"]), 0, transit_graph.graph["trav_time"]
    ... ) # doctest: +SKIP

    >>> all_routes = transit.get_table("routes") # doctest: +SKIP
    >>> rail_ids = all_routes.query("route_type in [1, 2]").route_id.to_numpy() # doctest: +SKIP

    # Assign zero travel time to all non-rail links
    >>> transit_graph.graph.loc[~transit_graph.graph.line_id.isin(rail_ids),"rail_trav_time"] =0 # doctest: +SKIP

    >>> # We mock a demand matrix
    >>> num_zones = len(transit_graph.centroids)

    >>> mat = AequilibraeMatrix()
    >>> mat.create_empty(zones=num_zones, matrix_names=["pt"], memory_only=True)
    >>> mat.index = transit_graph.centroids[:]
    >>> mat.matrices[:, :, 0] = np.full((num_zones, num_zones), 1.0)
    >>> mat.computational_view()

    >>> # We can now execute the assignment, and we will use some of the default skimming fields
    >>> skim_cols = ["trav_time", "boardings", "in_vehicle_trav_time", "egress_trav_time", "access_trav_time"]

    >>> assigclass = TransitClass(name="pt", graph=transit_graph, matrix=mat)

    >>> assig = TransitAssignment()
    >>> assig.add_class(assigclass)
    >>> assig.set_time_field("trav_time")
    >>> assig.set_frequency_field("freq")

    >>> # Skimming must be set after a transit assignment class is added
    >>> assig.set_skimming_fields(["rail_trav_time"])  # doctest: +SKIP

    >>> assig.set_algorithm("os")
    >>> assigclass.set_demand_matrix_core("pt")

    >>> assig.execute() # doctest: +SKIP

    >>> project.close()
