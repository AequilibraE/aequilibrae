Transit skimming
================

Transit skimming in AequilibraE is incredibly flexible, but more sophisticated use requires a good
understanding of the structure of the :ref:`transit_graph`, so we recommend reading that section first.

For typical use cases, the method `set_skimming_fields` accepts a set of predefined fields which are
defined based on the auto-generated link types. These include:

  - discrete: `'boardings'`, `'alightings'`, `'inner_transfers'`, `'outer_transfers'`, `'transfers'` and
    `'pt_probability'`.
  - continuous: `'trav_time'`, `'on_board_trav_time'`, `'dwelling_time'`, `'egress_trav_time'`, `'access_trav_time'`,
    `'walking_trav_time'`, `'transfer_time'`, `'in_vehicle_trav_time'`, `'waiting_time'`, `'first_waiting_time'` and
    `'transfer_waiting_time'`.

Each skim is the expected value of a quantity over the optimal strategy between two zones:

  - `'trav_time'` is the expected perceived cost of the strategy, the quantity the assignment
    minimises. It includes the class's weights and penalties (see ``TransitCosting``).
  - Times on links are physical. `'in_vehicle_trav_time'` adds the scheduled run times of on-board
    links and the scheduled dwell of dwell links, and the connector, walking and transfer times are
    walking distances over the walking speed. None of them are weighted.
  - `'waiting_time'` is computed while the strategy is built: at each stop, it is the class's headway
    fraction over the combined frequency of the lines the strategy boards there. It is never weighted
    by the class's wait weight. `'first_waiting_time'` is the wait at the first boarding, and
    `'transfer_waiting_time'` the rest.
  - `'boardings'` and `'alightings'` count boarding and alighting links, `'pt_probability'` is the
    probability that the strategy boards a vehicle at all, and `'transfers'` is `'boardings'` minus
    `'pt_probability'`, so a pair that walks all the way never reports negative transfers.
  - `'inner_transfers'` and `'outer_transfers'` count links of those types. Graphs built by
    AequilibraE have no inner transfer links, since a transfer at a stop alights to the stop and
    boards again, so prefer `'transfers'`.

Pairs of zones that no strategy connects skim as ``inf`` in every field, rather than zero, so that a
mode choice model cannot mistake them for free trips. Their demand is not loaded, and is reported in
the class results as ``unassigned_demand`` and ``unassigned_od``. Skims follow the order of the
graph's centroids, which is the zone order of the demand matrix.

Each transit class accepts either an ``AequilibraeMatrix`` or an ``AequilibraeSparseMatrix``
containing exactly one demand matrix, with finite, non-negative float64 values. Use
``matrix.subset(["pt"])`` to select a matrix before constructing the class if needed.
Skims are always dense ``AequilibraeMatrix`` objects using the demand's zone index.

On the Spiess and Florian (1989) example network, with waits of half a headway and no penalties, the
expected trip from A to B takes 27.75 minutes: 3.0 minutes of first wait, 1.25 minutes of transfer
wait and 23.5 minutes in vehicle, with 1.5 boardings and 0.5 transfers.

.. doctest::

    >>> import numpy as np
    >>> from aequilibrae.matrix import AequilibraeMatrix
    >>> from aequilibrae.paths import TransitAssignment, TransitClass, TransitCosting
    >>> from aequilibrae.transit import Transit
    >>> from aequilibrae.utils.create_example import create_example

    >>> project = create_example(project_path, "coquimbo")
    >>> data = Transit(project)

    >>> graph = data.create_graph(
    ...     with_outer_stop_transfers=False,
    ...     with_walking_edges=False,
    ...     blocking_centroid_flows=False,
    ...     connector_method="overlapping_regions",
    ... )

    >>> graph.create_line_geometry(method="direct", graph="c")

    >>> transit_graph = graph.to_transit_graph()

    >>> # We mock a demand matrix
    >>> num_zones = len(transit_graph.centroids)

    >>> mat = AequilibraeMatrix(index=transit_graph.centroids, index_name="zones")
    >>> mat["pt"] = np.full((num_zones, num_zones), 1.0)

    >>> # We can now execute the assignment, and we will use some of the default skimming fields
    >>> skim_cols = ["trav_time", "boardings", "transfers", "in_vehicle_trav_time", "waiting_time"]

    >>> # Waits of half a headway, perceived as twice as long as time in vehicle
    >>> assigclass = TransitClass(
    ...     name="pt", graph=transit_graph, matrix=mat, costing=TransitCosting(headway_fraction=0.5, wait_weight=2.0)
    ... )

    >>> assig = TransitAssignment()
    >>> assig.add_class(assigclass)
    >>> assig.set_time_field("trav_time")
    >>> assig.set_frequency_field("freq")

    >>> assig.set_skimming_fields(skim_cols) # Skimming must be set after a transit assignment class is added

    >>> assig.set_algorithm("os")

    >>> assig.execute()

    >>> skims = assig.get_skim_results()["pt"]

    >>> # Saving the skims writes an OMX file per class, recording the period, graph and costing they came from
    >>> file_names = assig.save_skims("pt_skims", project=project)

    >>> project.close()

More sophisticated skimming is also possible, such as skimming related to specific routes and/or modes.
As it is the case with traffic graphs, this type of exercise consists basically of defining fields in
the graph that represent the desired skimming metrics. Custom fields must be finite and nonnegative,
and are added up along the strategy.

One example is skimming in-vehicle time in rail only. The graph's physical ``run_time`` and
``dwell_time`` fields hold the scheduled times of on-board and dwell links.

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

    >>> transit_graph = graph.to_transit_graph()

    >>> # Line IDs end with the ID of the pattern they serve
    >>> all_routes = data.get_table("routes")
    >>> rail_patterns = all_routes.query("route_type in [1, 2]").pattern_id.astype(str).to_numpy()

    >>> # We now define a new field in the graph that will be used for skimming
    >>> edges = transit_graph.graph
    >>> on_rail = edges.link_type.isin(["on-board", "dwell"]) & edges.line_id.str.split("_").str[-1].isin(rail_patterns)
    >>> transit_graph.graph["rail_trav_time"] = np.where(on_rail, edges.run_time + edges.dwell_time, 0.0)

    >>> # We mock a demand matrix
    >>> num_zones = len(transit_graph.centroids)

    >>> mat = AequilibraeMatrix(index=transit_graph.centroids, index_name="zones")
    >>> mat["pt"] = np.full((num_zones, num_zones), 1.0)

    >>> assigclass = TransitClass(name="pt", graph=transit_graph, matrix=mat)

    >>> assig = TransitAssignment()
    >>> assig.add_class(assigclass)
    >>> assig.set_time_field("trav_time")
    >>> assig.set_frequency_field("freq")

    >>> # Skimming must be set after a transit assignment class is added
    >>> assig.set_skimming_fields(["rail_trav_time"])

    >>> assig.set_algorithm("os")

    >>> assig.execute()

    >>> project.close()
