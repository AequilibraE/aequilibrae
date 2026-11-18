Migrating road traffic assignment
=================================

In 1.x, you built and configured a graph for each traffic class, then called
setters on the assignment. In 2.0, instead build one independent ``Network``,
describe each class with a ``TrafficClass``, and let ``TrafficAssignment``
build the class graphs when you run it. See :doc:`aequilibrae2_0` for project
access and public imports, and :doc:`matrix_api_migration` for loading demand.

This new architecture helps reduce difficult to diagnose issues, where for
instance, different traffic classes where provided with incompatible networks.

Replace the graph and assignment setters
----------------------------------------

This example uses the Sioux Falls example project's car mode and ``demand.omx``.
With an open 2.0 ``project``, a basic assignment with skimming looks like this:

.. code-block:: python

    from aequilibrae.traffic import TrafficAssignment, TrafficClass
    from aequilibrae.traffic.vdf import bpr

    network = project.road.build_network()
    demand = project.matrices.get_matrix("demand.omx", subset=["matrix"])
    car = TrafficClass(
        name="car",
        mode="c",
        cost="free_flow_time",
        demand=demand,
        skimming=["free_flow_time", "distance"],
        graph_options={"block_centroid_flows": False},
    )
    assignment = TrafficAssignment(
        network=network,
        classes=[car],
        time_field="free_flow_time",
        capacity_field="capacity",
        vdf=bpr,
        name_mapping={"alpha": 0.15, "beta": 4.0},
        algorithm="bfw",
        project=project,
    )
    flows = assignment.execute(max_iterations=50, rgap=0.001, cores=4)

Use these replacements when rewriting a model script:

.. list-table:: Traffic assignment API replacements
   :header-rows: 1
   :widths: 46 54

   * - 1.x
     - 2.0
   * - ``TrafficClass(name, graph, matrix)`` and ``assignment.add_class()`` or
     ``set_classes()``
     - Construct ``TrafficClass`` with keyword arguments ``mode``, ``cost`` and
       ``demand``.  Pass the class list as ``classes=`` to
       ``TrafficAssignment``.
   * - ``graph.set_graph()``, ``set_skimming()`` and
     ``set_blocked_centroid_flows()``
     - Set ``cost``, ``skimming`` and ``graph_options`` on the class. The
       default class graph options enable link contraction, dead-end removal
       and centroid flow blocking.
   * - Class setters for PCE, value of time, fixed costs and select links
     - Pass ``pce``, ``vot``, ``fixed_cost_field``, ``fixed_cost_multiplier``
       and ``select_link_sets`` to the class constructor as needed. For
       example, ``select_link_sets={"screenline": [(link_id, direction)]}``
       uses a physical link ID and direction.
   * - ``set_vdf("BPR")`` and ``set_vdf_parameters()``
     - Pass a VDF object (``vdf=bpr``) and map its parameters to constants or
       network fields with ``name_mapping`` (for example, ``{"alpha": "b",
       "beta": "power"}``).
   * - ``set_time_field()``, ``set_capacity_field()`` and ``set_algorithm()``
     - Pass ``time_field``, ``capacity_field`` and ``algorithm`` to
       ``TrafficAssignment``.
   * - ``add_preload()``
     - Pass one ``preload=`` table with ``link_id``, ``direction`` and
       ``preload`` columns.  Combine multiple preloads by summing flows for
       each link and direction before constructing the assignment.
   * - Assigning ``max_iter`` or ``rgap_target``, or calling ``set_cores()``
     - Pass ``max_iterations``, ``rgap`` and ``cores`` to ``execute()``.

Use finite, non-negative ``float64`` demand and match the zone index to the
assignment centroids, including their order. Demand can be a dense or sparse
matrix collection. The class ``cost`` is free-flow generalised cost in time
units. ``time_field`` supplies the free-flow travel time used by the VDF, set
both to the same field when assigning on travel time alone. Preload flows must
be in PCE units. For more on costs, preloads and execution settings, see
:doc:`traffic_assignment/assignment_procedures` and
:doc:`traffic_assignment/volume_delay_functions`.

Read and save the results
-------------------------

Replace ``traffic_class.results`` with ``assignment.class_results[class_name]``
after ``execute()``. The run returns a link-level DataFrame, retrieve it again
with ``assignment.results()``. Link-load columns use the demand matrix names
with ``_ab``, ``_ba`` and ``_tot`` suffixes. Use distinct demand matrix names
across classes, result columns must be unique even when compared without case.

.. code-block:: python

    car_results = assignment.class_results["car"]
    convergence = assignment.report()
    assignment.save_results("base_year_assignment")
    assignment.save_skims("base_year_skims", which_ones="all")

The explicit ``project`` supplies parameters and a destination for saved
results.  For a standalone assignment, omit it to use package defaults, and
pass a project to ``save_results()`` or ``save_skims()`` when you need to save.

For select-link OD output, read ``car_results.select_link_od.matrices``: its
entries are keyed by selection name, then by demand matrix name. Each matrix is
a SciPy CSR array, use the demand index for its zone labels. Set
``select_link_od=False`` on the class if you only need select-link flows.

Road assignment no longer writes per-iteration path files. Remove calls to
``set_save_path_files()`` and ``set_path_file_format()`` from road assignment
scripts. This does not affect route choice Parquet exports. Path file saving
with support for turn penalties will be added in a future release. The
predecessor tree and all path finding results remain accessible at runtime.
