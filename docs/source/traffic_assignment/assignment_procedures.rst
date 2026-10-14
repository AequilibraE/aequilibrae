Traffic Assignment Procedure
============================

Traffic assignment starts with a network snapshot and a set of traffic class configurations.
The constructor validates the configuration. Execution prepares the graphs and runs the assignment.

Network data
------------

``project.network`` manages the network database. Its ``build()`` method returns a
``NetworkData`` snapshot with link fields, available modes, centroids, and turn restrictions.
``NetworkData`` also accepts a DataFrame for workflows without a project.

.. doctest::

    >>> from aequilibrae.paths import TrafficClass, TrafficAssignment, NetworkData
    >>> from aequilibrae.paths.vdf import bpr
    >>> project = create_example(project_path)
    >>> network = project.network.build(modes=["c", "T"])

The ``fields`` argument limits the loaded fields. A directional field such as
``capacity`` loads both ``capacity_ab`` and ``capacity_ba``.
For standalone paths, ``network.build_graph(mode="c", cost="free_flow_time")``
returns a graph for that mode and cost field.

Traffic classes
---------------

``TrafficClass`` requires a mode, a cost field, and a demand matrix.
The matrix must have a float64 computational view.
The name defaults to the mode. Each class in an assignment must have a unique name.

.. doctest::

    >>> matrix_car = project.matrices.get_matrix("demand_mc")
    >>> matrix_car.computational_view("car")
    >>> matrix_truck = project.matrices.get_matrix("demand_mc")
    >>> matrix_truck.computational_view("trucks")
    >>> tc_car = TrafficClass(
    ...     name="car", mode="c", cost="free_flow_time", demand=matrix_car,
    ...     skimming=["free_flow_time", "distance"], block_centroid_flows=False,
    ... )
    >>> tc_truck = TrafficClass(
    ...     name="truck", mode="T", cost="free_flow_time", demand=matrix_truck, pce=2.5,
    ...     skimming=["free_flow_time", "distance"], block_centroid_flows=False,
    ... )

Traffic classes contain configuration only. They do not hold graphs or results.
The configuration is immutable. ``dataclasses.replace()`` creates a configuration with changed values.

The ``cost`` field defines the free-flow generalized cost in time units.
The assignment adds congestion delay to this cost.
The assignment converts fixed monetary costs to time units through the value of time.
The cost formula is:

.. code-block:: text

    class cost + congestion delay + fixed cost * fixed_cost_multiplier / vot

``fixed_cost_field`` identifies the monetary cost field.
``pce`` and ``vot`` default to 1.0 and must be positive and finite.
``select_link_sets`` maps names to link IDs or ``(link_id, direction)`` pairs.
A link ID without a direction selects both directions.
``select_link_od=False`` retains selected link loads without allocating selected OD matrices.

Assignment configuration
------------------------

``TrafficAssignment`` requires the network, classes, time field, capacity field, VDF, and algorithm.
All classes use the same network fields for travel time, capacity, and VDF parameters.

``name_mapping`` maps VDF parameter names to network field names or constant values.
Without an explicit mapping, matching field names supply the values.
Otherwise, the VDF defaults supply the values.

.. doctest::

    >>> assig = TrafficAssignment(
    ...     network=network, classes=[tc_car, tc_truck], time_field="free_flow_time",
    ...     capacity_field="capacity", vdf=bpr,
    ...     name_mapping={"alpha": "b", "beta": "power"}, algorithm="bfw",
    ... )
    >>> results = assig.execute(max_iterations=20, rgap=1e-4, cores=1)

Algorithms are ``all-or-nothing``, ``msa``, ``frank-wolfe``, ``cfw``, and ``bfw``.
The ``max_iterations``, ``rgap``, and ``cores`` arguments apply to each execution.
Zero cores uses all available cores. A negative value leaves that many cores unused.
``bfw_conjugacy="exact"`` selects the exact BFW conjugacy calculation.

Each execution creates fresh graphs and result buffers.
The returned DataFrame contains link results.
``assig.report()`` returns the convergence report.
``assig.info()`` returns configuration information before or after execution.

Preloads
--------

The constructor accepts a ``preload`` array or a DataFrame.
The array uses directed links in ascending link-ID and direction order.
The DataFrame requires ``link_id``, ``direction``, and ``preload`` columns.
Preload values must be nonnegative and finite.
Preloads represent constant background traffic, such as public transport vehicles.

Results and skimming
--------------------

``assig.class_results[name]`` returns the accepted result for each traffic class.
Results remain in demand units. The optimizer applies PCE when it combines class flows.
``unassigned_demand`` reports demand with an unreachable destination and excludes intrazonal demand.

.. doctest::

    >>> car_results = assig.class_results["car"]
    >>> skims = car_results.skims
    >>> saved = assig.save_results("assignment_results")
    >>> saved_skims = assig.save_skims("assignment_skims", which_ones="all")

Fields in ``TrafficClass.skimming`` produce skims during assignment.
Without skimming fields, the path search can exit early for sparse demand.
``skim_congested()`` computes additional skims from the accepted final costs.

.. doctest::

    >>> final_skims = assig.skim_congested(["distance"], return_matrices=True)
    >>> project.close()

Congested skims contain ``__congested_time__`` and ``__assignment_cost__``.
Congested time includes travel time and turn delay.
Assignment cost also includes fixed costs and the class cost adjustment.

For VDF formulas and custom functions, see :ref:`volume_delay_functions`.
For a sparse demand example, see :ref:`example_assign_sparse`.
