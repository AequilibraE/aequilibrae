Traffic Assignment Procedure
============================

Traffic assignment starts with a project network or an in-memory graph and a set of traffic class configurations.
The constructor checks the configuration without reading network tables.
Execution loads and validates network values, prepares routing contexts, and runs the assignment.
For an example of a regional assignment procedure, see [1]_.

Network data
------------

``project.network`` manages the network database and can be passed directly to ``TrafficAssignment``.
Assignment derives the required modes and fields from its configuration.
No network build is necessary before assignment.

For workflows without a project, ``Graph(network=links)`` accepts an in-memory link DataFrame.
The table requires ``link_id``, ``a_node``, ``b_node``, and ``direction`` columns.
A ``modes`` column limits which classes can use each link.
Without this column, all assignment modes can use all links.
``Graph`` also accepts ``turn_restrictions`` and ``allow_uturns`` for in-memory assignments.

.. testsetup::

    from pathlib import Path
    from tempfile import TemporaryDirectory

    _assignment_docs_tmp = TemporaryDirectory()
    project_path = str(Path(_assignment_docs_tmp.name) / "project")

.. doctest::

    >>> from aequilibrae.utils.create_example import create_example
    >>> from aequilibrae.paths import TrafficClass, TrafficAssignment, Graph
    >>> from aequilibrae.paths.vdf import bpr
    >>> project = create_example(project_path)
    >>> network = project.network

For standalone paths, ``network.build_graph(mode="c", cost="free_flow_time")``
loads and prepares a graph for that mode and cost field.
Its optional ``fields`` argument limits the loaded fields.
A directional field such as ``capacity`` loads both ``capacity_ab`` and ``capacity_ba``.

Traffic classes
---------------

``TrafficClass`` requires a mode, a cost field, and a demand matrix.
The demand matrices must contain float64 values.
The name defaults to the mode. Each class in an assignment must have a unique name.

.. doctest::

    >>> matrix_car = project.matrices.get_matrix("demand_mc", subset=["car"])
    >>> matrix_truck = project.matrices.get_matrix("demand_mc", subset=["trucks"])
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
Passenger-car equivalents (PCE) put the flows from different vehicle classes on a common basis for equilibrium calculations [3]_.
Value of time (VoT) converts monetary costs to the time units of the cost field.
If fixed costs already use these time units, ``vot`` must equal 1.0.
``select_link_sets`` maps names to link IDs or ``(link_id, direction)`` pairs.
A link ID without a direction selects both directions.
``select_link_od=False`` retains selected link loads without allocating selected OD matrices.

Assignment configuration
------------------------

``TrafficAssignment`` requires the network, classes, time field, capacity field, VDF, and algorithm.
All classes use the same network fields for travel time, capacity, and VDF parameters.

``name_mapping`` maps VDF parameter names to network field names or constant values.
Without an explicit mapping, the VDF defaults supply the values.
Network fields require an explicit mapping, even when their names match VDF parameters.

.. doctest::

    >>> assig = TrafficAssignment(
    ...     network=network, classes=[tc_car, tc_truck], time_field="free_flow_time",
    ...     capacity_field="capacity", vdf=bpr,
    ...     name_mapping={"alpha": "b", "beta": "power"}, algorithm="bfw",
    ... )
    >>> results = assig.execute(max_iterations=20, rgap=1e-4, cores=1)

Algorithms are ``all-or-nothing``, ``msa``, ``frank-wolfe``, ``cfw``, and ``bfw``.
``fw`` is an alias for ``frank-wolfe``.
The ``max_iterations``, ``rgap``, and ``cores`` arguments apply to each execution.
Zero cores uses all available cores. A negative value leaves that many cores unused.
``bfw_conjugacy="exact"`` selects the exact BFW conjugacy calculation.

Each execution reads the current network and creates fresh routing contexts and result buffers.
Network edits before execution are included in that run.
The project must remain open until execution loads its inputs.
Network edits after execution do not change the results or additional skims from that run.
The execution retains numeric network attributes for additional skims.
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
AequilibraE can calculate these flows from transit schedules and vehicle PCE values.
The following example uses services between 8 AM and 10 AM.
The project must contain a transit database.

.. code-block:: python

    from aequilibrae.transit import Transit

    transit = Transit(project)
    preload = transit.build_pt_preload(start=8 * 3600, end=10 * 3600)
    preloaded_assig = TrafficAssignment(
        network=network, classes=[tc_car, tc_truck], time_field="free_flow_time",
        capacity_field="capacity", vdf=bpr,
        name_mapping={"alpha": "b", "beta": "power"}, algorithm="bfw",
        preload=preload,
    )
    preloaded_assig.execute(max_iterations=20, rgap=1e-4)

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

Skimming while assigning
~~~~~~~~~~~~~~~~~~~~~~~~

Fields in ``TrafficClass.skimming`` produce skims during assignment.
Assignment retains the skims from the latest all-or-nothing iteration and blends skims for the accepted solution.
The final travel-time skim describes the latest paths.
Blended distance and toll skims describe the paths used across iterations.
``save_skims()`` labels these outputs with ``_final`` and ``_blended`` suffixes.
The time-field skim includes congestion and turn delay. It excludes fixed monetary costs.
A separately named class-cost skim includes the generalized assignment cost.
If cost and time use the same field name, the skim reports travel time.
``skim_congested()`` also returns the generalized cost as ``__assignment_cost__``.
Assigning sparse matrices
~~~~~~~~~~~~~~~~~~~~~~~~~

Activity-based models often produce sparse demand matrices.
Without skimming fields, the path search can exit after it reaches all destinations with demand.
This can reduce assignment time, but the assignment does not produce blended skims.
``skim_congested()`` computes additional skims from the accepted final costs.
These skims include each requested field, ``__congested_time__``, and ``__assignment_cost__``.

.. doctest::

    >>> final_skims = assig.skim_congested(["distance"], return_matrices=True)
    >>> project.close()

Congested skims contain ``__congested_time__`` and ``__assignment_cost__``.
Congested time includes travel time and turn delay.
Assignment cost also includes fixed costs and the class cost adjustment.

For VDF formulas and custom functions, see :ref:`volume_delay_functions`.
For a sparse demand example, see :ref:`example_assign_sparse`.

Selected OD matrices use the sorted union of the participating class centroids during export.
The exporter reorders each matrix by external centroid ID and fills absent centroid pairs with zero.


Volume-delay function
~~~~~~~~~~~~~~~~~~~~~

AequilibraE supports five Volume Delay Functions (VDFs): BPR, BPR2, Conical, INRETS, and Akcelik.
Each VDF has distinct characteristics that make it suitable for different modeling scenarios.
The conical formulation follows Spiess [2]_.

For detailed information on VDF formulations, parameter guidance, and usage recommendations,
see the dedicated :ref:`volume_delay_functions` page, which includes:

* Mathematical formulas and technical descriptions
* Default parameter values and calibration guidance
* Behavioral charts showing V/C ratios from 0 to 3
* Origin and background for each function
* Recommendations for choosing the right VDF

**Quick reference:**

* **BPR**: Standard highway assignment (α=0.15, β=4.0)
* **BPR2**: Enhanced over-capacity penalty (α=0.15, β=4.0)
* **Conical**: Smooth theoretical function (α=2.0, β=1.5)
* **INRETS**: Urban arterials (α=1.0)
* **Akcelik**: Signalized intersections (α=0.25, τ=0.8)

References
----------

.. [1] Hampton Roads Transportation Planning Organization, Regional Travel Demand Model V2 (2020).
       Available in: https://www.hrtpo.org/uploads/docs/2020_HamptonRoads_Modelv2_MethodologyReport.pdf

.. [2] Spiess, H. (1990) "Technical Note—Conical Volume-Delay Functions."Transportation Science, 24(2): 153-158.
       Available in: https://doi.org/10.1287/trsc.24.2.153

.. [3] Zill, J., Camargo, P., Veitch, T., Daisy, N. (2019) "Toll Choice and Stochastic User Equilibrium:
       Ticking All the Boxes", Transportation Research Record, 2673(4):930-940.
       Available in: https://doi.org/10.1177%2F0361198119837496
