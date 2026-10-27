Traffic Assignment Procedure
============================

Along with a network data model, traffic assignment is the most technically challenging portion to 
develop in a modeling platform, especially if you want it to be *fast*. In AequilibraE, we aim to 
make it as fast as possible, without making it overly complex to use, develop and maintain, although
we know that *complex* is subjective.

Traffic assignment starts with a network and a set of traffic class configurations.
The constructor copies source data and checks network compatibility without preparing routing topology.
Execution prepares routing contexts and runs the assignment.
For an example of a regional assignment procedure, see [1]_.

Network data
------------

``project.network`` manages the network database and can be passed directly to ``TrafficAssignment``.
Assignment derives the required fields from its configuration and loads them during construction.
An explicit ``project.network.build(fields=..., modes=...)`` returns an independent ``Network`` for reuse.

For workflows without a project, ``Network(links)`` accepts an in-memory link DataFrame.
The table requires ``link_id``, ``a_node``, ``b_node``, and ``direction`` columns.
A ``modes`` column limits which classes can use each link.
Without this column, all assignment modes can use all links.
``Network`` also accepts ``turn_restrictions`` and ``allow_uturns`` for in-memory assignments.
Traffic assignment accepts networks, not prepared graphs.

.. testsetup::

    from pathlib import Path
    from tempfile import TemporaryDirectory

    _assignment_docs_tmp = TemporaryDirectory()
    project_path = str(Path(_assignment_docs_tmp.name) / "project")

.. doctest::

    >>> from aequilibrae.utils.create_example import create_example
    >>> from aequilibrae.paths import TrafficClass, TrafficAssignment, Network
    >>> from aequilibrae.paths.vdf import bpr
    >>> project = create_example(project_path)
    >>> network = project.network

For standalone paths, ``network.build_graph(mode="c", cost="free_flow_time")``
loads and prepares a graph for that mode and cost field.
Its optional ``fields`` argument limits the loaded fields.
A directional field such as ``capacity`` loads both ``capacity_ab`` and ``capacity_ba``.

``TrafficClass``
----------------

``TrafficClass`` requires a mode, a cost field, and a demand matrix.
The demand matrices must contain float64 values.
The name defaults to the mode. Each class in an assignment must have a unique name.
The demand matrix can have an arbitrary number of user-classes setup as different layers (cores) of the matrix object.

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

* **Passenger-car equivalent** (PCE) is the standard way of modeling multi-class traffic assignment 
  equilibrium in a consistent manner (see [3]_ for the technical detail), and its value is set to 1.0 
  by default.

* **Fixed costs**: in case there are fixed costs associated with the traversal of links in the network, the user 
  can provide the name of the field in ``fixed_cost_field``.

* **Value-of-Time** (VoT) is the mechanism to bring time and monetary costs into a consistent basis 
  within a generalized cost function. In the event that fixed cost is measured in the same unit as free-flow travel
  time, then *vot* must be set to 1.0.

``pce`` and ``vot`` must be positive and finite.
``select_link_sets`` maps names to link IDs or ``(link_id, direction)`` pairs.
A link ID without a direction selects both directions.
``select_link_od=False`` retains selected link loads without allocating selected OD matrices.

``TrafficAssignment``
---------------------

``TrafficAssignment`` requires the network, classes, time field, capacity field, VDF, and algorithm.
All classes use the same network fields for travel time, capacity, and VDF parameters.

``name_mapping`` maps VDF parameter names to network field names or constant values.
Without an explicit mapping, the VDF defaults supply the values.
Network fields require an explicit mapping, even when their names match VDF parameters.

Because AequilibraE supports different parameters for each link, its implementation is the most general possible
while still preserving the desired properties for multi-class assignment, but the user needs to provide individual
values for each link *OR* a single value for the entire network.

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

Construction copies the source network and checks modes, centroids, fields, VDF parameters and preloads.
Invalid combinations fail before execution. Graph preparation remains deferred until execution.
Later edits to the original network do not change the assignment.
The project can close after construction.
Each execution creates fresh routing contexts and result buffers from the assignment's network snapshot.
The execution retains numeric network attributes for additional skims.
The returned DataFrame contains link results.
``assig.report()`` returns the convergence report.
``assig.info()`` returns configuration information before or after execution.

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
AequilibraE can skim during assignment and keeps both the latest all-or-nothing
skims and the blended skims for the accepted solution.

This is the case because, strictly speaking, the equilibrium travel time is the one resulting
at the end of the last assignment iteration, while the most correct distance and toll skims, for example,
are those resulting from the blended skim of all iterations.

The user can select the fields they want to skim, as well save the skims to disk (and the project)
at the end of the assignment procedure, where skims are tagged with suffixes *"_final"* and *"_blended"*
for easy identification.
The time-field skim includes congestion and turn delay. It excludes fixed monetary costs.
A separately named class-cost skim includes the generalized assignment cost.
If cost and time use the same field name, the skim reports travel time.
``skim_congested()`` also returns the generalized cost as ``__assignment_cost__``.

Unassigned demand
~~~~~~~~~~~~~~~~~

Unassigned demand is demand that present within the demand matrix but it's destination was unreachable from the origin,
it is not loaded on the network and ``assig._class_states[0]._aon_results.unassigned_demand`` gives the latest AoN total,
excluding intrazonal demand. ``assig.class_results["car"].unassigned_demand`` gives the total of the accepted solution.

Assigning sparse matrices
^^^^^^^^^^^^^^^^^^^^^^^^^
Modern Activity-Based models (and even some trip-based and tour-based ones) result on incredibly sparse
demand matrices, which opens up a significant opportunity to save time during assignment by using early-exiting
during the path-computation phase of assignment.

AequilibraE is capable of leveraging this opportunity, and it does so automatically whenever the user
**does NOT set skimming for any individual traffic class**.

In this case, AequilibraE has a convenient method to skim the final iteration of the assignment
as it had been computed during the assignment itself. This method call requires a new iteration
of path computation to be made, but assignments with highly sparse matrices and more than 10 iterations
would still experience a significant speedup, which comes at the cost of not having blended skims
as a sub-product of the assignment.

Time savings of up to 40% should be achievable in cases of micro-simulated ABMs with a large number
of zones and over 100 iterations of assignment.

.. doctest::

    >>> skims = assig.skim_congested(skim_fields=["distance"], return_matrices=True)
    >>> assig.save_skims("another_matrix_name")  # doctest: +ELLIPSIS
    [MatrixRecord(name='another_matrix_name_car', ...), MatrixRecord(name='another_matrix_name_truck', ...)]

The list of fields defined by the user for skimming is added to the congested time and the assignment cost from the last
iteration of the assignment by default. These matrices are named *__congested_time__* and *__assignment_cost__*
respectively. The returned values are ``SkimmingOutputs`` objects.
Congested time includes link travel time and turn delay, but excludes fixed costs. Assignment cost includes those fixed
costs, converted to time units with the traffic classes value of time, and the class cost adjustment.

See the the example :ref:`example_assign_sparse` for a more practical explanation of this feature.

Selected OD matrices use the sorted union of the participating class centroids during export.
The exporter reorders each matrix by external centroid ID and fills absent centroid pairs with zero.


Volume-delay function
~~~~~~~~~~~~~~~~~~~~~

AequilibraE supports five Volume Delay Functions (VDFs): BPR, BPR2, Conical, INRETS, and Akcelik.
Each VDF has distinct characteristics that make it suitable for different modeling scenarios.

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

Setting Preloads
----------------

We can also optionally include a preload vector for constant flows which are not being otherwise modelled. 
For example, this can be used to account for scheduled  public transport vehicles, adding an equivalent 
load to each link along the route accordingly. AequilibraE supports various conditions for which PT trips 
to include in the preload, and allows the user to specify the PCE for each type of vehicle in the public transport 
network.

The constructor accepts a ``preload`` array or a DataFrame.
The array uses directed links in ascending link-ID and direction order.
The DataFrame requires ``link_id``, ``direction``, and ``preload`` columns.
Preload values must be nonnegative and finite.

To create a preload for public transport vehicles operating between 8 AM to 10 AM, do the following:
The project must contain a transit database.

.. code-block:: python

    from aequilibrae.transit import Transit

    # Times are specified in seconds from midnight
    transit = Transit(project)
    preload = transit.build_pt_preload(start=8*3600, end=10*3600)

    # Add the preload to the assignment
    preloaded_assig = TrafficAssignment(
        network=network, classes=[tc_car, tc_truck], time_field="free_flow_time",
        capacity_field="capacity", vdf=bpr,
        name_mapping={"alpha": "b", "beta": "power"}, algorithm="bfw",
        preload=preload,
    )
    preloaded_assig.execute(max_iterations=20, rgap=1e-4)

.. doctest::

    >>> project.close()

References
----------

.. [1] Hampton Roads Transportation Planning Organization, Regional Travel Demand Model V2 (2020). 
       Available in: https://www.hrtpo.org/uploads/docs/2020_HamptonRoads_Modelv2_MethodologyReport.pdf

.. [2] Spiess, H. (1990) "Technical Note—Conical Volume-Delay Functions."Transportation Science, 24(2): 153-158.
       Available in: https://doi.org/10.1287/trsc.24.2.153

.. [3] Zill, J., Camargo, P., Veitch, T., Daisy, N. (2019) "Toll Choice and Stochastic User Equilibrium: 
       Ticking All the Boxes", Transportation Research Record, 2673(4):930-940. 
       Available in: https://doi.org/10.1177%2F0361198119837496
