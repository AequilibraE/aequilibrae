Traffic Assignment Procedure
============================

Along with a network data model, traffic assignment is the most technically challenging portion to 
develop in a modeling platform, especially if you want it to be *fast*. In AequilibraE, we aim to 
make it as fast as possible, without making it overly complex to use, develop and maintain, although
we know that *complex* is subjective.

Running traffic assignment in AequilibraE consists in creating the traffic classes that are going
to be assigned, add them to a traffic assignment object, set the traffic assignment parameters, and
run the assignment.

``TrafficClass``
----------------

The ``TrafficClass`` object defines a single user class independently of the network. It holds the
mode, cost field, demand matrix, optional skimming fields, and any fixed-cost or select-link configuration.
The required information is:

* **mode**: the network mode identifier (for example ``"c"`` for cars)
* **cost**: the free-flow generalized cost field used for route choice
* **demand**: the ``AequilibraeMatrix`` containing OD demand for that class
* **name**: an optional class label; when omitted, it defaults to the mode

.. testsetup::

    from pathlib import Path
    from tempfile import TemporaryDirectory

    _assignment_docs_tmp = TemporaryDirectory()
    project_path = str(Path(_assignment_docs_tmp.name) / "project")

.. doctest::

    >>> from aequilibrae.paths import TrafficClass
    >>> from aequilibrae.utils.create_example import create_example

    >>> project = create_example(project_path)
    >>> project.network.build_graphs()

    # We get the graphs for cars and trucks
    >>> graph_car = project.network.graphs['c']
    >>> graph_truck = project.network.graphs['T']

    # And also get the matrices for cars and trucks
    >>> matrix_car = project.matrices.get_matrix("demand_mc", subset=["car"])
    >>> matrix_truck = project.matrices.get_matrix("demand_mc", subset=["trucks"])

    # We create the traffic classes
    >>> tc_car = TrafficClass(mode="c", cost="free_flow_time", demand=matrix_car,
    ...                     name="car", skimming=("free_flow_time", "distance"),
    ...                     block_centroid_flows=False)
    >>> tc_truck = TrafficClass(mode="T", cost="free_flow_time", demand=matrix_truck,
    ...                      name="truck", skimming=("free_flow_time", "distance"),
    ...                      block_centroid_flows=False)

It is also possible to set the common assignment parameters during class definition:

* **pce**: passenger-car equivalent for multi-class assignment; default is ``1.0``.
* **vot**: value of time used to convert fixed costs to time units; default is ``1.0``.
* **fixed_cost_field**: link field providing a time-equivalent fixed cost, if needed.
* **fixed_cost_multiplier**: scale applied to the fixed cost field.

.. doctest::

    >>> tc_truck = TrafficClass(mode="T", cost="free_flow_time", demand=matrix_truck,
    ...                      name="truck", pce=2.5, vot=0.35,
    ...                      fixed_cost_field="distance", fixed_cost_multiplier=1.0)

``TrafficAssignment``
---------------------

.. doctest::

    >>> from aequilibrae.paths import TrafficAssignment
    >>> from aequilibrae.paths.vdf import bpr

    >>> assignment = TrafficAssignment(
    ...     network=project.network,
    ...     classes=[tc_car, tc_truck],
    ...     time_field="free_flow_time",
    ...     capacity_field="capacity",
    ...     vdf=bpr,
    ...     algorithm="bfw",
    ...     name_mapping={"alpha": "b", "beta": "power"},
    ... )

AequilibraE's traffic assignment is organized around a single constructor that snapshots the network and
current class configuration, then prepares the required routing contexts during execution.

* **network**: the source network object or project network containing the links and modes
* **classes**: one or more fully-specified ``TrafficClass`` objects
* **vdf**: the volume-delay function (VDF) to be used, being one of ``BPR``, ``BPR2``, ``CONICAL``,
  ``INRETS``, or ``AKCELIK``
* **name_mapping**: parameter names for the VDF, either as network fields or numeric constants
* **time_field**: link field representing free-flow travel time
* **capacity_field**: link field representing capacity
* **algorithm**: assignment algorithm, such as ``"bfw"``, ``"fw"``, ``"msa"``, or ``"all-or-nothing"``

.. doctest::

    >>> assignment = TrafficAssignment(
    ...     network=project.network,
    ...     classes=[tc_car, tc_truck],
    ...     time_field="free_flow_time",
    ...     capacity_field="capacity",
    ...     vdf=bpr,
    ...     algorithm="bfw",
    ...     name_mapping={"alpha": 0.15, "beta": 4.0},
    ... )

Skimming while assigning
~~~~~~~~~~~~~~~~~~~~~~~~

AequilibraE can skim during assignment and keeps both the latest all-or-nothing
skims and the blended skims for the accepted solution.

This is the case because, strictly speaking, the equilibrium travel time is the one resulting
at the end of the last assignment iteration, while the most correct distance and toll skims, for example,
are those resulting from the blended skim of all iterations.

The user can select the fields they want to skim, as well save the skims to disk (and the project)
at the end of the assignment procedure, where skims are tagged with suffixes *"_final"* and *"_blended"*
for easy identification.

::

  >>> assig.set_skimming_fields(["distance"]) # doctest: +SKIP
  >>> assig.execute() # doctest: +SKIP
  >>> assig.save_skims("one_matrix_name") # doctest: +SKIP

Unassigned demand
~~~~~~~~~~~~~~~~~

Unassigned demand is demand that present within the demand matrix but it's destination was unreachable from the origin,
it is not loaded on the network and ``traffic_class._aon_results.unassigned_demand`` gives the latest AoN total,
excluding intrazonal demand. ``traffic_class.results.unassigned_demand`` gives the total of the accepted solution.

Assigning sparse matrices
^^^^^^^^^^^^^^^^^^^^^^^^^
Modern Activity-Based models (and even some trip-based and tour-based ones) result on incredibly sparse
demand matrices, which opens up a significant opportunity to save time during assignment by using early-exiting
during the path-computation phase of assignment.

AequilibraE is capable of leveraging this opportunity, and it does so automatically whenever the user
**does NOT set skimming for the assignment or any individual traffic class graphs**.

In this case, AequilibraE has a convenient method to skim the final iteration of the assignment
as it had been computed during the assignment itself. This method call requires a new iteration
of path computation to be made, but assignments with highly sparse matrices and more than 10 iterations
would still experience a significant speedup, which comes at the cost of not having blended skims
as a sub-product of the assignment.

Time savings of up to 40% should be achievable in cases of micro-simulated ABMs with a large number
of zones and over 100 iterations of assignment.

::

  >>> assig.execute()
  >>> skims = assig.skim_congested(skim_fields=["distance"], return_matrices=True)
  >>> assig.save_skims("another_matrix_name")  # doctest: +ELLIPSIS
  [MatrixRecord(name='another_matrix_name_car', ...), MatrixRecord(name='another_matrix_name_truck', ...)]

The list of fields defined by the user for skimming is added to the congested time and the assignment cost from the last
iteration of the assignment by default. These matrices are named *__congested_time__* and *__assignment_cost__*
respectively. The returned values are ``SkimmingOutputs`` objects, also stored in ``traffic_class.congested_skims``.
Congested time includes link travel time and turn delay, but excludes fixed costs. Assignment cost includes those fixed
costs, converted to time units with the traffic classes value of time.

See the the example :ref:`example_assign_sparse` for a more practical explanation of this feature.

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
* **Conical**: Smooth theoretical function (α=0.15, β=4.0)
* **INRETS**: Urban arterials (α=1.0)
* **Akcelik**: Signalized intersections (α=0.25, τ=0.8)

Setting Preloads
----------------

We can also optionally include a preload vector for constant flows which are not being otherwise modelled. 
For example, this can be used to account for scheduled  public transport vehicles, adding an equivalent 
load to each link along the route accordingly. AequilibraE supports various conditions for which PT trips 
to include in the preload, and allows the user to specify the PCE for each type of vehicle in the public transport 
network.

To create a preload for public transport vehicles operating between 8 AM to 10 AM, do the following:

.. doctest::

    >>> from aequilibrae.transit import Transit

    # Times are specified in seconds from midnight
    >>> transit = project.scenario.create_transit_database()
    >>> preload = transit.build_pt_preload(start=8*3600, end=10*3600)

    # Add the preload to the assignment
    >>> assig.add_preload(preload, 'PT_vehicles') # doctest: +SKIP


Executing an Assignment
-----------------------

Finally, run traffic assignment!

.. doctest::

  >>> assig.execute()

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
