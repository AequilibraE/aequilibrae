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

The ``TrafficClass`` object holds all the information pertaining to a specific traffic class to be 
assigned. There are three pieces of information that are required in the instantiation of this class:

* **name**: name of the class. It has to be unique among all classes used in a multi-class traffic assignment

* **graph**: it is the ``Graph`` object corresponding to that particular traffic class/mode

* **matrix**: it is the AequilibraE matrix with the demand for that traffic class, which can have
  an arbitrary number of user-classes setup as different layers (cores) of the matrix object.

.. doctest::

    >>> from aequilibrae.paths import TrafficClass

    >>> project = create_example(project_path)
    >>> project.network.build_graphs()

    # We get the graphs for cars and trucks
    >>> graph_car = project.network.graphs['c']
    >>> graph_truck = project.network.graphs['T']

    # And also get the matrices for cars and trucks
    >>> matrix_car = project.matrices.get_matrix("demand_mc")
    >>> matrix_car.computational_view("car")

    >>> matrix_truck = project.matrices.get_matrix("demand_mc")
    >>> matrix_truck.computational_view("trucks")

    # We create the Traffic Classes
    >>> tc_car = TrafficClass("car", graph_car, matrix_car)
    >>> tc_truck = TrafficClass("truck", graph_truck, matrix_truck)

It is also possible to modify the default values for the following parameters of a traffic classe by using a 
method call:

* **Passenger-car equivalent** (PCE) is the standard way of modeling multi-class traffic assignment 
  equilibrium in a consistent manner (see [3]_ for the technical detail), and its value is set to 1.0 
  by default.

.. doctest::

    >>> tc_truck.set_pce(2.5)

* **Fixed costs**: in case there are fixed costs associated with the traversal of links in the network, the user 
  can provide the name of the field in the graph that contains that network.

.. doctest::

    >>> tc_truck.set_fixed_cost("distance")

* **Value-of-Time** (VoT) is the mechanism to bring time and monetary costs into a consistent basis 
  within a generalized cost function. In the event that fixed cost is measured in the same unit as free-flow travel
  time, then *vot* must be set to 1.0.

.. doctest::

    >>> tc_truck.set_vot(0.35)

``TrafficAssignment``
---------------------

.. doctest::

    >>> from aequilibrae.paths import TrafficAssignment

    >>> assig = TrafficAssignment()

AequilibraE's traffic assignment is organized within an object with the same name which contains a series of
member variables that should be populated by the user, providing thus a complete specification of the assignment
procedure.

* **classes**: list of completely specified traffic classes

.. doctest::

    # You can add one or more traffic classes to the assignment instance
    >>> assig.add_class(tc_truck) # doctest: +SKIP

    >>> assig.set_classes([tc_car, tc_truck])

* **vdf**: the volume-delay function (VDF) to be used, being one of ``BPR``, ``BPR2``, ``CONICAL``, or ``INRETS``

.. doctest::

    >>> assig.set_vdf('BPR')

* **vdf_parameters**: the parameters to be used in the volume-delay function, other than volume, capacity and 
  free-flow time. VDF parameters must be consistent across all graphs.

  Because AequilibraE supports different parameters for each link, its implementation is the most general possible 
  while still preserving the desired properties for multi-class assignment, but the user needs to provide individual
  values for each link *OR* a single value for the entire network.

  Setting the VDF parameters should be done *AFTER* setting the VDF function of choice and adding traffic classes 
  to the assignment, or it will *fail*.

.. doctest::

    # The VDF parameters can be either an existing field in the graph, passed as a parameter:
    >>> assig.set_vdf_parameters({"alpha": "b", "beta": "power"}) # doctest: +SKIP

    # Or as a global value:
    >>> assig.set_vdf_parameters({"alpha": 0.15, "beta": 4})

* **time_field**: the field of the graph that corresponds to free-flow travel time. The procedure will 
  collect this information from the graph associated with the first traffic class provided, but will check 
  if all graphs have the same information on free-flow travel time
  
.. doctest::

    >>> assig.set_time_field("free_flow_time")

* **capacity_field**: the field of the graph that corresponds to the link capacity. The procedure will collect 
  this information from the graph associated with the first traffic class provided, but will check if all graphs
  have the same information on capacity

.. doctest::

    >>> assig.set_capacity_field("capacity")

* **algorithm**: the assignment algorithm to be used, being one of ``all-or-nothing``, ``bfw``, ``cfw``, ``fw``,  
  ``franke-wolfe``, or ``msa``.

.. doctest::

    >>> assig.set_algorithm("bfw")

Skimming while assigning
~~~~~~~~~~~~~~~~~~~~~~~~

AequilibraE allows for skimming to be performed during assignnment, and maintains both the skimming
of the final iteration, as well as the blended skim for all iterations.

This is the case because, strictly speaking, the equilibrium travel time is the one resulting
at the end of the last assignment iteration, while the most correct distance and toll skims, for example,
are those resulting from the blended skim of all iterations.

The user can select the fields they want to skim, as well save the skims to disk (and the project)
at the end of the assignment procedure, where skims are tagged with suffixes *"_final"* and *"_blended"*
for easy identification.

::

  >>> assig.set_skimming_fields(["distance"]) # doctest: +SKIP
  >>> assig.execute() # doctest: +SKIP
  >>> assig.save_skims("one_matrix_name")


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
  >>> assig.save_skims("another_matrix_name")

The list of fields defined by the user for skimming is added to the congested time and the assignment
cost from the last iteration of the assignment by default. These matrices are named *__congested_time__*
and *__assignment_cost__* respectively.

See the the example :ref:`example_assign_sparse` for a more practical explanation of this feature.

Volume-delay function
~~~~~~~~~~~~~~~~~~~~~

For now, the VDF functions available in AequilibraE are

* BPR [1]_

.. math:: CongestedTime_{i} = FreeFlowTime_{i} * (1 + \alpha * (\frac{Volume_{i}}{Capacity_{i}})^\beta)

* Spiess' conical [2]_

.. math:: CongestedTime_{i} = FreeFlowTime_{i} * (2 + \sqrt[2][\alpha^2*(1- \frac{Volume_{i}}{Capacity_{i}})^2 + \beta^2] - \alpha *(1-\frac{Volume_{i}}{Capacity_{i}})-\beta)

* and French INRETS (alpha < 1)

Before capacity

.. math:: CongestedTime_{i} = FreeFlowTime_{i} * \frac{1.1- (\alpha *\frac{Volume_{i}}{Capacity_{i}})}{1.1-\frac{Volume_{i}}{Capacity_{i}}}

and after capacity

.. math:: CongestedTime_{i} = FreeFlowTime_{i} * \frac{1.1- \alpha}{0.1} * (\frac{Volume_{i}}{Capacity_{i}})^2

More functions will be added as needed/requested/possible.

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
    >>> transit = Transit(project)
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
