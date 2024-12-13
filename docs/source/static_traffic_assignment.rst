:orphan:

Static Traffic Assignment
=========================

Performing traffic assignment or computing paths through a network is always a little different in 
each platform, and in AequilibraE is no exception, but we strive to make the static traffic assignment 
process as simple as possible so that seasoned modelers can easily migrate their models and workflows 
to the platform.

Although modeling with AequilibraE should feel somewhat familiar to seasoned modelers, especially
those used to programming, the mechanics of traffic assignment in AequilibraE might be foreign to
some users, so this section of the documentation will include discussions of the mechanics
of some of these procedures and some light discussion on its motivation.

Path computation
----------------

The complexity in computing paths through a network comes from the fact that transportation models 
usually house networks for multiple transport modes, so the loads (links) available for a passenger 
car may be different than those available for a heavy truck, as it happens in practice.

For this reason, all path computation in AequilibraE happens through ``Graph`` objects. While users 
can operate models by simply selecting the mode they want AequilibraE to create graphs for, ``Graph`` 
objects can also be manipulated in memory or even created from networks that are 
:ref:`NOT housed inside an AequilibraE model <plot_assignment_without_model>`.

Traffic Assignment Procedure
----------------------------

Along with a network data model, traffic assignment is the most technically challenging portion to 
develop in a modeling platform, especially if you want it to be *fast*. In AequilibraE, we aim to 
make it as fast as possible, without making it overly complex to use, develop and maintain, although
we know that *complex* is subjective.

Running traffic assignment in AequilibraE consists in creating the traffic classes that are going
to be assigned, add them to a traffic assignment object, set the traffic assignment parameters, and
run the assignment.

``TrafficClass``
~~~~~~~~~~~~~~~~

The ``TrafficClass`` object holds all the information pertaining to a specific traffic class to be 
assigned. There are three pieces of information that are required in the instantiation of this class:

* **name**: name of the class. It has to be unique among all classes used in a multi-class traffic assignment

* **graph**: it is the ``Graph`` object corresponding to that particular traffic class/mode

* **matrix**: it is the AequilibraE matrix with the demand for that traffic class, which can have
  an arbitrary number of user-classes setup as different layers (cores) of the matrix object.

.. code-block:: python

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

.. code-block:: python

    >>> tc_truck.set_pce(2.5)

* **Fixed costs**: in case there are fixed costs associated with the traversal of links in the network, the user 
  can provide the name of the field in the graph that contains that network.

.. code-block:: python

    >>> tc_truck.set_fixed_cost("distance")

* **Value-of-Time** (VoT) is the mechanism to bring time and monetary costs into a consistent basis 
  within a generalized cost function. In the event that fixed cost is measured in the same unit as free-flow travel
  time, then *vot* must be set to 1.0.

.. code-block:: python

    >>> tc_truck.set_vot(0.35)

``TrafficAssignment``
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    >>> from aequilibrae.paths import TrafficAssignment

    >>> assig = TrafficAssignment()

AequilibraE's traffic assignment is organized within an object with the same name which contains a series of
member variables that should be populated by the user, providing thus a complete specification of the assignment
procedure.

* **classes**: list of completely specified traffic classes

.. code-block:: python

    # You can add one or more traffic classes to the assignment instance
    >>> assig.add_class(tc_truck) # doctest: +SKIP

    >>> assig.set_classes([tc_car, tc_truck])

* **vdf**: the volume-delay function (VDF) to be used, being one of ``BPR``, ``BPR2``, ``CONICAL``, or ``INRETS``

.. code-block:: python

    >>> assig.set_vdf('BPR')

* **vdf_parameters**: the parameters to be used in the volume-delay function, other than volume, capacity and 
  free-flow time. VDF parameters must be consistent across all graphs.

  Because AequilibraE supports different parameters for each link, its implementation is the most general possible 
  while still preserving the desired properties for multi-class assignment, but the user needs to provide individual
  values for each link *OR* a single value for the entire network.

  Setting the VDF parameters should be done *AFTER* setting the VDF function of choice and adding traffic classes 
  to the assignment, or it will *fail*.

.. code-block:: python

    # The VDF parameters can be either an existing field in the graph, passed as a parameter:
    >>> assig.set_vdf_parameters({"alpha": "b", "beta": "power"}) # doctest: +SKIP

    # Or as a global value:
    >>> assig.set_vdf_parameters({"alpha": 0.15, "beta": 4})

* **time_field**: the field of the graph that corresponds to free-flow travel time. The procedure will 
  collect this information from the graph associated with the first traffic class provided, but will check 
  if all graphs have the same information on free-flow travel time
  
.. code-block:: python

    >>> assig.set_time_field("free_flow_time")

* **capacity_field**: the field of the graph that corresponds to the link capacity. The procedure will collect 
  this information from the graph associated with the first traffic class provided, but will check if all graphs
  have the same information on capacity

.. code-block:: python

    >>> assig.set_capacity_field("capacity")

* **algorithm**: the assignment algorithm to be used, being one of ``all-or-nothing``, ``bfw``, ``cfw``, ``fw``,  
  ``franke-wolfe``, or ``msa``.

.. code-block:: python

    >>> assig.set_algorithm("bfw")

Volume-delay function
^^^^^^^^^^^^^^^^^^^^^

For now, the only VDF functions available in AequilibraE are

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
~~~~~~~~~~~~~~~~

We can also optionally include a preload vector for constant flows which are not being otherwise modelled. 
For example, this can be used to account for scheduled  public transport vehicles, adding an equivalent 
load to each link along the route accordingly. AequilibraE supports various conditions for which PT trips 
to include in the preload, and allows the user to specify the PCE for each type of vehicle in the public transport 
network.

To create a preload for public transport vehicles operating between 8 AM to 10 AM, do the following:

.. code-block:: python

    >>> from aequilibrae.transit import Transit

    # Times are specified in seconds from midnight
    >>> transit = Transit(project)
    >>> preload = transit.build_pt_preload(start=8*3600, end=10*3600)

    # Add the preload to the assignment
    >>> assig.add_preload(preload, 'PT_vehicles') # doctest: +SKIP

Executing an Assignment
~~~~~~~~~~~~~~~~~~~~~~~

Finally, run traffic assignment!

.. code-block:: python

  >>> assig.execute()

.. [1] Hampton Roads Transportation Planning Organization, Regional Travel Demand Model V2 (2020). 
       Available in: https://www.hrtpo.org/uploads/docs/2020_HamptonRoads_Modelv2_MethodologyReport.pdf

.. [2] Spiess H. (1990) "Technical Note—Conical Volume-Delay Functions."Transportation Science, 24(2): 153-158.
       Available in: https://doi.org/10.1287/trsc.24.2.153

.. [3] Zill, J., Camargo, P., Veitch, T., Daisy,N. (2019) "Toll Choice and Stochastic User Equilibrium: 
       Ticking All the Boxes", Transportation Research Record, 2673(4):930-940. 
       Available in: https://doi.org/10.1177%2F0361198119837496
