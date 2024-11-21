Traffic Assignment Procedure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Along with a network data model, traffic assignment is the most technically
challenging portion to develop in a modeling platform, especially if you want it
to be **FAST**. In AequilibraE, we aim to make it as fast as possible, without
making it overly complex to use, develop and maintain (we know *complex* is
subjective).

Below we detail the components that go into performing traffic assignment, but for
a comprehensive use case for the traffic assignment module, please see the complete
application in :ref:`this example <example_usage_forecasting>`.

Traffic Assignment Class
^^^^^^^^^^^^^^^^^^^^^^^^

Traffic assignment is organized within a object introduced on version 0.6.1 of
AequilibraE, and includes a small list of member variables which should be populated
by the user, providing a complete specification of the assignment procedure:

* **classes**:  List of objects :ref:`assignment_class_object`, each of which
  are a completely specified Traffic Class

* **vdf**: The Volume delay function (VDF) to be used

* **vdf_parameters**: The parameters to be used in the volume delay function,
  other than volume, capacity and free flow time

* **time_field**: The field of the graph that corresponds to **free-flow**
  **travel time**. The procedure will collect this information from the graph
  associated with the first traffic class provided, but will check if all graphs
  have the same information on free-flow travel time

* **capacity_field**: The field of the graph that corresponds to **link**
  **capacity**. The procedure will collect this information from the graph
  associated with the first traffic class provided, but will check if all graphs
  have the same information on capacity

* **algorithm**: The assignment algorithm to be used. (e.g. "all-or-nothing", "bfw")

Assignment parameters such as maximum number of iterations and target relative
gap come from the global software parameters, that can be set using the
:ref:`parameters_file`.

There are also some strict technical requirements for formulating the
multi-class equilibrium assignment as an unconstrained convex optimization problem,
as we have implemented it. These requirements are loosely listed in
:ref:`technical_requirements_multi_class` .

If you want to see the assignment log on your terminal during the assignment,
please look in the :ref:`logging to terminal <logging_to_terminal>` example.

To begin building the assignment it is easy:

.. code-block:: python

    >>> from aequilibrae.paths import TrafficAssignment

    >>> project = create_example(project_path)

    >>> assig = TrafficAssignment()

    # Set the assignment parameters
    >>> assig.max_iter = 10
    >>> assig.rgap_target = 0.01

.. seealso::

    * :func:`aequilibrae.paths.TrafficAssignment`
        Class documentation

.. _assignment_class_object:

Traffic class
^^^^^^^^^^^^^

The Traffic class object holds all the information pertaining to a specific
traffic class to be assigned. There are three pieces of information that are
required in the instantiation of this class:

* **name** - Name of the class. Unique among all classes used in a multi-class
  traffic assignment

* **graph** - It is the Graph object corresponding to that particular traffic class/
  mode

* **matrix** - It is the AequilibraE matrix with the demand for that traffic class,
  but which can have an arbitrary number of user-classes, setup as different
  layers of the matrix object

.. code-block:: python

    >>> from aequilibrae.paths import TrafficClass

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

One can also edit some information related to the passenger-car equivalent, the fixed cost, 
or the value of time for each traffic class.

* **pce** - The passenger-car equivalent is the standard way of modeling
  multi-class traffic assignment equilibrium in a consistent manner (see [3]_ for
  the technical detail), and it is sevalue# doctest: +SKIPt to 1.0 by default. If the **pce** for a
  certain class should be different than 1.0, one can make a quick method call
  to set the appropriate value.

* **fixed_cost** - In case there are fixed costs associated with the traversal of
  links in the network, the user can provide the name of the field in the graph
  that contains that network.

* **vot** - Value-of-Time (VoT) is the mechanism to bring time and monetary
  costs into a consistent basis within a generalized cost function. In the event
  that fixed cost is measured in the same unit as free-flow travel time, then
  **vot** must be set to 1.0. If the **vot** for a certain class should be different
  than 1.0, one can make a quick method call to set the appropriate value.

.. code-block:: python

    >>> tc_truck.set_pce(2.5)
    >>> tc_truck.set_fixed_cost("distance")
    >>> tc_truck.set_vot(0.35)

Traffic classes must be assigned to a Traffic Assignment instance:

.. code-block:: python

    # You can add one or more traffic classes to the assignment instance
    >>> assig.add_class(tc_truck) # doctest: +SKIP

    >>> assig.set_classes([tc_car, tc_truck])

.. seealso::

    * :func:`aequilibrae.paths.TrafficClass`
        Class documentation

Volume Delay Function
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

Setting the volume delay function is one of the first things you should do after
instantiating an assignment problem in AequilibraE, and it is as simple as:

.. code-block:: python

    >>> assig.set_vdf('BPR')

The implementation of the VDF functions in AequilibraE is written in Cython and
fully multi-threaded, and therefore descent methods that may evaluate such
function multiple times per iteration should not become unecessarily slow,
especially in modern multi-core systems.

Setting VDF Parameters
^^^^^^^^^^^^^^^^^^^^^^

Parameters for VDF functions can be passed as a fixed value to use for all
links, or as graph fields. As it is the case for the travel time and capacity
fields, VDF parameters need to be consistent across all graphs.

Because AequilibraE supports different parameters for each link, its
implementation is the most general possible while still preserving the desired
properties for multi-class assignment, but the user needs to provide individual
values for each link **OR** a single value for the entire network.

Setting the VDF parameters should be done **AFTER** setting the VDF function of
choice and adding traffic classes to the assignment, or it will **fail**.

.. code-block:: python

    # The VDF parameters can be either an existing field in the graph, passed
    # as a parameter:
    >>> assig.set_vdf_parameters({"alpha": "b", "beta": "power"}) # doctest: +SKIP

    # Or as a global value:
    >>> assig.set_vdf_parameters({"alpha": 0.15, "beta": 4})

.. seealso::

    * :func:`aequilibrae.paths.VDF`
        Class documentation

Setting final parameters
^^^^^^^^^^^^^^^^^^^^^^^^

There are still three parameters missing for the assignment.

* Capacity field
* Travel time field
* Equilibrium algorithm to use

.. code-block:: python

  >>> assig.set_capacity_field("capacity")
  >>> assig.set_time_field("free_flow_time")
  >>> assig.set_algorithm("bfw")

Setting Preloads
^^^^^^^^^^^^^^^^

We can also optionally include a preload vector for constant flows which are not
being otherwise modelled. For example, this can be used to account for scheduled 
public transport vehicles, adding an equivalent load to each link along the route accordingly.
AequilibraE supports various conditions for which PT trips to include in the preload, 
and allows the user to specify the PCE for each type of vehicle in the public transport 
network.

To create a preload for public transport vehicles operating between 8am to
10am, do the following:

.. code-block:: python

    >>> from aequilibrae.transit import Transit

    # Times are specified in seconds from midnight
    >>> transit = Transit(project)
    >>> preload = transit.build_pt_preload(start=8*3600, end=10*3600)

    # Add the preload to the assignment
    >>> assig.add_preload(preload, 'PT_vehicles') # doctest: +SKIP

Executing an Assignment
^^^^^^^^^^^^^^^^^^^^^^^

Finally, one can execute assignment:

.. code-block:: python

  >>> assig.execute()

:ref:`convergence_criteria` is discussed in a different section.

.. [1] Hampton Roads Transportation Planning Organization, Regional Travel Demand Model V2 (2020). 
       Available in: https://www.hrtpo.org/uploads/docs/2020_HamptonRoads_Modelv2_MethodologyReport.pdf

.. [2] Spiess H. (1990) "Technical Note—Conical Volume-Delay Functions."Transportation Science, 24(2): 153-158.
       Available in: https://doi.org/10.1287/trsc.24.2.153

.. [3] Zill, J., Camargo, P., Veitch, T., Daisy,N. (2019) "Toll Choice and Stochastic User Equilibrium: 
       Ticking All the Boxes", Transportation Research Record, 2673(4):930-940. 
       Available in: https://doi.org/10.1177%2F0361198119837496