Path-finding and assignment mechanics
-------------------------------------

Performing traffic assignment, or even just computing paths through a network is
always a little different in each platform, and in AequilibraE is not different.

The complexity in computing paths through a network comes from the fact that
transportation models usually house networks for multiple transport modes, so
the toads (links) available for a passenger car may be different than those available
for a heavy truck, as it happens in practice.

For this reason, all path computation in AequilibraE happens through **Graph** objects.
While users can operate models by simply selecting the mode they want AequilibraE to
create graphs for, **Graph** objects can also be manipulated in memory or even created
from networks that are :ref:`NOT housed inside an AequilibraE model <plot_assignment_without_model>`.

.. _aequilibrae-graphs:

AequilibraE Graphs
~~~~~~~~~~~~~~~~~~

As mentioned above, AequilibraE's graphs are the backbone of path computation,
skimming and Traffic Assignment. Besides handling the selection of links available to
each mode in an AequilibraE model, **Graphs** also handle the existence of bi-directional
links with direction-specific characteristics (e.g. speed limit, congestion levels, tolls,
etc.).

The **Graph** object is rather complex, but the difference between the physical links and
those that are available two class member variables consisting of Pandas DataFrames, the
***network** and the **graph**.

.. code-block:: python

    from aequilibrae.paths import Graph

    g = Graph()

    # g.network
    # g.graph

Directionality
^^^^^^^^^^^^^^

Links in the Network table (the Pandas representation of the project's *Links* table) are
potentially bi-directional, and the directions allowed for traversal are dictated by the
field *direction*, where -1 and 1 denote only BA and AB traversal respectively and 0 denotes
bi-directionality.

Direction-specific fields must be coded in fields **_AB** and **_BA**, where the name of
the field in the *graph* will be equal to the prefix of the directional fields. For example:

The fields **free_flow_travel_time_AB** and **free_flow_travel_time_BA** provide the same
metric (*free_flow_travel_time*) for each of the directions of a link, and the field of
the graph used to set computations (e.g. field to minimize during path-finding, skimming,
etc.) will be **free_flow_travel_time**.

Graphs from a model
^^^^^^^^^^^^^^^^^^^

Building graphs directly from an AequilibraE model is the easiest option for beginners
or when using AequilibraE in anger, as much of the setup is done by default.

.. code-block:: python

    from aequilibrae import Project

    project = Project.from_path("/tmp/test_project")
    project.network.build_graphs(modes=["c"]) # We build the graph for cars only

    graph = project.network.graphs['c'] # we grab the graph for cars

Manipulating graphs in memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As mentioned before, the AequilibraE Graph can be manipulated in memory, with all its
components available for editing.  One of the simple tools available directly in the
API is a method call for excluding one or more links from the Graph, **which is done**
**in place**.

.. code-block:: python

    graph.exclude_links([123, 975])

More sophisticated graph editing is also possible, but it is recommended that
changes to be made in the network DataFrame. For example:

.. code-block:: python

    graph.network.loc[graph.network.link_type="highway", "speed_AB"] = 100
    graph.network.loc[graph.network.link_type="highway", "speed_BA"] = 100

    graph.prepare_graph(graph.centroids)
    if graph.skim_fields:
        graph.set_skimming(graph.skim_fields)

Skimming settings
^^^^^^^^^^^^^^^^^
Skimming the field of a graph when computing shortest path or performing
traffic assignment must be done by setting the skimming fields in the
**Graph** object, and there are no limits (other than memory) to the number
of fields that can be skimmed.


.. code-block:: python

    graph.set_skimming(["tolls", "distance", "free_flow_travel_time"])

Setting centroids
^^^^^^^^^^^^^^^^^

Like other elements of the AequilibraE **Graph**, the user can also manipulate the
set of nodes interpreted by the software as centroids in the **Graph** itself.
This brings the advantage of allowing the user to perform assignment of partial
matrices, matrices of travel between arbitrary network nodes and to skim the network
for an arbitrary number of centroids in parallel, which can be useful when using
AequilibraE as part of more general analysis pipelines. As seen above, this is also
necessary when the network has been manipulated in memory.

When setting regular network nodes as centroids, the user should take care in
not blocking flows through "centroids".

.. code-block:: python

    graph.prepare_graph(np.array([13, 169, 2197, 28561, 371293], np.int))
    graph.set_blocked_centroid_flows(False)

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

Traffic assignment is organized within a object introduces on version 0.6.1 of the
AequilibraE, and includes a small list of member variables which should be populated
by the user, providing a complete specification of the assignment procedure:

* **classes**:  List of objects :ref:`assignment_class_object` , each of which
  are a completely specified traffic class

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
:ref:`parameters_file` .

There are also some strict technical requirements for formulating the
multi-class equilibrium assignment as an unconstrained convex optimization problem,
as we have implemented it. These requirements are loosely listed in
:ref:`technical_requirements_multi_class` .

If you want to see the assignment log on your terminal during the assignment,
please look in the :ref:`logging to terminal <logging_to_terminal>` example.

To begin building the assignment it is easy:

.. code-block:: python

    from aequilibrae.paths import TrafficAssignment

    assig = TrafficAssignment()

Volume Delay Function
^^^^^^^^^^^^^^^^^^^^^

For now, the only VDF functions available in AequilibraE are the

* BPR [3]_

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

    assig.set_vdf('BPR')

The implementation of the VDF functions in AequilibraE is written in Cython and
fully multi-threaded, and therefore descent methods that may evaluate such
function multiple times per iteration should not become unecessarily slow,
especially in modern multi-core systems.

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

Example:

.. code-block:: python

  tc = TrafficClass("car", graph_car, matrix_car)

  tc2 = TrafficClass("truck", graph_truck, matrix_truck)

* **pce** - The passenger-car equivalent is the standard way of modeling
  multi-class traffic assignment equilibrium in a consistent manner (see [1]_ for
  the technical detail), and it is set to 1 by default. If the **pce** for a
  certain class should be different than one, one can make a quick method call.

* **fixed_cost** - In case there are fixed costs associated with the traversal of
  links in the network, the user can provide the name of the field in the graph
  that contains that network.

* **vot** - Value-of-Time (VoT) is the mechanism to bring time and monetary
  costs into a consistent basis within a generalized cost function.in the event
  that fixed cost is measured in the same unit as free-flow travel time, then
  **vot** must be set to 1.0, and can be set to the appropriate value (1.0,
  value-of-timeIf the **vot** or whatever conversion factor is appropriate) with
  a method call.


.. code-block:: python

  tc2.set_pce(2.5)
  tc2.set_fixed_cost("truck_toll")
  tc2.set_vot(0.35)

To add traffic classes to the assignment instance it is just a matter of making
a method call:

.. code-block:: python

  assig.set_classes([tc, tc2])

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

To choose a field that exists in the graph, we just pass the parameters as
follows:

.. code-block:: python

  assig.set_vdf_parameters({"alpha": "alphas", "beta": "betas"})


To pass global values, it is simply a matter of doing the following:

.. code-block:: python

  assig.set_vdf_parameters({"alpha": 0.15, "beta": 4})


Setting final parameters
^^^^^^^^^^^^^^^^^^^^^^^^

There are still three parameters missing for the assignment.

* Capacity field

* Travel time field

* Equilibrium algorithm to use

.. code-block:: python

  assig.set_capacity_field("capacity")
  assig.set_time_field("free_flow_time")
  assig.set_algorithm(algorithm)


Setting Public Transport Preload
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can also optionally include a preload vector for constant flows from public transport 
which are not being otherwise modelled. This accounts for 
the predetermined public transport vehicles at their regularly scheduled times,
and reduces the capacity on each link accordingly. AequilibraE supports various
conditions for which PT trips to include in the preload, and allows the user to
specify the PCE for each type of vehicle in the public transport network.

To create a preload for public transport vehicles operating between 8am to
10am, do the following:

.. code-block:: python

  # Time period, in seconds from midnight
  start = 8 * 60 * 60
  end = 10 * 60 * 60

  preload = project.network.build_pt_preload(start, end)

Next, set the preload vector in the assignment.

.. code-block:: python

  assig.set_pt_preload(preload)


Executing an Assignment
^^^^^^^^^^^^^^^^^^^^^^^

Finally, one can execute assignment:

.. code-block:: python

  assig.execute()

:ref:`convergence_criteria` is discussed in a different section.

References
~~~~~~~~~~
.. [1] Zill, J., Camargo, P., Veitch, T., Daisy,N. (2019) "Toll Choice and Stochastic User Equilibrium: 
       Ticking All the Boxes", Transportation Research Record, 2673(4):930-940. 
       Available in: https://doi.org/10.1177%2F0361198119837496

.. [2] Spiess H. (1990) "Technical Note—Conical Volume-Delay Functions."Transportation Science, 24(2): 153-158.
       Available in: https://doi.org/10.1287/trsc.24.2.153

.. [3] Hampton Roads Transportation Planning Organization, Regional Travel Demand Model V2 (2020). 
       Available in: https://www.hrtpo.org/uploads/docs/2020_HamptonRoads_Modelv2_MethodologyReport.pdf