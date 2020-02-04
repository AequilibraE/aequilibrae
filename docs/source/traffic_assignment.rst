.. _traffic_assignment:

Traffic Assignment
==================

Along with a network data model, traffic assignment is the most technically
challenging portion to develop in a modeling platform, especially if you want it
to be **FAST**. In AequilibraE, we aim to make it as fast as possible, without
making it overly complex to use, develop and maintain (we know how subjective
*complex* is).

.. note::
   AequilibraE has had efficient multi-threaded All-or-Nothing (AoN) assignment
   for a while, but since the Method-of-Successive-Averages, Frank-Wolfe,
   Conjugate-Frank-Wolfe and Biconjugate-Frank-Wolfe are new in the software, it
   should take some time for these implementations to reach full maturity



Traffic assignment
------------------

Traffic assignment is organized within a object new to version 0.6.1 that
includes a small list of member variables which should be populated by the user,
providing a complete specification of the assignment procedure:

* **classes**:  List of :ref:`assignment_class_object`
* **algorithm**: The assignment algorithm to be used. e.g. "all-or-nothing" or "bfw"
* **vdf_parameters**: The Volume delay function to be used
* **time_field**: The field of the graph that corresponds to **free-flow**
  **travel time**. The procedure will collect this information from the graph
  associated with the first traffic class provided, but will check if all graphs
  have the same information on free-flow travel time
* **capacity_field**:

Assignment paramters such as maximum number of iterations and target relative
gap come from the global software parameters, that can be set using the
:ref:`example_usage_parameters`

There are also some strict technical requirements for multi-class equilibrium
assignment, which listed in :ref:`_technical_requirements_multi_class`

.. _assignment_class_object:
Assignment class object
~~~~~~~~~~~~~~~~~~~~~~~

Need to describe the assignment class object (graph, matrix, results)


Assignment specifications
+++++++++++++++++++++++++

Need to create this guy.  Will hold

* VDF to be used (e.g. "BPR")
* Name of the fields from the graph to use for vdf parameters (e.g. {"alpha": "a", "beta": "b"}), or
  the individual values to use if that is the case


Volume delay functions
++++++++++++++++++++++

For now, only the traditional BPR is available for assignment using AequilibraE.

Parameters for VDF functions can be passed as a fixed value to use for all links,
or as graph fields. As it is the case for the travel time and capacity fields,
VDF parameters need to be consistent across all graphs.

.. We need something on VDFs here, more specifically on how they work


Multi-class Equilibrium assignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By introducing equilibrium assignment [1] with as many algorithms as we have, it
becomes necessary to also introduce multi-class assignment, which goes along
with the pre-existing capability of assigning multiple user-classes without
having to compute the same paths set multiple times.

.. _technical_requirements_multi_class:

Technical requirements
++++++++++++++++++++++

- Identical free-flow travel time for all links
- Unique Passenger Car Equivalency (PCE) for each class
- Monotonically increasing volume-delay functions
- Differentiable volume-delay functions

For a mathematically strict discussion, see
`Zill et all <https://doi.org/10.1177%2F0361198119837496>`_


Method of successive Averages (MSA)
+++++++++++++++++++++++++++++++++++

Frank-Wolfe (FW)
++++++++++++++++

Conjugate Frank-Wolfe
+++++++++++++++++++++


Biconjugate Frank-Wolfe
+++++++++++++++++++++++

References
++++++++++

[1] Wardrop J. G. (1952) "Some theoretical aspects of road traffic research."
Proc. Inst. Civil Eng. 1 Part II, pp.325-378.

[2] LeBlanc L. J., Morlok E. K. and Pierskalla W. P. (1975) "An efficient
approach to solving the road network equilibrium traffic assignment problem"
Transpn Res. 9, 309-318.

[3] Maria Mitradjieva and Per Olov Lindberg "The Stiff Is Moving—Conjugate
Direction Frank-Wolfe Methods with Applications to Traffic Assignment",
`Transportation Science 2013 47:2, 280-293 <https://doi.org/10.1287/trsc.1120.0409>`_



Handling the network
--------------------
The under the hood

Super-network
~~~~~~~~~~~~~
We deal with a super-network by having all classes with the same links in their
sub-graphs, but assigning b_node identical to a_node for all links whenever a
link is not available for a certain user class.
It is slightly less efficient when we are computing shortest paths, but a LOT
more efficient when we are aggregating flows.

The Graph class
~~~~~~~~~~~~~~~

Graph format remains the same, but should describe it well

* free-flow time
*