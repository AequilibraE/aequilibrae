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

The implementation of Frank-Wolfe in AequilibraE is extremely simple from an
implementation point of view, as we use a generic optimizer from SciPy as an
engine for the line search.

Implementation details & tricks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A few implementation details and tricks are worth mentioning not because it is
needed to use the software, but because they were things we grappled with during
implementation, and it would be a shame not register it for those looking to
implement their own variations of this algorithm or to slight change it for
their own purposes.

* The relative gap is computed with the cost used to compute the All-or-Nothing
  portion of the iteration, and although the literature on this is obvious, we
  took some time to realize that we should re-compute the travel costs only
  **AFTER** checking for convergence.

* In some instances, Frank-Wolfe is extremely unstable during the first
  iterations on assignment, resulting on numerical errors on our line search.
  We found that setting the step size to the corresponding MSA value (1/
  current iteration) resulted in the problem quickly becoming stable and moving
  towards a state where the line search started working properly.

Conjugate Frank-Wolfe
+++++++++++++++++++++


Biconjugate Frank-Wolfe
+++++++++++++++++++++++

Opportunities for multi-threading
+++++++++++++++++++++++++++++++++

Most multi-threading opportunities have already been taken advantage of during
the implementation of the All-or-Nothing portion of the assignment. However, the
optimization engine using for line search, as well as a few functions from NumPy
could still be paralellized for maximum performance on system with high number
of cores, such as the latest Threadripper CPUs.  These numpy functions are the
following:

* np.sum
* np.power
* np.fill


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

Numerical Study
---------------
Similar to other complex algorthms that handle a large amount of data through
complex computations, traffic assignment procedures can always be subject to at
least one very reasonable question:  Are the results right?

For this reason, we have used all equilibrium traffic assignment algorithms
available in AequilibraE to solve standard instances used in academia for
comparing algorithm results, some of which have are available with highly
converged solutions (~1e-14):
`<https://github.com/bstabler/TransportationNetworks/>`_

Sioux Falls
~~~~~~~~~~~~

Network has:

* Links: 76
* Nodes: 24
* Zones: 24

.. image:: images/sioux_falls_msa-500_iter.png
    :width: 590
    :alt: Sioux Falls MSA 500 iterations
.. image:: images/sioux_falls_frank-wolfe-500_iter.png
    :width: 590
    :alt: Sioux Falls Frank-Wolfe 500 iterations
.. image:: images/sioux_falls_regional_bfw-500_iter.png
    :width: 590
    :alt: Sioux Falls Biconjugate Frank-Wolfe 500 iterations

Anaheim
~~~~~~~

Network has:

* Links: 914
* Nodes: 416
* Zones: 38

.. image:: images/anaheim_msa-500_iter.png
    :width: 590
    :alt: Anaheim MSA 500 iterations
.. image:: images/anaheim_frank-wolfe-500_iter.png
    :width: 590
    :alt: Anaheim Frank-Wolfe 500 iterations
.. image:: images/Anaheim_regional_bfw-500_iter.png
    :width: 590
    :alt: Anaheim Biconjugate Frank-Wolfe 500 iterations

Winnipeg
~~~~~~~~

Network has:

* Links: 914
* Nodes: 416
* Zones: 38

.. image:: images/winnipeg_msa-500_iter.png
    :width: 590
    :alt: Winnipeg MSA 500 iterations
.. image:: images/winnipeg_frank-wolfe-500_iter.png
    :width: 590
    :alt: Winnipeg Frank-Wolfe 500 iterations

Barcelona
~~~~~~~~~

Network has:

* Links: 2,522
* Nodes: 1,020
* Zones: 110

.. image:: images/barcelona_msa-500_iter.png
    :width: 590
    :alt: Barcelona MSA 500 iterations
.. image:: images/barcelona_frank-wolfe-500_iter.png
    :width: 590
    :alt: Barcelona Frank-Wolfe 500 iterations

Chicago Regional
~~~~~~~~~~~~~~~~

Network has:

* Links: 2,522
* Nodes: 1,020
* Zones: 110

.. image:: images/chicago_regional_msa-500_iter.png
    :width: 590
    :alt: Chicago MSA 500 iterations
.. image:: images/chicago_regional_frank-wolfe-500_iter.png
    :width: 590
    :alt: Chicago Frank-Wolfe 500 iterations
.. image:: images/chicago_regional_bfw-500_iter.png
    :width: 590
    :alt: Chicago Biconjugate Frank-Wolfe 500 iterations


Convergence Study
---------------

Besides validating the final results from the algorithms, we have also compared
how well they converge for the largest instance we have tested (Chicago
Regional), as that instance has a comparable size to real-world models.

.. image:: images/convergence_comparison.png
    :width: 590
    :alt: Algorithm convergence comparison
.. image:: images/convergence_comparison.png
    :width: 590
    :alt: Algorithm convergence comparison


Not surprinsingly, one can see that Frank-Wolfe far outperforms the Method of
Successive Averages for a number of iterations larger than 25, and is capable of
reaching 1.0e-04 just after 800 iterations, while MSA is still at 3.5e-4 even
after 1,000 iterations.

Computational performance
-------------------------
Running on a Thinkpad X1 extreme equipped with a 6 cores 9750H CPU and 32Gb of
2667Hz RAM, AequilibraE performed 1,000 iterations of Frank-Wolfe assignment
on the Chicago Network in just under 46 minutes, while Biconjugate Frank Wolfe
takes just under 47 minutes.

During this process, the sustained CPU clock fluctuated between 3.05 and 3.2GHz,
which suggests that performance in modern desktops would be substantially better
