.. _multiclass_equilibrium:

Multi-class Equilibrium assignment
----------------------------------

While single-class equilibrium traffic assignment [1]_ is mathematically simple,
multi-class traffic assignment [7]_, especially when including monetary costs
(e.g. tolls) and multiple classes with different Passenger Car Equivalent (PCE)
factors, requires more sophisticated mathematics.

As it is to be expected, strict convergence of multi-class equilibrium assignments
comes at the cost of specific technical requirements and more advanced equilibration
algorithms have slightly different requirements.

Cost function
~~~~~~~~~~~~~

AequilibraE supports class-=specific cost functions, where each class can include
the following:

* PCE
* Link-based fixed financial cost components
* Value-of-Time (VoT)

.. _technical_requirements_multi_class:

Technical requirements
~~~~~~~~~~~~~~~~~~~~~~

This documentation is not intended to discuss in detail the mathematical
requirements of multi-class traffic assignment, which can be found discussed in
detail on [4]_.

A few requirements, however, need to be made clear.

* All traffic classes shall have identical free-flow travel times throughout the
  network

* Each class shall have an unique Passenger Car Equivalency (PCE) factor for all links

* Volume delay functions shall be monotonically increasing. *Well behaved*
  functions are always something we are after

For the conjugate and Biconjugate Frank-Wolfe algorithms it is also necessary
that the VDFs are differentiable.

.. _convergence_criteria:

Convergence criteria
~~~~~~~~~~~~~~~~~~~~

Convergence in AequilibraE is measured solely in terms of relative gap, which is
a somewhat old recommendation [5]_, but it is still the most used measure in
practice, and is detailed below.

.. math:: RelGap = \frac{\sum_{a}V_{a}^{*}*C_{a} - \sum_{a}V_{a}^{AoN}*C_{a}}{\sum_{a}V_{a}^{*}*C_{a}}

The algorithm's two stop criteria currently used are the maximum number of
iterations and the target Relative Gap, as specified above. These two parameters
are described in detail in the :ref:`parameters_assignment` section, in the
:ref:`parameters_file`.

Algorithms available
~~~~~~~~~~~~~~~~~~~~

All algorithms have been implemented as a single software class, as the
differences between them are simply the step direction and step size after each
iteration of all-or-nothing assignment, as shown in the table below

+-------------------------------+-----------------------+----------------------------------+
| Algorithm                     | Step direction        | Step size                        |
+===============================+=======================+==================================+
| Method of Successive Avergaes | All-or-Nothing        | Function of the iteration number |
|                               | Assignment (AoN)      |                                  |
+-------------------------------+-----------------------+----------------------------------+
| Frank-Wolfe                   | All-or-Nothing        | Optimal value derived from       |
|                               | Assignment (AoN)      | Wardrop's principle              |
+-------------------------------+-----------------------+----------------------------------+
| Conjugate Frank-Wolfe         | Conjugate direction   | Optimal value derived from       |
|                               | (Current and          | Wardrop's principle              |
|                               | previous AoN)         |                                  |
+-------------------------------+-----------------------+----------------------------------+
| Biconjugate Frank-Wolfe       | Biconjugate direction | Optimal value derived from       |
|                               | (Current and two      | Wardrop's principle              |
|                               | previous AoN)         |                                  |
+-------------------------------+-----------------------+----------------------------------+

.. note::
   Our implementations of the conjudate and Biconjugate-Frank-Wolfe methods
   should be inherently proportional [6]_, but we have not yet carried the
   appropriate testing that would be required for an empirical proof.

Method of Successive Averages (MSA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This algorithm has been included largely for historical reasons, and we see very
little reason to use it. Yet, it has been implemented with the appropriate
computation of relative gap computation and supports all the analysis features
available.

Frank-Wolfe (FW)
^^^^^^^^^^^^^^^^

The implementation of Frank-Wolfe in AequilibraE is extremely simple from an
implementation point of view, as we use a generic optimizer from SciPy as an
engine for the line search, and it is a standard implementation of the algorithm
introduced by LeBlanc in 1975 [2]_.


Conjugate Frank-Wolfe
^^^^^^^^^^^^^^^^^^^^^

The conjugate direction algorithm was introduced in 2013 [3]_, which is quite
recent if you consider that the Frank-Wolfe algorithm was first applied in the
early 1970's, and it was introduced at the same as its Biconjugate evolution,
so it was born outdated.

Biconjugate Frank-Wolfe
^^^^^^^^^^^^^^^^^^^^^^^

The Biconjugate Frank-Wolfe algorithm is currently the fastest converging link-
based traffic assignment algorithm used in practice, and it is the recommended
algorithm for AequilibraE users. Due to its need for previous iteration data,
it **requires more memory** during runtime, but very large networks should still
fit nicely in systems with 16Gb of RAM.

Implementation details & tricks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A few implementation details and tricks are worth mentioning not because they are
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
  towards a state where the line search started working properly. This technique
  was generalized to the conjugate and biconjugate Frank-Wolfe algorithms.

Multi-threaded implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

AequilibraE's All-or-Nothing assignment (the basis of all the other algorithms)
has been parallelized in Python using the threading library, which is possible
due to the work we have done with memory management to release Python's Global
Interpreter Lock.
Other opportunities for parallelization, such as the computation of costs and
its derivatives (required during the line-search optimization step), as well as
all linear combination operations for vectors and matrices have been achieved
through the use of OpenMP in pure Cython code. These implementations can be
cound on a file called *parallel_numpy.pyx* if you are curious to look at.

Much of the gains of going back to Cython to parallelize these functions came
from making in-place computation using previously existing arrays, as the
instantiation of large NumPy arrays can be computationally expensive.

.. _traffic-assignment-references:


Handling the network
~~~~~~~~~~~~~~~~~~~~
The other important topic when dealing with multi-class assignment is to have
a single consistent handling of networks, as in the end there is only physical
network across all modes, regardless of access differences to each mode (e.g. truck
lanes, High-Occupancy Lanes, etc.). This handling is often done with something
called a **super-network**.

Super-network
^^^^^^^^^^^^^
We deal with a super-network by having all classes with the same links in their
sub-graphs, but assigning *b_node* identical to *a_node* for all links whenever a
link is not available for a certain user class.
This approach is slightly less efficient when we are computing shortest paths, but
it gets eliminated when topologically compressing the network for centroid-to-centroid
path computation and it is a LOT more efficient when we are aggregating flows.

The use of the AequilibraE project and its built-in methods to build graphs
ensure that all graphs will be built in a consistent manner and multi-class
assignment is possible.

References
~~~~~~~~~~

Traffic assignment and equilibrium
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. [1] Wardrop J. G. (1952) "Some theoretical aspects of road traffic research."Proceedings of the Institution of Civil Engineers 1952, 1(3):325-362. Available in: https://www.icevirtuallibrary.com/doi/abs/10.1680/ipeds.1952.11259

.. [2] LeBlanc L. J., Morlok E. K. and Pierskalla W. P. (1975) "An efficient approach to solving the road network equilibrium traffic assignment problem". Transportation Research, 9(5):309-318. Available in: https://doi.org/10.1016/0041-1647(75)90030-1

.. [3] Mitradjieva, M. and Lindberg, P.O. (2013) "The Stiff Is Moving—Conjugate Direction Frank-Wolfe Methods with Applications to Traffic Assignment". Transportation Science, 47(2):280-293. Available in: https://doi.org/10.1287/trsc.1120.0409

.. [4] Zill, J., Camargo, P., Veitch, T., Daisy,N. (2019) "Toll Choice and Stochastic User Equilibrium: Ticking All the Boxes", Transportation Research Record, 2673(4):930-940. Available in: https://doi.org/10.1177%2F0361198119837496

.. [5] Rose, G., Daskin, M., Koppelman, F. (1988) "An examination of convergence error in equilibrium traffic assignment models",  Transportation Res. B, 22(4):261-274. Available in: https://doi.org/10.1016/0191-2615(88)90003-3

.. [6] Florian, M., Morosan, C.D. (2014) "On uniqueness and proportionality in multi-class equilibrium assignment", Transportation Research Part B, 70:261-274. Available in: https://doi.org/10.1016/j.trb.2014.06.011

.. [7] Marcotte, P., Patriksson, M. (2007) "Chapter 10 Traffic Equilibrium - Handbooks in Operations Research and Management Science, Vol 14", Elsevier. Editors Barnhart, C., Laporte, G. https://doi.org/10.1016/S0927-0507(06)14010-4
