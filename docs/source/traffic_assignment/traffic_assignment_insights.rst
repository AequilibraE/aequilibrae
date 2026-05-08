.. _multiclass_equilibrium:

Traffic Assignment Insights
==================================

While single-class equilibrium traffic assignment [1]_ is mathematically simple, multi-class traffic 
assignment [2]_, especially when including monetary costs (e.g. tolls) and multiple classes with 
different passenger-car equivalent (PCE) factors, requires more sophisticated mathematics.

As it is to be expected, strict convergence of multi-class equilibrium assignments comes at the cost 
of specific technical requirements and more advanced equilibration algorithms have slightly different 
requirements.

.. _technical_requirements_multi_class:

Technical requirements
----------------------

This documentation is not intended to discuss in detail the mathematical requirements of multi-class 
traffic assignment, which can be found on [3]_.

A few requirements, however, need to be made clear.

* All traffic classes shall have identical free-flow travel times throughout the network

* Each class shall have an unique passenger-car equivalency (PCE) factor for all links

* Volume-delay functions shall be monotonically increasing. *Well behaved* functions are 
  always something we are after

For the conjugate and biconjugate Frank-Wolfe algorithms it is also necessary that the VDFs are differentiable.

Cost function
-------------

AequilibraE supports class-specific cost functions, where each class can include the following:

* Passenger-car equivalent (PCE)
* Link-based fixed financial cost components
* Value-of-Time (VoT)

.. _convergence_criteria:

Convergence criteria
--------------------

Convergence in AequilibraE is *driven* by the classical relative gap (``RelGap``) — a somewhat old
recommendation [4]_, but still the most used measure in practice — but several additional
convergence metrics are *reported* alongside it on every iteration. They are all written to the
``convergence_report`` attribute of the assignment and printed to the iteration log; only
``RelGap`` is compared against the stopping criterion. The additional metrics, equivalent by those
available in commercial packages, give complementary views of how close the assignment is
to user equilibrium and are particularly useful when comparing AequilibraE against other packages
or when diagnosing slow convergence.

The algorithm's two stop criteria currently used are the maximum number of iterations and the
target ``RelGap``. These two parameters are described in detail in the
:ref:`parameters_assignment` section, in the :ref:`parameters_file`.

Notation used below: :math:`x_a` is the flow on link ``a`` at the current iteration,
:math:`y_a` is the all-or-nothing (AON) flow on link ``a`` at the current congested cost,
:math:`s_a` is the BFW combined step direction flow on link ``a``,
:math:`c_a(x_a)` is the congested cost on link ``a`` at the current flow,
:math:`D` is the total assigned demand (PCE-adjusted, summed across classes), and
:math:`Z(x) = \sum_{a} \int_0^{x_a} c_a(s)\, ds` is the Beckmann objective.

Relative gap (``rgap``) — used for stopping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math:: \mathit{RelGap} = \frac{\left| \sum_{a} x_{a} \cdot c_{a}(x_{a}) - \sum_{a} y_{a} \cdot c_{a}(x_{a}) \right|}{\sum_{a} x_{a} \cdot c_{a}(x_{a})}

This is the classical Wardrop-style measure of how close the current flow assignment is to the
result of an all-or-nothing assignment on the current congested costs. **This is the only quantity
compared against the user-supplied target gap**; the other metrics below are reported but never
trigger termination.

Direction-based gap (``rgap_direction``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math:: \mathit{rgap\_direction} = \frac{\sum_{a} x_{a} \cdot c_{a}(x_{a}) - \sum_{a} s_{a} \cdot c_{a}(x_{a})}{\sum_{a} x_{a} \cdot c_{a}(x_{a})}

A variant of the relative gap that uses the BFW combined step direction
:math:`s_a = \beta_0\, y_a + \beta_1\, s^{k-1}_a + \beta_2\, s^{k-2}_a` instead of the pure AON
flow in the numerator. Because :math:`s` is a convex combination that includes contributions from
the current flow's history, this measure is systematically smaller than ``RelGap`` and is the
quantity reported by some other packages under the same name "relative gap". It is provided for
direct comparison.

Normalized gap (``normalized_gap``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math:: \mathit{NormGap} = \frac{\sum_{a} x_{a} \cdot c_{a}(x_{a}) - \sum_{a} y_{a} \cdot c_{a}(x_{a})}{D}

The absolute gap divided by the total assigned demand. Equivalent to the difference between the
average trip cost on the network and the average trip cost on the current shortest paths,
expressed in cost units per assigned vehicle. Useful for cross-network comparison because it has
the same units regardless of network size.

Beckmann objective and lower bounds (``objective``, ``lower_bound``, ``best_lower_bound``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math:: \mathit{Objective} = Z(x) = \sum_{a} \int_0^{x_a} c_a(s)\, ds

The value of the Beckmann objective at the current iterate. Because :math:`x` is feasible, this is
always an *upper bound* on the unknown user-equilibrium objective :math:`Z^{*}`. AequilibraE
computes :math:`Z(x)` link-by-link in closed form for each of the supported VDFs (BPR, BPR2,
Conical, INRETS).

.. math:: \mathit{LowerBound} = Z(x) + \sum_{a} c_{a}(x_{a}) \cdot (y_{a} - x_{a})

A *lower bound* on :math:`Z^{*}` derived from the convex linearisation of :math:`Z` at :math:`x`
combined with the AON optimality of :math:`y`. The bound can be negative on the first iterations
of a freshly-started run because the objective contribution can be much smaller than the inner
product term until the assignment stabilises.

.. math:: \mathit{BestLowerBound} = \max\left( \mathit{LowerBound}_{k},\ \mathit{BestLowerBound}_{k-1} \right)

The running maximum of ``LowerBound`` across iterations. AequilibraE initialises this to
:math:`-\infty` at the start of every assignment, so resumed runs (including more iterations on
top of an already-converged assignment) discard the best lower bound from the previous run.
``BestLowerBound`` is monotonically non-decreasing
across iterations within a single assignment.

Best relative gap (``best_rgap``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math:: \mathit{BestRGap} = \frac{Z(x) - \mathit{BestLowerBound}}{\left| Z(x) \right|}

A measure of how close the current flow's Beckmann objective is to the (unknown) user-equilibrium
optimum :math:`Z^{*}`. A ``BestRGap`` of e.g. ``0.001`` means the current flow's Beckmann objective
is within 0.1 % of :math:`Z^{*}`. Because ``BestLowerBound`` is initialised to :math:`-\infty`,
``BestRGap`` is reported as ``inf`` on the very first iteration where it is computed, and on the
first iteration of any resumed run.

Available algorithms
--------------------

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
| Biconjugate Frank-Wolfe       | Biconjugate direction | Optimal value derived from       |
|                               | (Current and two      | Wardrop's principle              |
|                               | previous AoN)         |                                  |
+-------------------------------+-----------------------+----------------------------------+
| Conjugate Frank-Wolfe         | Conjugate direction   | Optimal value derived from       |
|                               | (Current and          | Wardrop's principle              |
|                               | previous AoN)         |                                  |
+-------------------------------+-----------------------+----------------------------------+

.. note::
   Our implementations of the conjugate and biconjugate Frank-Wolfe methods should be inherently 
   proportional [5]_, but we have not yet carried the appropriate testing that would be required 
   for an empirical proof.

Method of Successive Averages (MSA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This algorithm has been included largely for historical reasons, and we see very little reason to 
use it. Yet, it has been implemented with the appropriate computation of relative gap computation 
and supports all the analysis features available.

Frank-Wolfe (FW)
~~~~~~~~~~~~~~~~

The implementation of Frank-Wolfe in AequilibraE is extremely simple from an implementation point of 
view, as we use a generic optimizer from SciPy as an engine for the line search, and it is a standard 
implementation of the algorithm introduced by LeBlanc in 1975 [6]_.

Biconjugate Frank-Wolfe (BFW)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The biconjugate Frank-Wolfe algorithm is currently the fastest converging link-based traffic assignment 
algorithm used in practice, and it is the recommended algorithm for AequilibraE users. Due to its need for 
previous iteration data, it **requires more memory** during runtime, but very large networks should still
fit nicely in systems with 16Gb of RAM.

Conjugate Frank-Wolfe
~~~~~~~~~~~~~~~~~~~~~

The conjugate direction algorithm was introduced in 2013 [7]_, which is quite recent if you consider 
that the Frank-Wolfe algorithm was first applied in the early 1970's, and it was introduced at the same 
time as its Biconjugate evolution, so it was born outdated.

Implementation details & tricks
-------------------------------

A few implementation details and tricks are worth mentioning not because they are needed to use the software, 
but because they were things we grappled with during implementation, and it would be a shame not register 
it for those looking to implement their own variations of this algorithm or to slight change it for their own 
purposes.

* The relative gap is computed with the cost used to compute the All-or-Nothing portion of the iteration, 
  and although the literature on this is obvious, we took some time to realize that we should re-compute the 
  travel costs only **AFTER** checking for convergence.

* In some instances, Frank-Wolfe is extremely unstable during the first iterations on assignment, resulting on 
  numerical errors on our line search. We found that setting the step size to the corresponding MSA value 
  (1/current iteration) resulted in the problem quickly becoming stable and moving towards a state where the line search started working properly. This technique was generalized to the conjugate and biconjugate Frank-Wolfe algorithms.

* All five gap measures listed under :ref:`convergence_criteria` are written into the
  ``convergence_report`` attribute of the assignment object (a dict of lists, one entry per
  iteration) and emitted to the iteration log line. Only ``RelGap`` is compared against the
  user-supplied target gap; the others are reported for diagnostic and cross-package comparison
  purposes. The Beckmann integral :math:`Z(x)` is computed in closed form per VDF in dedicated
  Cython helpers (``integral_bpr``, ``integral_bpr2``, ``integral_conical``, ``integral_inrets``)
  exposed through ``vdf.apply_integral``.

Multi-threaded implementation
-----------------------------

AequilibraE's All-or-Nothing assignment (the basis of all the other algorithms) has been parallelized in Python 
using the threading library, which is possible due to the work we have done with memory management to release 
Python's Global Interpreter Lock.

Other opportunities for parallelization, such as the computation of costs and its derivatives (required during 
the line-search optimization step), as well as all linear combination operations for vectors and matrices have 
been achieved through the use of OpenMP in pure Cython code. These implementations can be found on a file called
``parallel_numpy.pyx`` if you are curious to look at.

Much of the gains of going back to Cython to parallelize these functions came from making in-place computation 
using previously existing arrays, as the instantiation of large NumPy arrays can be computationally expensive.

Handling the network
--------------------

The other important topic when dealing with multi-class assignment is to have a single consistent handling of 
networks, as in the end there is only physical network across all modes, regardless of access differences to 
each mode (e.g. truck lanes, high-occupancy lanes, etc.). This handling is often done with something called a
*super-network*.

A super-network consists in having all classes with the same links in their sub-graphs, but assigning `b_node` 
identical to `a_node` for all links whenever a link is not available for a certain user class.

This approach is slightly less efficient when we are computing shortest paths, but it gets eliminated when 
topologically compressing the network for centroid-to-centroid path computation and it is a LOT more efficient 
when we are aggregating flows.

The use of the AequilibraE project and its built-in methods to build graphs ensure that all graph will be built 
in a consistent manner and multi-class assignment is possible.

References
----------

.. [1] Wardrop, J.G. (1952) "Some theoretical aspects of road traffic research."
       Proceedings of the Institution of Civil Engineers 1952, 1(3):325-362. 
       Available in: https://www.icevirtuallibrary.com/doi/abs/10.1680/ipeds.1952.11259

.. [2] Marcotte, P., Patriksson, M. (2007) 
       "Chapter 10 Traffic Equilibrium - Handbooks in Operations Research and Management Science, Vol 14", 
       Elsevier. Editors Barnhart, C., Laporte, G. https://doi.org/10.1016/S0927-0507(06)14010-4

.. [3] Zill, J., Camargo, P., Veitch, T., Daisy, N. (2019) 
       "Toll Choice and Stochastic User Equilibrium: Ticking All the Boxes",
       Transportation Research Record, 2673(4):930-940. 
       Available in: https://doi.org/10.1177%2F0361198119837496

.. [4] Rose, G., Daskin, M., Koppelman, F. (1988) 
       "An examination of convergence error in equilibrium traffic assignment models", 
       Transportation Research Part B, 22(4):261-274. 
       Available in: https://doi.org/10.1016/0191-2615(88)90003-3

.. [5] Florian, M., Morosan, C.D. (2014) "On uniqueness and proportionality in multi-class equilibrium assignment",
       Transportation Research Part B, 70:261-274. 
       Available in: https://doi.org/10.1016/j.trb.2014.06.011

.. [6] LeBlanc, L.J., Morlok, E.K., Pierskalla, W.P. (1975) 
       "An efficient approach to solving the road network equilibrium traffic assignment problem". 
       Transportation Research, 9(5):309-318. 
       Available in: https://doi.org/10.1016/0041-1647(75)90030-1

.. [7] Mitradjieva, M., Lindberg, P.O. (2013) 
       "The Stiff Is Moving—Conjugate Direction Frank-Wolfe Methods with Applications to Traffic Assignment".
       Transportation Science, 47(2):280-293. 
       Available in: https://doi.org/10.1287/trsc.1120.0409