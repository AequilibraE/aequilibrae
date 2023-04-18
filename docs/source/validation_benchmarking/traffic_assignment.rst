.. _numerical_study_traffic_assignment:

Traffic Assignment
==================

Similar to other complex algorthms that handle a large amount of data through
complex computations, traffic assignment procedures can always be subject to at
least one very reasonable question: Are the results right?

For this reason, we have used all equilibrium traffic assignment algorithms
available in AequilibraE to solve standard instances used in academia for
comparing algorithm results, some of which have are available with highly
converged solutions (~1e-14). Instances can be downloaded `here <https://github.com/bstabler/TransportationNetworks/>`_.

Sioux Falls
-----------
Network has:

* Links: 76
* Nodes: 24
* Zones: 24

.. image:: ../images/sioux_falls_msa-500_iter.png
    :align: center
    :width: 590
    :alt: Sioux Falls MSA 500 iterations
|
.. image:: ../images/sioux_falls_frank-wolfe-500_iter.png
    :align: center
    :width: 590
    :alt: Sioux Falls Frank-Wolfe 500 iterations
|
.. image:: ../images/sioux_falls_cfw-500_iter.png
    :align: center
    :width: 590
    :alt: Sioux Falls Conjugate Frank-Wolfe 500 iterations
|
.. image:: ../images/sioux_falls_bfw-500_iter.png
    :align: center
    :width: 590
    :alt: Sioux Falls Biconjugate Frank-Wolfe 500 iterations

Anaheim
-------
Network has:

* Links: 914
* Nodes: 416
* Zones: 38

.. image:: ../images/anaheim_msa-500_iter.png
    :align: center
    :width: 590
    :alt: Anaheim MSA 500 iterations
|
.. image:: ../images/anaheim_frank-wolfe-500_iter.png
    :align: center
    :width: 590
    :alt: Anaheim Frank-Wolfe 500 iterations
|
.. image:: ../images/anaheim_cfw-500_iter.png
    :align: center
    :width: 590
    :alt: Anaheim Conjugate Frank-Wolfe 500 iterations
|
.. image:: ../images/anaheim_bfw-500_iter.png
    :align: center
    :width: 590
    :alt: Anaheim Biconjugate Frank-Wolfe 500 iterations

Winnipeg
--------
Network has:

* Links: 914
* Nodes: 416
* Zones: 38

.. image:: ../images/winnipeg_msa-500_iter.png
    :align: center
    :width: 590
    :alt: Winnipeg MSA 500 iterations
|
.. image:: ../images/winnipeg_frank-wolfe-500_iter.png
    :align: center
    :width: 590
    :alt: Winnipeg Frank-Wolfe 500 iterations
|
.. image:: ../images/winnipeg_cfw-500_iter.png
    :align: center
    :width: 590
    :alt: Winnipeg Conjugate Frank-Wolfe 500 iterations
|
.. image:: ../images/winnipeg_bfw-500_iter.png
    :align: center
    :width: 590
    :alt: Winnipeg Biconjugate Frank-Wolfe 500 iterations

The results for Winnipeg do not seem extremely good when compared to a highly,
but we believe posting its results would suggest deeper investigation by one
of our users :-)


Barcelona
---------
Network has:

* Links: 2,522
* Nodes: 1,020
* Zones: 110

.. image:: ../images/barcelona_msa-500_iter.png
    :align: center
    :width: 590
    :alt: Barcelona MSA 500 iterations
|
.. image:: ../images/barcelona_frank-wolfe-500_iter.png
    :align: center
    :width: 590
    :alt: Barcelona Frank-Wolfe 500 iterations
|
.. image:: ../images/barcelona_cfw-500_iter.png
    :align: center
    :width: 590
    :alt: Barcelona Conjugate Frank-Wolfe 500 iterations
|
.. image:: ../images/barcelona_bfw-500_iter.png
    :align: center
    :width: 590
    :alt: Barcelona Biconjugate Frank-Wolfe 500 iterations

Chicago Regional
----------------
Network has:

* Links: 39,018
* Nodes: 12,982
* Zones: 1,790

.. image:: ../images/chicago_regional_msa-500_iter.png
    :align: center
    :width: 590
    :alt: Chicago MSA 500 iterations
|
.. image:: ../images/chicago_regional_frank-wolfe-500_iter.png
    :align: center
    :width: 590
    :alt: Chicago Frank-Wolfe 500 iterations
|
.. image:: ../images/chicago_regional_cfw-500_iter.png
    :align: center
    :width: 590
    :alt: Chicago Conjugate Frank-Wolfe 500 iterations
|
.. image:: ../images/chicago_regional_bfw-500_iter.png
    :align: center
    :width: 590
    :alt: Chicago Biconjugate Frank-Wolfe 500 iterations

Convergence Study
-----------------

Besides validating the final results from the algorithms, we have also compared
how well they converge for the largest instance we have tested (Chicago
Regional), as that instance has a comparable size to real-world models.

.. image:: ../images/convergence_comparison.png
    :align: center
    :width: 590
    :alt: Algorithm convergence comparison
|

Not surprinsingly, one can see that Frank-Wolfe far outperforms the Method of
Successive Averages for a number of iterations larger than 25, and is capable of
reaching 1.0e-04 just after 800 iterations, while MSA is still at 3.5e-4 even
after 1,000 iterations.

The actual show, however, is left for the Biconjugate Frank-Wolfe
implementation, which delivers a relative gap of under 1.0e-04 in under 200
iterations, and a relative gap of under 1.0e-05 in just over 700 iterations.

This convergence capability, allied to its computational performance described
below suggest that AequilibraE is ready to be used in large real-world
applications.

Computational performance
-------------------------
Running on a Thinkpad X1 extreme equipped with a 6 cores 8750H CPU and 32Gb of
2667Hz RAM, AequilibraE performed 1,000 iterations of Frank-Wolfe assignment
on the Chicago Network in just under 46 minutes, while Biconjugate Frank Wolfe
takes just under 47 minutes.

During this process, the sustained CPU clock fluctuated between 3.05 and 3.2GHz
due to the laptop's thermal constraints, suggesting that performance in modern
desktops would be better

Noteworthy items
----------------

.. note::
   The biggest opportunity for performance in AequilibraE right now it to apply
   network contraction hierarchies to the building of the graph, but that is
   still a long-term goal
