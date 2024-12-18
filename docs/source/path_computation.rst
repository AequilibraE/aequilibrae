Path Computation
================

Given AequilibraE's incredibly fast path computation capabilities, one of its important use cases
is the computation of paths on general transportation networks and between any two nodes, regardless
of their type (centroid or not).

This use case supports the development of a number of computationally intensive systems, such as
map-matching GPS data and simulation of Demand Responsive Transport (DRT, e.g. Uber) operators,
for example.

Some basic usages of the AequilibraE path module consist on:

#. **Path computation**: computes the path between two arbritrary nodes.

#. **Network skimming**: can compute either the distance, the travel time, or your own cost matrix
   between a series of nodes.

Regarding computing paths through a network, part of its complexity comes from the fact that 
transportation models usually house networks for multiple transport modes, so the loads (links)
available for a passenger car may be different than those available for a heavy truck, as it happens 
in practice.

For this reason, all path computation in AequilibraE happens through ``Graph`` objects. While users 
can operate models by simply selecting the mode they want AequilibraE to create graphs for, ``Graph`` 
objects can also be manipulated in memory or even created from networks that are 
:ref:`NOT housed inside an AequilibraE model <plot_assignment_without_model>`.

AequilibraE's graphs are the backbone of path computation, skimming and traffic assignment. 
Besides handling the selection of links available to each mode in an AequilibraE model, graphs 
also handle the existence of bi-directional links with direction-specific characteristics 
(e.g. speed limit, congestion levels, tolls, etc.). For this reason, the next section is
entirely dedicated to this object.

.. seealso::
    
    * :func:`aequilibrae.paths.PathResults`
        Class documentation
    * :ref:`example_usage_path_computation` 
        Usage example
    * :ref:`example_usage_skimming`
        Usage example

.. toctree::
    :caption: Path Computation
    :maxdepth: 1

    path_computation/aequilibrae_graph
    path_computation/_auto_examples/index
