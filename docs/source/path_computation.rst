Path Computation
================

Given AequilibraE's incredibly fast path computation capabilities, one of its important use cases
is the computation of paths on general transportation networks and between any two nodes, regardless
of their type (centroid or not).

This use case supports the development of a number of computationally intensive systems, such as
map-matching GPS data and simulation of Demand Responsive Transport (DRT, e.g. Uber) operators,
for example.

#. **Path computation**: computes the path between two arbritrary nodes.
#. **Network skimming**: can compute either the distance, the travel time, or your own cost matrix
   between a series of nodes.

.. toctree::
    :hidden:
    :maxdepth: 1

    path_computation/_auto_examples/index

.. seealso::
    
    * :func:`aequilibrae.paths.results`
        Class documentation
    * :ref:`example_usage_path_computation` 
        Usage example
    * :ref:`example_usage_skimming`
        Usage example
