.. _aequilibrae_as_path_engine:

Path computation engine
=======================

Given AequilibraE's incredibly fast path computation capabilities, one of its
important use cases is the computation of paths on general transportation
networks and between any two nodes, regardless of their type (centroid or not).

This use case supports the development of a number of computationally intensive
systems, such as map matching of GPS data, simulation of Demand Responsive
Transport (DRT, e.g. Uber) operators.

This capability is implemented within an specific class *PathResults*, which is
fully documented in the :ref:`aequilibrae_api` section of this documentation.

Below we detail its capability for a number of use-cases outside traditional
modeling, from a simple path computation to a more sophisticated map-matching
use.
