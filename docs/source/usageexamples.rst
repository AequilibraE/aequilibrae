
Use examples
============
This page is still under development, so most of the headers are just place-holders for the actual examples

.. note::
   The examples provided here are not meant as a through description of AequilibraE's capabilities. For that, please
   look into the API documentation or email aequilibrae@googlegroups.com

Paths module
------------


::

  from aequilibrae.paths import Graph
  from aequilibrae.paths import allOrNothing
  from aequilibrae.paths import NetworkSkimming
  from aequilibrae.paths import path_computation
  from aequilibrae.paths.results import AssignmentResults as asgr
  from aequilibrae.paths.results import PathResults as pthr
  from aequilibrae.paths.results import SkimResults as skmr


Building a graph
~~~~~~~~~~~~~~~~

Path computation
~~~~~~~~~~~~~~~~

Skimming
~~~~~~~~

Let's suppose you want to compute travel times between all zone on your network

::

    some code

And if you want to also skim distance while computing fastest path

::

    some code

And if you want to compute skims between all nodes in the network

::

    some code

Traffic Assignment
~~~~~~~~~~~~~~~~~~

::

    some code

Gravity Models
--------------

::

    some code

Synthetic gravity calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    some code

Synthetic gravity application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    some code

Iterative Proportional Fitting (IPF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    some code

Transit
-------
We only have import for now, and it is likely to not work on Windows if you want the geometries

GTFS import
~~~~~~~~~~~

::

    some code

Matrix computation
------------------

::

    some code