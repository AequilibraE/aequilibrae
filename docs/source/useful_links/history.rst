:orphan:

.. _history_of_aequilibrae:

History
=======

Before there was AequilibraE, there was a need for something like AequilibraE out there.

The very early days
-------------------

It all started when `Pedro <https://www.xl-optim.com/>`_ was a student at `UCI-ITS <https://www.its.uci.edu/>`_  and
needed low level access to outputs of standard algorithms used in transportation modeling (e.g. path files from traffic
assignment) and had that denied by the maker of the commercial software he normally used. There, the
`earliest sketch of a traffic assignment routine <https://www.xl-optim.com/python-traffic-assignment>`_ came to life.

After that, there were a couple of scripts developed to implement synthetic gravity models (calibration and application)
written for `IPEA <https://www.ipea.gov.br/>`_, a government think-tank in Brazil.

Around the same time, another student needed a piece of code that transformed a GIS link layer into a proper graph,
where each link would become the connection between two nodes.

So there were three fundamental pieces that would come to be part of AequilibraE.

The first take on a release software
------------------------------------

With all those algorithms at hand, bundling them into something a wider audience could use was the obvious
next step, and QGIS looked like the right vehicle at the time, which led Pedro to build the
`very first version of AequilibraE <http://www.xl-optim.com/introducing_aequilibrae>`_.

It was buggy as hell and there was very little, if any, software engineering built into it, but it put AequilibraE on
the map. That was 16/December/2014.

The first reasonable version
----------------------------

The first important thing Pedro noticed after releasing AequilibraE was that the code was written in procedural style,
even though an Object-Oriented design would have suited it far better, which set him on the path of
creating the objects (graph, assignment results, matrix) that the software still relies on and were the foundation
blocks of the proper API that is in the making. That
`version was released in 2016 <http://www.xl-optim.com/new-version-of-aequilibrae>`_.

Evolving into proper software
-----------------------------

A few distinct improvements deserve to be highlighted.

* The separation of the GUI and the Python library in `two repositories <http://www.xl-optim.com/separating-the-women-from-the-girls>`_
* Introduction of Unit Tests and automatic testing using `Travis (replaced with GitHub Actions) <https://travis-ci.org/AequilibraE/aequilibrae>`_
* Development of proper documentation, more software engineering with style-checking (Flake8 and Black)
* Rewrite of the core algorithms (path-finding, IPF, etc) as highly parallelized code tuned for the transportation use case

Release of AequilibraE 1.0
--------------------------

On its 9th anniversary (16/12/2023), AequilibraE reached
`release 1.0 <https://www.outerloop.io/blog/20231216_aequilibrae1.0/>`_,
which is the first to include a suite of Public Transport tools, making AequilibraE a fully-featured software.
