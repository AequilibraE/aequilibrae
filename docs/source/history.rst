History
=======

Before there was AequilibraE, there was a need for something like AequilibraE out there.

The very early days
-------------------

It all started when `Pedro <https://www.xl-optim.com/>`_ was a student at `UCI-ITS <https://www.its.uci.edu/>`_  and
needed low level access to outputs of standard algorithms used in transportation modeling (e.g. path files from traffic
assignment) and had that denied by the maker of the commercial software he normally used. There, the
`first scratch of a traffic assignment procedure <www.xl-optim.com/python-traffic-assignment>`_ was born.
After that, there were a couple of scripts developed to implement synthetic gravity models (calibration and application)
that were develop for a government think-tank in Brazil `IPEA <https://www.ipea.gov.br/>`_.
Around the same time, another student needed a piece of code that transformed a GIS link layer into a proper graph,
where each link would become the connection between two nodes.
So there were three fundamental pieces that would come to be part of AequilibraE.

The first take on a release software
------------------------------------

Having all those algorithms at hand, it made sense combining them into something more people could use, and by them it
seemed that QGIS was the way to go, so I developed the
`very first version of AequilibraE <http://www.xl-optim.com/introducing_aequilibrae>`_.

It was buggy as hell and there was very little, if any, software engineering built into it, but it put Aequilibrae on
the map. That was 16/December/2014.

The first reasonable version
----------------------------

The first important thing Pedro noticed after releasing AequilibraE was that the code was written in procedural style,
even though it would make a lot more sense doing it in a Object-Oriented fashion, which let me down the path of
creating the objects (graph, assignment results, matrix) that the software still relies on and were the foundation
blocks of the proper API that is in the making. That
`version was released in 2016 <http://www.xl-optim.com/new-version-of-aequilibrae>`_

Evolving into proper software
-----------------------------

A few distinct improvements deserve to be highlighted.

* The separation of the GUI and the Python library in `two repositories <http://www.xl-optim.com/separating-the-women-from-the-girls>`_
* Introduction of Unit Tests and automatic testing using `Travis (replaced with GitHub Actions) <https://travis-ci.org/AequilibraE/aequilibrae>`_
* Development of proper documentation, more software engineering with style-checking (Flake8 and Black)
* Re-development of the the most basic algorithms (path-finding, IPF, etc) with highly parallelized code optimized for the transportation use case

Release of AequilibraE 1.0
--------------------------

On its 9th anniversary (16/12/2023), AequilibraE saw the release of its landmark
`version 1.0 <https://www.outerloop.io/blog20231216_aequilibrae1.0>`_
which is the first to include a suite of Public Transport tools, making AequilibraE a fully-featured software.
