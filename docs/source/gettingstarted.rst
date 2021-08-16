.. _getting_started:

Getting Started
===============

This page describes how to get started with AequilibraE.

.. note::
   Although AequilibraE is under intense development, we try to avoid making
   breaking changes to the API. In any case, you should check for new features
   and possible API changes often.

.. note::
   The recommendations on this page are current as of August 2021.

.. index:: installation

Installation
------------

1. Install `Python 3.6, 3.7, 3.8 or 3.9 <www.python.org>`__. We recommend Python
   3.8.

2. Install AequilibraE

::

  pip install aequilibrae

.. _dependencies:

Dependencies
~~~~~~~~~~~~

Aequilibrae relies on a series of compiled libraries, such as NumPy and Scipy.
If you are working on Windows and have trouble installing any of the
requirements, you can look at
`Christoph Gohlke's wonderful repository <https://www.lfd.uci.edu/~gohlke/pythonlibs/>`_
of compiled Python packages for windows. Particularly on Python 3.9, it may be
necessary to resort to Christoph's binaries.

OMX support
+++++++++++
AequilibraE also supports OMX starting on version 0.5.3, but that comes with a
few extra dependencies. Installing **openmatrix** solves all those dependencies:

::

  pip install openmatrix

.. _installing_spatialite_on_windows:

Spatialite
++++++++++

Although the presence of Spatialite is rather obiquitous in the GIS ecosystem,
it has to be installed separately from Python or AequilibraE.

This `blog post <https://xl-optim.com/spatialite-and-python-in-2020/>`_ has a more
comprehensive explanation of what is the setup you need to get Spatialite working,
but below there is something you can start with.

Windows
^^^^^^^
Spatialite does not have great support on Python for Windows. For this reason,
it is necessary to download Spatialite for Windows and inform AequilibraE of its
location.

One can download the appropriate version of the latest SpatiaLite release
directly from its `project page <https://www.gaia-gis.it/gaia-sins/>`_ , or the
cached versions on AequilibraE's website for
`64-Bit Python <https://www.aequilibrae.com/binaries/spatialite/mod_spatialite-5.0.1-win-amd64.zip>`_
or
`32-Bit Python <https://www.aequilibrae.com/binaries/spatialite/mod_spatialite-5.0.1-win-x86.zip>`_

After unpacking the zip file into its own folder (say D:/spatialite), one can
start their Python session by creating a *temporary* environment variable with said
location, as follows:

::

  import os
  from aequilibrae.utils.create_example import create_example

  os.environ['PATH'] = 'D:/spatialite' + ';' + os.environ['PATH']

  project = create_example(fldr, 'nauru')

For a permanent recording of the Spatialite location on your system, please refer
to the blog post referenced above or Windows-specific documentation.

Ubuntu Linux
^^^^^^^^^^^^

On Ubuntu it is possible to install Spatialite by simply using apt-get

::

  sudo apt-get install libsqlite3-mod-spatialite
  sudo apt-get install -y libspatialite-dev


MacOS
^^^^^

On MacOS one can use brew as per
`this answer on StackOverflow <https://stackoverflow.com/a/48370444/1480643>`_.

::

  brew install libspatialite

Hardware requirements
---------------------

AequilibraE's requirements depend heavily of the size of the model you are using
for computation. The most important
things to keep an eye on are:

* Number of zones on your model (size of the matrices you are dealing with)

* Number of matrices (vehicles classes (and user classes) you are dealing with)

* Number of links and nodes on your network (far less likely to create trouble)

Substantial testing has been done with large real-world models (up to 8,000
zones) and memory requirements did not exceed the traditional 32Gb found in most
modelling computers these days. In most cases 16Gb of RAM is enough even for
large models (5,000+ zones).  Parallelization is fully implemented for path
computation, and can make use of as many CPUs as there are available in the
system when doing traffic assignment.
