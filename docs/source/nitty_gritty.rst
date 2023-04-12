.. _nitty_gritty:

Nitty-Gritty
===============

This page describes how to get started with AequilibraE.

.. note::
   Although AequilibraE is under intense development, we try to avoid making
   breaking changes to the API. In any case, you should check for new features
   and possible API changes often.

.. note::
   The recommendations on this page are current as of April 2023.

.. index:: installation

Installation
------------

1. Install `Python 3.7, 3.8, 3.9, 3.10 or 3.11 <www.python.org>`__. We recommend Python
   3.9 or 3.10.

2. Install AequilibraE

::

  pip install aequilibrae

.. _dependencies:

Dependencies
~~~~~~~~~~~~

All of AequilibraE's dependencies are readily available from `PyPI
<https://www.pypi.org/>`_ for all currently supported Python versions and major
platforms.

.. _installing_spatialite_on_windows:

Spatialite
++++++++++

Although the presence of Spatialite is rather ubiquitous in the GIS ecosystem,
it has to be installed separately from Python or AequilibraE in any platform.

This `blog post <https://xl-optim.com/spatialite-and-python-in-2020/>`_ has a more
comprehensive explanation of what is the setup you need to get Spatialite working,
but that is superfluous if all you want is to get it working.

Windows
^^^^^^^

.. note::
   On Windows ONLY, AequilibraE automatically verifies if you have SpatiaLite
   installed in your system and downloads it to your temporary folder if you do
   not.

Spatialite does not have great support on Python for Windows. For this reason,
it is necessary to download Spatialite for Windows and inform and load it
to the Python SQLite driver every time you connect to the database.

One can download the appropriate version of the latest SpatiaLite release
directly from its `project page <https://www.gaia-gis.it/gaia-sins/>`_ , or the
cached versions on AequilibraE's website for
`64-Bit Python <https://github.com/AequilibraE/aequilibrae/releases/tag/V.0.7.5>`_

After unpacking the zip file into its own folder (say *D:/spatialite*), one can
**temporarily** add the spatialite folder to system path environment variable,
as follows:

::

  import os
  os.environ['PATH'] = 'D:/spatialite' + ';' + os.environ['PATH']

For a permanent recording of the Spatialite location on your system, please refer
to the blog post referenced above or Windows-specific documentation.

Ubuntu Linux
^^^^^^^^^^^^

On Ubuntu it is possible to install Spatialite by simply using apt-get

::

  sudo apt update -y
  sudo apt install -y libsqlite3-mod-spatialite
  sudo apt install -y libspatialite-dev


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
modeling computers these days. In most cases 16Gb of RAM is enough even for
large models (5,000+ zones). Computationally intensive procedures such as
skimming and traffic assignment have been parallelized, so AequilibraE can make
use of as many CPUs as there are available in the system for such procedures.

.. toctree::
    :maxdepth: 1
    :hidden:

    :ref:`installation`
    nitty_gritty/softwaredevelopment
    nitty_gritty/roadmap
    nitty_gritty/version_history
    nitty_gritty/useful_links
  