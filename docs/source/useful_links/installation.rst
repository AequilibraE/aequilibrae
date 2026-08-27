:orphan:

.. _installation:

Installation
============

In this section we describe how to install AequilibraE.

.. important::
   Although AequilibraE is under intense development, we try to avoid making
   breaking changes to the API. In any case, you should check for new features
   and possible API changes often.

.. index:: installation

Installation
------------

1. Install `Python 3.11, 3.12, 3.13 & 3.14 <https://www.python.org/downloads/>`_. We recommend Python 3.12 or 3.13
2. Install AequilibraE

::

  pip install aequilibrae

That is all: AequilibraE's spatial database engine is implemented in pure Python on
top of shapely, pyproj and SQLite's built-in R*Tree module, so no native SpatiaLite
library (``mod_spatialite``), system package or runtime download is required on any
platform. The project databases AequilibraE creates remain standard SpatiaLite
files, fully compatible with QGIS and other SpatiaLite-aware tools.

macOS
^^^^^

AequilibraE does not provide pre-built wheel files for macOS. When installing from PyPi, the source distribution will be used and the library will be compiled locally. AequilibraE can also be built from source. For both methods you will need to:

#. Install `homebrew <https://brew.sh/>`_, a package manager for macOS, if you do not have it already.
#. Install LLVM or another C/C++ compiler with OpenMP support: ``brew install llvm``
#. Set the C and C++ compilers: ``export CXX=/opt/homebrew/opt/llvm/bin/clang++`` and ``export CC=/opt/homebrew/opt/llvm/bin/clang``

Alternatively, run all steps at once:

::

  brew install llvm
  export CXX=/opt/homebrew/opt/llvm/bin/clang++
  export CC=/opt/homebrew/opt/llvm/bin/clang

AequilibraE may also require raising the "open files" limit, this can be achieved with ``ulimit -n 10240``. This should be placed in ``.zshrc`` or similar user shell configuration file.

.. _dependencies:

Dependencies
------------

All of AequilibraE's dependencies are readily available from `PyPI <https://www.pypi.org/>`_
for all currently supported Python versions and major platforms.

.. _installing_spatialite:

SpatiaLite
++++++++++

Older versions of AequilibraE (up to 1.7) required the native SpatiaLite extension
(``mod_spatialite``) to be present: a Debian/Homebrew package on Linux/macOS, or a
binary download on Windows. This is **no longer necessary**: the subset of
SpatiaLite that AequilibraE uses is provided by a bundled, pure-Python
implementation, and projects created either way are interchangeable.

The ``AEQ_SPATIALITE_DIR`` environment variable and the automatic Windows download
have been removed. If you maintain scripts that referenced them, they can simply be
deleted.

Hardware requirements
---------------------

How much hardware AequilibraE needs is driven mostly by the size of the model
being used. The most important things to keep an eye on are:

* Number of zones on your model (size of the matrices you are dealing with)
* Number of matrices (vehicles classes (and user classes) you are dealing with)
* Number of links and nodes on your network (far less likely to create trouble)

Substantial testing has been done with large real-world models (up to 8,000
zones) and memory requirements did not exceed the traditional 32Gb found in most
modeling computers these days. In most cases 16Gb of RAM is enough even for
large models (5,000+ zones). Computationally intensive procedures such as
skimming and traffic assignment have been parallelized, so AequilibraE can make
use of as many CPUs as there are available in the system for such procedures.
