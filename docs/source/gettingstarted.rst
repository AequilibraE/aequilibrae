
Getting Started
===============

This page describes how to get started with AequilibraE.

.. note::
   Although AequilibraE is under intense development, you don't need to expect important API changes, but you should \
   check for new features often.
   

.. index:: installation

Installation
------------

1. Install `Python 3.5, 3.6 or 3.7 <www.python.org>`__. We recommend Python 3.7 As of late 2019.

2. Install AequilibraE
  
::
    
  pip install aequilibrae


Dependencies
~~~~~~~~~~~~

Aequilibrae relies on a series of compiled libraries, such as NumPy and Scipy. If you are working on Windows and have
trouble installing any of the requirements, you can look at `Christoph Gohlke's wonderful repository <https://www.lfd.uci.edu/~gohlke/pythonlibs/>`_ of compiled Python packages for windows.

OMX support
+++++++++++
AequilibraE also supports OMX starting on version 0.5.3, but that comes with a few extra dependencies. Installing
**openmatrix** solves all those dependencies:

::

  pip install aequilibrae

Hardware requirements
---------------------

AequilibraE's requirements depend heavily of the size of the model you are using for computation. The most important
things to keep an eye on are:

* Number of zones on your model (size of the matrices you are dealing with)

* Number of matrices (classes you are dealing with)

* Number of links and nodes on your network (far less likely to create trouble)


Substantial testing has been done with large real-world models (up to 6,000 zones) and memory requirements did not
exceed the traditional 32Gb found in most modelling computers these days. In most cases 16Gb of RAM is enough even for
large models.  Parallelization is fully implemented for graph computation, and can make use of as many CPUs as there
are available in the system when doing traffic assignment.
