Distribution Procedures
=======================

In the context of transportation modeling, a distribution model tries to estimate the number of 
trips in each of the matrix cells on the basis of any information available [1]_. 

AequilibraE's distribution module comprises three different classes: ``GravityApplication``, 
``GravityCallibration``, and ``Ipf``.

``GravityApplication``
----------------------



``GravityCallibration``
-----------------------

Calibrate the model consists in checking if all the parameters set are appropriate. This class,
as its own name explains, calibrates a traditional gravity model, using one of the available
deterrence funcions: ``EXPO``, ``POWER``, or ``GAMMA``.


``Ipf``
-------

IPF is an acronym for Iterative Proportial Fitting, also known as Fratar or Furness. The IPF 
procedure is used to "distribute" future trips based on a growth factor. The procedure can be 
run with or without an AequilibraE model, with the latter using one of AequilibraE matrices 
or NumPy arrays as data input.

In the following section, we present the validation of the results produced with AequilibraE's
IPF.

.. toctree::
    :hidden:
    :maxdepth: 1

    IPF_benchmark

.. seealso::

    * :func:`aequilibrae.distribution.Ipf`
        Function documentation
    * :ref:`plot_ipf_without_model`
        Usage example
    * :ref:`plot_ipf_core`
        Usage example


References
----------

.. [1] Ortúzar, J. de D. and Willumsen, L.G. (2011) Modelling transport. 4th edition. Chichester: Wiley.
