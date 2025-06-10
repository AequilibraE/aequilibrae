Distribution procedure classes
==============================

AequilibraE's distribution module comprises three different classes: ``GravityApplication``, 
``GravityCalibration``, and ``Ipf``.

``GravityApplication``
----------------------

This class, as its own name explains, applies a synthetic gravity model, using one of the available
deterrence funcions: ``EXPO``, ``POWER``, or ``GAMMA``. It requires some parameters, such as:

* Synthetic gravity model (which is an instance of ``SyntheticGravityModel``)
* Impedance matrix (``AequilibraeMatrix``);
* Vector (``Pandas.DataFrame``) with data for row and column totals;
* Row and column fields, which are the names of the fields that contain the data for row and column
  totals.

The synthetic gravity model instance can be either created or loaded, if you have already calibared
a model.

Plase check other arguments and parameters that are passed to ``GravityApplication`` in its 
documentation.

.. seealso::
    
    * :func:`aequilibrae.distribution.SyntheticGravityModel`
        Function documentation
    * :func:`aequilibrae.distribution.GravityApplication`
        Function documentation


``GravityCalibration``
-----------------------

Calibrate the model consists in checking if all the parameters set are appropriate. This class,
as its own name explains, calibrates a traditional gravity model, using one of the available
deterrence funcions: ``EXPO``, ``POWER``, or ``GAMMA``. It requires some arguments such as:

* Matrix containing the base trips (``AequilibraeMatrix``);
* Impedance matrix (``AequilibraeMatrix``);
* Deterrence function name.

Plase check other arguments and parameters that are passed to ``GravityCalibration`` in its 
documentation.

.. seealso::
    
    * :func:`aequilibrae.distribution.GravityCalibration`
        Function documentation

.. 
    I'm a bit confused here. We currently don't have gamma deterrence function, right?
    Can I keep it, as something to be implemented or can I delete it? Also check the class
    documentation.


``Ipf``
-------

IPF is an acronym for Iterative Proportial Fitting, also known as Fratar or Furness. The IPF 
procedure is used to "distribute" future trips based on a growth factor. The procedure can be 
run with or without an AequilibraE model, with the latter using one of AequilibraE matrices 
or NumPy arrays as data input.

In the following section, we present the validation of the results produced with AequilibraE's
IPF.

.. seealso::

    * :func:`aequilibrae.distribution.Ipf`
        Function documentation
    * :ref:`plot_ipf_without_model`
        Usage example
    * :ref:`plot_ipf_core`
        Usage example
