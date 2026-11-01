Distribution procedure classes
==============================

AequilibraE's distribution module comprises three different classes: ``GravityApplication``, 
``GravityCalibration``, and ``Ipf``.

``GravityApplication``
----------------------

This class, as its own name explains, applies a synthetic gravity model, using one of the available
deterrence functions: ``EXPO``, ``POWER``, or ``GAMMA``. It requires some parameters, such as:

* Synthetic gravity model (which is an instance of ``SyntheticGravityModel``)
* Impedance matrix (``AequilibraeMatrix``);
* Vector (``Pandas.DataFrame``) with data for row and column totals;
* Row and column fields, which are the names of the fields that contain the data for row and column
  totals.

The synthetic gravity model instance can be either created or loaded, if you have already calibrated
a model.

Please check other arguments and parameters that are passed to ``GravityApplication`` in its 
documentation.

.. seealso::
    
    * :func:`aequilibrae.distribution.synthetic_gravity_model.SyntheticGravityModel`
        Function documentation
    * :func:`aequilibrae.distribution.gravity_application.GravityApplication`
        Function documentation


``GravityCalibration``
-----------------------

Calibrate the model consists in checking if all the parameters set are appropriate. This class,
as its own name explains, calibrates a traditional gravity model, using one of the available
deterrence functions: ``EXPO``, ``POWER``, or ``GAMMA``. It requires some arguments such as:

* Matrix containing the base trips (``AequilibraeMatrix``);
* Impedance matrix (``AequilibraeMatrix``);
* Deterrence function name.

Please check other arguments and parameters that are passed to ``GravityCalibration`` in its 
documentation.

.. seealso::
    
    * :func:`aequilibrae.distribution.gravity_calibration.GravityCalibration`
        Function documentation

.. 
    I'm a bit confused here. We currently don't have gamma deterrence function, right?
    Can I keep it, as something to be implemented or can I delete it? Also check the class
    documentation.


``Ipf``
-------

IPF is an acronym for Iterative Proportional Fitting, also known as Fratar or Furness. The IPF
procedure is used to "distribute" future trips based on a growth factor. The procedure can be
run with or without an AequilibraE model, with the latter using one of AequilibraE matrices
or NumPy arrays as data input.

Under the hood, ``Ipf`` calls ``ipf_core``, which balances a NumPy array in place. It never rescales
the matrix during the iterations: it only updates one factor per row and one per column, and applies
them in a single pass at the end. If you only need the factors, ``ipf_factors`` returns them without
modifying the seed matrix.

In the following section, we compare the performance of AequilibraE's IPF with reference implementations.

.. seealso::

    * :func:`aequilibrae.distribution.ipf.Ipf`
        Function documentation
    * :ref:`plot_ipf_without_model`
        Usage example
    * :ref:`plot_ipf_core`
        Usage example
