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

This class calibrates the parameters of a doubly constrained gravity model, using one of the available
deterrence functions: ``EXPO``, ``POWER``, or ``GAMMA``. It requires some arguments such as:

* Matrix containing the base trips (``AequilibraeMatrix``);
* Impedance matrix (``AequilibraeMatrix``);
* Deterrence function name.

The parameters are estimated by maximum likelihood, treating the observed trips as Poisson counts. For a
doubly constrained model, the maximum likelihood estimate is the one for which the modelled trips reproduce
the observed totals of the deterrence function's sufficient statistics [2]_:

+----------+-----------------------------------------+---------------------------------------------------+
| Function | Deterrence                              | The calibrated model reproduces the observed      |
+==========+=========================================+===================================================+
| EXPO     | :math:`f(c) = e^{-\beta c}`             | average trip cost                                 |
+----------+-----------------------------------------+---------------------------------------------------+
| POWER    | :math:`f(c) = c^{-\alpha}`              | average logarithm of the trip cost                |
+----------+-----------------------------------------+---------------------------------------------------+
| GAMMA    | :math:`f(c) = c^{\alpha} e^{-\beta c}`  | average trip cost and average logarithm of the    |
|          |                                         | trip cost                                         |
+----------+-----------------------------------------+---------------------------------------------------+

For the exponential function, this is the same target as Hyman's method [1]_. For every candidate set of
parameters, the model is balanced to the observed row and column totals, and the parameters are updated with
Newton's method on the resulting profile log-likelihood. That function is concave, so the procedure converges
from any starting point, typically in four to six iterations.

Observed trips in cells the deterrence function cannot reproduce are left out of the calibration: cells with a
missing cost, cells at or above the maximum trip length (when one is set) and, for the ``POWER`` and ``GAMMA``
functions, cells with zero cost, such as intrazonal trips without an intrazonal impedance. The calibration
report states how many trips were left out.

K-factors
~~~~~~~~~

K-factors multiply the trips between pairs of districts, capturing interactions the deterrence function
cannot, such as trips across a river or a state line. Passing a ``pd.Series`` with the district of each zone as
``districts`` calibrates a k-factor for every pair of districts together with the deterrence function, so the
calibrated model reproduces the observed trips between each pair of districts as well as the trip length
statistics above. Fitting both at once matters: k-factors fitted after the deterrence function change the trip
length distribution it was calibrated to reproduce.

Each balancing of the model adjusts the row, column and district pair totals together, with k-factors starting
at one and multiplied by the ratio between observed and modelled trips of their district pair, so they measure how
far each pair departs from the gravity model without k-factors. In a doubly constrained model, multiplying all
k-factors of one origin district, or of one destination district, by the same amount does not change any trip,
as the zone balancing factors absorb it. What k-factors express is therefore how the pairs of an origin district
compare to each other, and likewise for a destination district.

Pairs of districts without observed trips get a k-factor of zero, so the model never sends trips between them.
``k_factor_bounds`` sets a minimum and a maximum k-factor, for example ``(0.5, 2)``, which keeps every pair of
districts connected at the cost of not reproducing the observed trips of pairs at a bound.

The k-factors are saved with the model, and ``GravityApplication`` applies them to every zone with a district in
the model.

Please check other arguments and parameters that are passed to ``GravityCalibration`` in its
documentation.

.. seealso::

    * :func:`aequilibrae.distribution.gravity_calibration.GravityCalibration`
        Function documentation


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

References
----------

.. [1] Hyman, G.M. (1969) The calibration of trip distribution models. Environment and Planning, 1(1), 105-112.

.. [2] Wilson, A.G. (1970) Entropy in Urban and Regional Modelling. London: Pion.
