Route choice models
===================

Path-Size logit is based on the multinomial logit (MNL) model, which is one of the most used models in the
transportation field in general [1]_. It can be derived from random utility-maximizing
principles with certain assumptions on the distribution of the random part of the utility. To
account for the correlation of alternatives, Ramming [2]_ introduced a correction factor that measures the
overlap of each route with all other routes in a choice set based on shared link attributes, which gives rise to the PSL
model.  The PSL is currently the most used route choice model in practice, hence its choice as the first algorithm
to be implemented in AequilibraE

Path-Size Logit (PSL)
~~~~~~~~~~~~~~~~~~~~~

The PSL model’s utility function is defined by

.. math:: U_{i} = V_{i} + \beta_{PSL} \times \log{\gamma_i} + \varepsilon_{i}

with path overlap correction factor

.. math:: \gamma_i = \sum_{a \in A_i} \frac{l_a}{L_i} \times \frac{1}{\sum_{k \in R} \delta_{a,k}}

Here, :math:`U_i` is the total utility of alternative :math:`i`, :math:`V_i` is the observed utility,
:math:`\varepsilon_i` is an identical and independently distributed random variable with a Gumbel distribution,
:math:`\delta_{a,k}` is the Kronecker delta, :math:`l_a` is cost of link :math:`a`, :math:`L_i` is total cost of
route :math:`i`, :math:`A_i` is the link set and :math:`R` is the route choice set for individual :math:`j` (index
:math:`j` suppressed for readability). The path overlap correction factor :math:`\gamma` can be theoretically derived by
aggregation of alternatives under certain assumptions, see [3]_ and references therein.

.. note::

    **AequilibraE uses cost to compute path overlaps rather than distance**

Binary logit filter
~~~~~~~~~~~~~~~~~~~

A binary logit filter is available to remove unfavourable routes from the route set before applying the path-sized logit
assignment. This filters accepts a numerical parameter for the minimum demand share acceptable for any path, which is
approximated by the binary logit considering the shortest path and each subsequent path.

References
----------

.. [1] Ben-Akiva, M., and S. Lerman. Discrete Choice Analysis. The MIT Press, 1985.

.. [2] Ramming, M. S. Network Knowledge and Route Choice. Massachusetts Institute of Technology, 2002.

.. [3] Frejinger, E. (2008) Route Choice Analysis : Data , Models , Algorithms and Applications.
