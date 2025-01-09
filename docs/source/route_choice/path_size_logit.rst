Path-size logit (PSL)
=====================

Path-size logit is based on the multinomial logit (MNL) model, which is one of the most used models 
in the transportation field in general [1]_. It can be derived from random utility-maximizing 
principles with certain assumptions on the distribution of the random part of the utility. To
account for the correlation of alternatives, Ramming (2002) [2]_ introduced a correction factor 
that measures the overlap of each route with all other routes in a choice set based on shared 
link attributes, which gives rise to the PSL model. The PSL is currently the most used route 
choice model in practice, hence its choice as the first algorithm to be implemented in AequilibraE.

The PSL model's utility function is defined by:

.. math:: U_{i} = V_{i} + \beta_{PSL} \times \log{\gamma_i} + \varepsilon_{i}

with path overlap correction factor:

.. math:: \gamma_i = \sum_{a \in A_i} \frac{l_a}{L_i} \times \frac{1}{\sum_{k \in R} \delta_{a,k}}

Here, :math:`U_i` is the total utility of alternative :math:`i`, :math:`V_i` is the observed utility,
:math:`\varepsilon_i` is an identical and independently distributed random variable with a Gumbel 
distribution, :math:`\delta_{a,k}` is the Kronecker delta, :math:`l_a` is cost of link 
:math:`a`, :math:`L_i` is total cost of route :math:`i`, :math:`A_i` is the link set and :math:`R` 
is the route choice set for individual :math:`j` (index :math:`j` suppressed for readability). The 
path overlap correction factor :math:`\gamma` can be theoretically derived by aggregation of 
alternatives under certain assumptions, see [3]_ and references therein.

Notice that AequilibraE's path computation procedures require all link costs to be positive. For 
that reason, link utilities (or disutilities) must be positive, while its obvious minus sign is 
handled internally. This mechanism prevents the possibility of links with actual positive utility, 
but those cases are arguably not reasonable to exist in practice.

.. important::

    **AequilibraE uses cost to compute path overlaps rather than distance.**

Binary logit filter
-------------------

A binary logit filter is available to remove unfavourable routes from the route set before applying
the path-sized logit assignment. This filters accepts a numerical parameter for the minimum demand 
share acceptable for any path, which is approximated by the binary logit considering the shortest 
path and each subsequent path.

Full process overview
---------------------

The estimation of route choice models based on vehicle GPS data can be explored on a family of papers 
scheduled to be presented at the ATRF 2024 [4]_ [5]_ [6]_.

References
----------

.. [1] Ben-Akiva, M., and Lerman, S. (1985) Discrete Choice Analysis. The MIT Press.

.. [2] Ramming, M.S. (2002) Network Knowledge and Route Choice. Massachusetts Institute of Technology.
       Available at: https://dspace.mit.edu/bitstream/handle/1721.1/49797/50436022-MIT.pdf?sequence=2&isAllowed=y

.. [3] Frejinger, E. (2008) Route Choice Analysis: Data, Models, Algorithms and Applications.
       Available at: https://infoscience.epfl.ch/server/api/core/bitstreams/6d43511f-e9c4-4fb4-b5c9-83a4515154b8/content

.. [4] Zill, J.C. and Camargo, P.V. (2024) State-Wide Route Choice Models.
       Presented at the ATRF, Melbourne, Australia.

.. [5] Camargo, P.V. and Imai, R. (2024) Map-Matching Large Streams of Vehicle GPS Data into 
       Bespoke Networks. Presented at the ATRF, Melbourne.

.. [6] Moss, J., Camargo, P.V., de Freitas, C. and Imai, R. (2024) High-Performance Route Choice 
       Set Generation on Large Networks. Presented at the ATRF, Melbourne.
