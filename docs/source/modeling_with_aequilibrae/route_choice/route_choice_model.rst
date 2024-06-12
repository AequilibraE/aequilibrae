Route choice models
===================

As argued in the literature [3]_, the route choice problem does not have a closed solution, and the selection
of one of the many modelling frameworks [4]_ depends on many factors. A common modelling framework in practice
is explicit path generation, where the route choice problem can be broken down into two sub-problems: Choice set
generation and the choice selection process. Most choice set generation algorithms are versions of the k-shortest path
algorithm.

Path-Size Logit (PSL)
~~~~~~~~~~~~~~~~~~~~~

Path-Size logit is based on the multinomial logit (MNL) model, which is one of the most used models in the
transportation field in general (Ben-Akiva and Lerman, 1985). It can be derived from random utility-maximizing
principles with certain assumptions on the distribution of the random part of the utility. To
account for the correlation of alternatives, Ramming (Ramming, 2002) introduced a correction factor that measures the
overlap of each route with all other routes in a choice set based on shared link attributes, which gives rise to the PSL
model. The PSL model’s utility function is defined by

.. math:: U_i = V_i + \beta_{PSL} \cross \log{\gamma_i} + \varepsilon_i

with path overlap correction factor

.. math:: \gamma_i = \sum_{a \in A_i} \frac{l_a}{L_i} \cross \frac{1}{\sum_{k \in R} \delta_{a,k}}

Here, :math:`U_i` is the total utility of alternative :math:`i`, :math:`V_i` is the observed utility,
:math:`\varepsilon_i` is an identical and independently distributed random variable with a Gumbel distribution,
:math:`\delta_{a,k}` is the Kronecker delta, :math:`l_a` is length of link :math:`a`, :math:`L_i` is total length of
route :math:`i`, :math:`A_i` is the link set and :math:`R` is the route choice set for individual :math:`j` (index
:math:`j` suppressed for readability). The path overlap correction factor :math:`\gamma` can be theoretically derived by
aggregation of alternatives under certain assumptions, see [5]_ and references therein.

Binary logit filter
~~~~~~~~~~~~~~~~~~~

A binary logit filter is available to remove unfavourable routes from the route set before applying the path-sized logit
assignment. This filters takes a percentage and computes the difference in utility based on the minimum cost path. Paths
with a cost that exceeds the minimum cost + the difference in utility are marked with `mask=False` and excluded from
path-size logit.

Algorithms available
~~~~~~~~~~~~~~~~~~~~

All algorithms have been implemented as a single software class

+----------------------------------+----------------------------------+
| Algorithm                        | Brief description                |
+==================================+==================================+
| Link-Penalisation                | Classical link penalisation.     |
|                                  |                                  |
+----------------------------------+----------------------------------+
| Breadth-First Search with        | As described in [6]_.            |
| Link Removal                     |                                  |
+----------------------------------+----------------------------------+
| Breadth-First Search with        | A combination of BFS-LE and LP   |
| Link Removal + Link-Penalisation | See `RouteChoice` documentation  |
+----------------------------------+----------------------------------+

Link-Penalisation
^^^^^^^^^^^^^^^^^

Traditional link penalisation. Every step, the new shortest path is stored and it's link costs are penalised with a
configurable factor.

Breadth-First Search with Link Removal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As described in [6]_, BFS-LE explores of graph of graphs formed by removing links from the graph based on the shortest
path found. See `RouteChoice` documentation for a more thorough description. 

Breadth-First Search with Link Removal + Link-Penalisation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A combination of BFS-LE and LP. In addition to removing each link from the shortest path in a separate graph, all links
are penalised for the next depth to discourage slight detours formed by removing links.

Imports
-------

.. code:: python

   from aequilibrae.paths.route_choice_set import RouteChoiceSet
   from aequilibrae import Project

   proj = Project()
   proj.load('path/to/project/folder')

   proj.network.build_graphs()
   graph = proj.network.graphs['c']

   # Measure that will be used to compute paths
   graph.network = graph.network.assign(utility=graph.network.distance * theta)
   graph.set_graph('utility')

   graph.prepare_graph(centroids=[list_of_all_nodes_network_centroids])

   rc = RouteChoice(graph, mat)
   rc.set_choice_set_generation("bfsle", max_routes=5, beta=1.1)
   rc.execute(perform_assignment=True)







References
----------

.. [1] Ramming, M. S. Network Knowledge and Route Choice. Massachusetts Institute of Technology, 2002.

.. [2] Zill, J. C., and P. V. de Camargo. State-Wide Route Choice Models. (Submitted)
       Presented at the ATRF, Melbourne, Australia, 2024.

.. [3] Camargo, P.V. (2014) ReMulAA - a New Algorithm for the Route Choice Problem. Available at:
       https://books.google.com.au/books?id=q3vsoAEACAAJ.

.. [4] Prato, C.G. (2009) ‘Route choice modeling: past, present and future research directions’, Journal of Choice
       Modelling, 2(1), pp. 65–100. Available at: https://doi.org/10.1016/S1755- 5345(13)70005-8.

.. [5] Frejinger, E. (2008) Route Choice Analysis : Data , Models , Algorithms and Applications.

.. [6] Rieser-Schüssler, N., Balmer, M., Axhausen, K.W., 2013. Route choice sets for very high-resolution
       data. Transportmetrica A: Transport Science 9, 825–845. https://doi.org/10.1080/18128602.2012.671383
