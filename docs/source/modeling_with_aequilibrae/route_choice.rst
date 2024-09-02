.. _route_choice:

Route Choice
============

As argued in the literature [1]_, the route choice problem does not have a closed solution, and the selection
of one of the many modelling frameworks [2]_ depends on many factors. A common modelling framework in practice
is consists of two steps: Choice set generation and the choice selection process.

AequilibraE is the first modeling package with full support for route choice, from the creation of choice sets through
multiple algorithms to the assignment of trips to the network using the traditional Path-Size logit.

Costs, utilities and signs
--------------------------

AequilibraE's path computation procedures require all link costs to be positive. For that reason,
link utilities (or disutilities) must be positive, while its obvious minus sign is handled internally.
This mechanism prevents the possibility of links with actual positive utility, but those cases are arguably
not reasonable to exist in practice.

Choice set generation algorithms available
------------------------------------------

All algorithms have been implemented as a single software class

+----------------------------------+----------------------------------+
| Algorithm                        | Brief description                |
+==================================+==================================+
| Link-Penalisation                | Classical link penalisation.     |
|                                  |                                  |
+----------------------------------+----------------------------------+
| Breadth-First Search with        | As described in [2]_.            |
| Link Removal                     |                                  |
+----------------------------------+----------------------------------+
| Breadth-First Search with        | A combination of BFS-LE and LP   |
| Link Removal + Link-Penalisation | See `RouteChoice` documentation  |
+----------------------------------+----------------------------------+

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
   theta = 0.0014
   graph.network = graph.network.assign(utility=graph.network.distance * theta)
   graph.set_graph('utility')

   graph.prepare_graph(centroids=[list_of_all_nodes_network_centroids])

   rc = RouteChoice(graph, mat)
   rc.set_choice_set_generation("bfsle", max_routes=5)
   rc.execute(perform_assignment=True)

Full process overview
---------------------

The estimation of route choice models based on vehicle GPS data can be explored on a family of papers scheduled to
be presented at the ATRF 2024 [1]_ [3]_ [4]_.

.. [1] Zill, J. C., and P. V. de Camargo. State-Wide Route Choice Models (Submitted).
       Presented at the ATRF, Melbourne, Australia, 2024.

.. [2] Rieser-Schüssler, N., Balmer, M., & Axhausen, K. W. (2012). Route choice sets for very high-resolution data.
       Transportmetrica A: Transport Science, 9(9), 825–845. DOI: https://doi.org/10.1080/18128602.2012.671383

.. [3] Camargo, P. V. de, and R. Imai. Map-Matching Large Streams of Vehicle GPS Data into Bespoke Networks (Submitted).
       Presented at the ATRF, Melbourne, 2024.

.. [4] Moss, J., P. V. de Camargo, C. de Freitas, and R. Imai. High-Performance Route Choice Set Generation on
       Large Networks (Submitted). Presented at the ATRF, Melbourne, 2024.

.. seealso::
       
       LOREM IPSUM

.. toctree::
    :maxdepth: 1
    :caption: Route Choice

    route_choice/route_choice_model.rst
    route_choice/choice_set_generation.rst
