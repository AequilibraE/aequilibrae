Route choice models
===================

BRING TEXT FROM THE PAPER
XXXXXXXXXXXXXX

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

