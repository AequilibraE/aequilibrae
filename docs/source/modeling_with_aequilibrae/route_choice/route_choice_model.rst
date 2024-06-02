Route choice models
===================

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

   graph.prepare_graph(centroids=[list_of_all_nodes_which_are_origin_or_destination_in_the_observed_dataset])

   # Measure that will be used to compute paths
   graph.set_graph('free_flow_time')

   nodes = [(1, 50), (2, 100), (3, 150)]  # List of tuples with (origin, destination) nodes
   max_routes = 10  # Maximum number of routes to be computed for each OD pair
   penalty = 1.01  # Penalty to be applied to links used in paths.
   cores = 60  # Number of cores to be used in the computation
   psl = True  # If True, the path size logit will be used to compute probabilities already
   bfsle=True # Should we use BFSLE? If False, defaults to Link Penalization
   # This is only useful if you are already using an utility measure to compute paths

   rc = RouteChoiceSet(graph)  # Builds data structures -> can take a minute
   rc.batched(nodes, max_routes=max_routes, cores=cores, bfsle=bfsle, penalty=penalty, path_size_logit=psl)

   results = rc.get_results().to_pandas()
   results.to_parquet(Path(r"/my_choice_set.parquet")



References
----------

.. [1] Ramming, M. S. Network Knowledge and Route Choice. Massachusetts Institute of Technology, 2002.

.. [2] Zill, J. C., and P. V. de Camargo. State-Wide Route Choice Models.
       Presented at the ATRF, Melbourne, Australia, 2024.

