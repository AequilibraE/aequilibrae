.. _aequilibrae_as_path_engine:

Path computation engine
=======================

Given AequilibraE's incredibly fast path computation capabilities, one of its
important use cases is the computati`on of paths on general transportation
networks and between any two nodes, regardless of their type (centroid or not).

This use case supports the development of a number of computationally intensive
systems, such as map matching of GPS data, simulation of Demand Responsive
Transport (DRT, e.g. Uber) operators.

This capability is implemented within an specific class *PathResults*, which is
fully documented in the :ref:`aequilibrae_api` section of this documentation.

Below we detail its capability for a number of use-cases outside traditional
modeling, from a simple path computation to a more sophisticated map-matching
use.
`
Basic setup

::

    from aequilibrae import Project
    from aequilibrae.paths.results import PathResults

    proj_path = 'D:/release/countries/United Kingdom'

    proj = Project()
    proj.open(proj_path)

    # We assume we are going to compute walking paths (mode *w* in our example)
    # We also assume that we have fields for distance and travel time in the network
    proj.network.build_graphs(['distance', 'travel_time'], modes = 'w')

    # We get the graph
    graph = proj.network.graphs['w']

    # And prepare it for computation

    # Being primarily a modeling package, AequilibraE expects that your network
    # will have centroids (synthetic nodes) and connectors (synthetic links)
    # and we therefore need to account for it when computing paths
    # Here we will assume that we do not have centroids in the network, so
    # we will have to *trick* the Graph object

    # let's get 10 of our nodes (completely arbitrary, do as you please) to
    # serve as *centroids*
    # The AequilibraE project file is based on SQLite, so we can just do a query
    curr = proj.conn.cursor()
    curr.execute('Select node_id from Nodes WHERE modes like "%c%" limit 100')
    nodes = list(set([x[0] for x in curr.fetchall()]))

    # Just use the arbitrary node set as centroids
    graph.prepare_graph(np.array(nodes))

    # Tell AequilibraE that no link is synthetic (no need to block paths going through *"centroids"*).
    graph.set_blocked_centroid_flows(False)

    # We will minimize travel_time
    graph.set_graph('travel_time')

    # And *skim* (compute the corresponding) distance for the resulting paths
    # you should do this ONLY if you require skims for any field other than the minimizing field
    # or for all the nodes in the graph
    # It can increase computation time in up to 30%
    graph.set_skimming(['distance', 'travel_time'])

    # Finally, we get the path result computation object and prepare it to work with our graph
    res = PathResults()
    res.prepare(g)

    # We are now ready to compute paths between any two nodes in the network


path computation and finding your way around
--------------------------------------------

Building on the code above, we can just compute paths between two arbitrary
nodes.

::

    res.compute_path(32568, 179)

    # You can consult the origin & destination for the path you computed
    res.origin
    res.destination

    # You can also consult the sequence of links traversed from origin to destination
    res.path

    # And the sequence of nodes visited in that path
    res.path_nodes

    # You can also know the direction you traversed each link with
    res.path_link_directions # Array of the same size as res.path

    # If you chose to compute skims, you can access them for ALL NODES
    # Array is indexed on node IDs
    res.skims
    # Order of columns is the same as in
    graph.skim_fields
    # disconnected and non-existing nodes are set to np.inf

    # The metric used to compute the path is also summarized for all nodes along the path
    res.milepost
    # This is especially useful when you want to interpolate other metrics along the path
    # This is the case in route-reconstruction when map-matching GPS data

    # The shortest path tree is stored during computation, so recomputing the path from
    # the same origin but for a different destination is lightning fast
    res.update_trace(195)

    # Skims obviously won't change, but the OD pair specific data will
    res.path_nodes
    res.path
    res.path_link_directions
    res.milepost


Network skimming
----------------
If your objective is just to compute distance/travel_time/your_own_cost matrix
between a series of nodes, then the process is even simpler


::

    from aequilibrae.paths.results import SkimResults

    res.compute_path(32568, 179)

    # You can consult the origin & destination for the path you computed
    res.origin
    res.destination

    # You would prepare the graph with "centroids" that correspond to the nodes
    # you are interested in
    graph.prepare_graph(np.array(my_nodes_of_interest))

    # And do the steps from the setup phase accordingly
    graph.set_blocked_centroid_flows(False)
    graph.set_graph('travel_time')
    graph.set_skimming(['distance', 'travel_time'])

    # Finally, we get the path result computation object and prepare it to work with our graph
    skm_res = SkimResults()
    skm_res.prepare(graph)

    # You can tell AequilibraE to use an specific number of cores
    skm_res.set_cores(12)

    # And then compute it
    skm_res.compute_skims()

    skm_res.skims.export('path/to/matrix.omx')
    # or
    skm_res.skims.export('path/to/matrix.aem')
    # or
    skm_res.skims.export('path/to/matrix.csv')



