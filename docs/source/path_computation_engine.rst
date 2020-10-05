.. _aequilibrae_as_path_engine:

AequilibraE as a path computation engine
========================================

Given AequilibraE's incredibly fast path computation capabilities, of of its
important use cases is the computation of paths in general transportation
networks and between any two nodes, regardless of their type (centroid or not).

This use case supports the development of a number of computationally intensive
systems, such as map matching of GPS data, simulation of Demand Responsive
Transport (DRT, e.g. Uber) operators.

This capability is implemented within an specific class *PathResults*, which is
fully documented in the :ref:`aequilibrae_api` section of this documentation.

Below we detail its capability for a number of use-cases outside traditional
modeling, from a simple path computation to a more sophisticated map-matching
use.

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
    graph.prepare_graph(np.array(nodes))
    graph.set_blocked_centroid_flows(False)
    graph.set_graph('distance')
    graph.set_skimming(['distance'])

    # We get the path result computation object and create models for me
    res = PathResults()
    res.prepare(g)

    curr = p.conn.cursor()
    curr.execute('Select a_node, link_id, distance from Links order by distance limit 10')
    nodes = list(set([x[0] for x in curr.fetchall()]))


    logger.info('Preparing Graph')

    logger.info("Let's compute")

    t = perf_counter()
    res.compute_path(nodes[0], nodes[-1])
    t = perf_counter() - t
    p.logger.info(f'Computing path took {round(t, 4)}')

    res = SkimResults()
    res.prepare(g)
    t = perf_counter()
    res.compute_skims()
    t = perf_counter() - t
    p.logger.info(f'Computing Skim matrix {round(t, 4)}')
    p.close()




