.. _aequilibrae-graphs:

AequilibraE Graphs
==================

The Graph object is rather complex, but the difference between the graph and the physical 
links are the availability of two class member variables consisting of Pandas DataFrames: the
**network** and the **graph**.

.. code-block:: python

    >>> from aequilibrae.paths import Graph

    >>> g = Graph()

    >>> g.network # doctest: +SKIP
    >>> g.graph # doctest: +SKIP

Directionality
--------------

Links in the Network table (the Pandas representation of the project's *Links* table) are
potentially bi-directional, and the directions allowed for traversal are dictated by the
field *direction*, where -1 and 1 denote only BA and AB traversal respectively and 0 denotes
bi-directionality.

Direction-specific fields must be coded in fields **_AB** and **_BA**, where the name of
the field in the graph will be equal to the prefix of the directional fields. For example:

The fields **free_flow_travel_time_AB** and **free_flow_travel_time_BA** provide the same
metric (*free_flow_travel_time*) for each of the directions of a link, and the field of
the graph used to set computations (e.g. field to minimize during path-finding, skimming,
etc.) will be **free_flow_travel_time**.

Graphs from a model
-------------------

Building graphs directly from an AequilibraE model is the easiest option for beginners
or when using AequilibraE in anger, as much of the setup is done by default.

.. code-block:: python

    >>> project = create_example(project_path, "coquimbo")

    >>> project.network.build_graphs() # We build the graph for all modes
    >>> graph = project.network.graphs['c'] # we grab the graph for cars

Manipulating graphs in memory
-----------------------------

As mentioned before, the AequilibraE Graph can be manipulated in memory, with all its
components available for editing. One of the simple tools available directly in the
API is a method call for excluding one or more links from the Graph, **which is done**
**in place**.

.. code-block:: python

    >>> graph.exclude_links([123, 975])

When working with very large networks, it is possible to filter the database to a small
area for computation by providing a polygon that delimits the desired area, instead of
selecting the links for deletion. The selection of links and nodes is limited to a spatial
inidex search, which is very fast but not accurate.

.. code-block:: python

    >>> polygon = Polygon([(-71.35, -29.95), (-71.35, -29.90), (-71.30, -29.90), (-71.30, -29.95), (-71.35, -29.95)])
    >>> project.network.build_graphs(limit_to_area=polygon) # doctest: +SKIP

More sophisticated graph editing is also possible, but it is recommended that
changes to be made in the network DataFrame. For example:

.. code-block:: python

    # We can add fields to our graph
    >>> graph.network["link_type"] = project.network.links.data["link_type"]

    # And manipulate them
    >>> graph.network.loc[graph.network.link_type == "motorway", "speed_ab"] = 100
    >>> graph.network.loc[graph.network.link_type == "motorway", "speed_ba"] = 100

Skimming settings
-----------------

Skimming the field of a graph when computing shortest path or performing
traffic assignment must be done by setting the skimming fields in the
Graph object, and there are no limits (other than memory) to the number
of fields that can be skimmed.

.. code-block:: python

    >>> graph.set_skimming(["distance", "travel_time"])

Setting centroids
-----------------

Like other elements of the AequilibraE Graph, the user can also manipulate the
set of nodes interpreted by the software as centroids in the Graph itself.
This brings the advantage of allowing the user to perform assignment of partial
matrices, matrices of travel between arbitrary network nodes and to skim the network
for an arbitrary number of centroids in parallel, which can be useful when using
AequilibraE as part of more general analysis pipelines. As seen above, this is also
necessary when the network has been manipulated in memory.

**When setting regular network nodes as centroids, the user should take care in
not blocking flows through "centroids".**

.. code-block:: python

    >>> graph.prepare_graph(np.array([13, 169, 2197, 28561, 37123], np.int32))
    >>> graph.set_blocked_centroid_flows(False)

.. seealso::

    * :func:`aequilibrae.paths.Graph`
        Class documentation
    * :func:`aequilibrae.paths.TransitGraph`
        Class documentation


Topological simplification
---------------------------

When using the AequilibraE graph to perform topological simplification of the graph,
the user should be explicit in not removing dead ends from the graph, as that will
result in simplification beyond pure topological simplification.

.. code-block:: python

    >>> graph.prepare_graph(np.array([13, 169, 2197, 28561, 37123], np.int32), remove_dead_ends=False)
