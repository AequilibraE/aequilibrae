.. _aequilibrae-graphs:

Networks and Graphs
===================

A ``Network`` holds source link tables, fields, modes, centroids and turn restrictions.
A ``Graph`` holds one routing context and mappings back to source links.
Graphs do not retain DataFrames or project connections.

Network data
------------

``project.network`` manages the project database.
Its ``build()`` method copies source data into an independent ``Network``.
The optional ``fields``, ``modes`` and ``limit_to_area`` arguments limit that data.

.. code-block:: python

    >>> network = project.network.build(fields=["distance", "free_flow_time"], modes=["c"])

For workflows without a project, ``Network(links)`` accepts a link DataFrame.
The required columns are ``link_id``, ``a_node``, ``b_node`` and ``direction``.
A direction of 1 permits AB travel, -1 permits BA travel, and 0 permits both directions.
An optional ``modes`` column identifies the modes that can use each link.
Without that column, every mode can use every link.

Directional attributes use paired ``_ab`` and ``_ba`` columns.
For example, ``time_ab`` and ``time_ba`` provide values for the graph field ``time``.
The source table remains available as ``network.links``.
Changes to that table affect new graphs, not existing graphs.

Building a graph
----------------

``build_graph()`` selects the mode, cost field, centroids and skim fields before it prepares routing arrays.

.. code-block:: python

    >>> graph = network.build_graph(
    ...     mode="c", cost="distance", skimming=["distance", "free_flow_time"],
    ...     block_centroid_flows=False,
    ... )
    >>> result = graph.compute_path(1, 17)
    >>> result.path_nodes
    >>> result.path

``path_nodes`` contains external node IDs. ``path`` contains source link IDs.
``path_link_directions`` identifies each link's travel direction.
``milepost`` contains cumulative costs, including turn penalties.

For a single graph, ``project.network.build_graph(...)`` combines the source load and graph preparation.
For several graphs or assignments, an explicit ``Network`` can reuse the source data.
Both forms work independently of the project after the source load.

Full and compact graphs
-----------------------

A graph contains either full or compact routing topology, not both.
Full graphs are the default and support paths between arbitrary network nodes.
``compact=True`` prepares topology for centroid-based computations.
Traffic assignment builds compact graphs internally.

Compression contracts chains between intersections and removes dead ends that contain no centroids.
Centroids and nodes that control turn restrictions remain in the graph.
The expansion mapping converts compact paths and loads back to source links.

.. code-block:: python

    >>> graph = network.build_graph(
    ...     "c", "distance", centroids=np.array([1, 17]), compact=True,
    ...     skimming=["distance"], block_centroid_flows=False,
    ... )
    >>> skims = graph.compute_skims().results.skims

Topology changes
----------------

Modes, centroids, skim fields and turn restrictions are fixed for each graph.
A topology change requires a new graph from the network.

.. code-block:: python

    >>> graph = network.build_graph(
    ...     "c", "distance", centroids=np.array([1, 17]),
    ...     turn_restrictions=turns, allow_uturns=False,
    ... )

For area selection, ``project.network.build(limit_to_area=polygon)`` uses the database spatial index.
The selection is fast, but it uses bounding boxes rather than exact geometric intersections.

Cost changes and link closures
------------------------------

``Graph.set_costs()`` changes costs without changing topology.
The array uses directed network links in ascending link-ID and direction order.
``network.link_ids`` and ``network.directions`` describe that order.
Positive infinity closes a link direction to routing.

.. code-block:: python

    >>> costs = network.field_values("distance").copy()
    >>> costs[np.isin(network.link_ids, [123, 975])] = np.inf
    >>> graph.set_costs(costs)

Existing path results retain their own cost snapshot.
New searches use the updated costs.

Blocking centroid flows
-----------------------

``block_centroid_flows=True`` prevents paths from passing through other centroids.
With explicit turns, the builder inserts the corresponding connector-turn prohibitions.
For ordinary network nodes used as centroids, ``block_centroid_flows=False`` permits through travel.

.. image:: ../_images/aequilibrae_graph-1.png
    :align: center
    :alt: example of blocked travel through a centroid

.. seealso::

    * :class:`aequilibrae.paths.network.Network`
    * :class:`aequilibrae.paths.graph.Graph`
    * :class:`aequilibrae.paths.graph.TransitGraph`
