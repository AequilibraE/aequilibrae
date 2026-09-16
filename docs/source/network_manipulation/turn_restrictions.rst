.. _turn_restrictions:

Turn Restrictions
=================

AequilibraE supports turn restrictions and turn penalties for traffic assignment
and path computation. This feature enables modeling of prohibited turns (e.g.,
no left turn signs) and turn penalties (e.g., additional time for turning movements).

Overview
--------

Turn restrictions in AequilibraE are stored in a dedicated database table and can
be applied to graphs during path computation and traffic assignment. When turn
restrictions are enabled, AequilibraE uses an arc-based Dijkstra algorithm instead
of the standard node-based algorithm, which allows for modeling turn-to-turn
transitions with associated costs.

Database Schema
---------------

Turn restrictions are stored in the ``turn_restrictions`` table with the following columns:

- **restriction_id**: Unique identifier for the restriction (generated if omitted)
- **from_node**: Incoming movement origin node
- **via_node**: The turn node
- **to_node**: Outgoing movement destination node
- **penalty**: Turn penalty in time units (same as graph cost field). NULL indicates a prohibited turn.
  ``+inf`` is also accepted by the Python API and is normalised to NULL before storage.
  Negative values are not allowed.
- **modes**: String with the list of modes to which the restriction applies (e.g., "ct", "c", "w").

Turn definitions are node-sequences: ``from_node -> via_node -> to_node``.

Trigger Behavior
----------------

Database triggers enforce and maintain turn restriction integrity:

- node-sequence consistency is enforced
- mode codes are validated against the ``modes`` table
- duplicate restrictions with overlapping mode coverage are blocked
- restrictions are removed or remapped when linked ``links.link_id`` records
    are deleted or changed

When geometry support is enabled, ``turn_restrictions.geometry`` is generated
for the 3-node movement and updated when movement definition or node
geometry changes.

See :ref:`turn_restrictions_trigger_behaviour` for network-editing details.

Global U-Turn Setting
---------------------

The ``allow_uturns`` setting in the ``about`` table controls whether U-turns are
permitted globally in the network. By default, this is set to ``0`` (False),
meaning U-turns are prohibited.

U-turn detection is node-based: a transition is considered a U-turn when the next
directed arc returns to the tail node of the current directed arc.

When U-turns are allowed and graph compression is enabled, nodes where U-turns
could occur (nodes with bidirectional links) are preserved during compression
to maintain the correct turn possibilities.

Usage
-----

Managing Turn Restrictions
~~~~~~~~~~~~~~~~~~~~~~~~~~

Turn restrictions use the same table API as links and nodes. ``get()`` returns
an immutable record, ``data`` returns a GeoDataFrame, and ``len(turns)`` counts
records. Missing keys raise ``ValueError`` unless ``get()`` is given a default.

Supply ``modes`` on insert. Database triggers reject missing, empty, unknown or
repeated mode IDs, and overlapping restrictions for the same turn. These errors
raise ``sqlite3.IntegrityError``. Modes are stored in the supplied order.

Updates leave omitted fields unchanged; ``penalty=None`` prohibits the turn.
Use ``project.transaction()`` to group writes.

.. code-block:: python

    >>> project = create_example(project_path)

    # Get the turn restrictions object
    >>> turns = project.network.turn_restrictions

    # Add a prohibited turn (penalty=None means prohibited)
    >>> restriction_id = turns.insert(from_node=1, via_node=2, to_node=3, modes="c")

    # Add a turn with a 30-second penalty
    >>> other_id = turns.insert(from_node=3, via_node=4, to_node=5, penalty=30.0, modes="c")

    # Change only the penalty
    >>> turns.update(restriction_id, penalty=10.0)

    # Get all restrictions, including their generated geometry
    >>> df = turns.data

    # Remove a specific restriction
    >>> turns.delete(restriction_id)

    # Clear all restrictions
    >>> turns.clear_restrictions()

    >>> project.close()

Bulk Loading from DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``insert_from()`` to load turns from a DataFrame. It returns the inserted
IDs. Use ``update_from()`` with a ``restriction_id`` column to change existing
turns. Include a ``modes`` column when inserting. Both methods normalize
penalties just like single-record writes and leave mode validation to the
database. They roll back the whole batch if a row fails.

.. code-block:: python

    >>> import pandas as pd

    >>> restrictions_df = pd.DataFrame({
    ...     'from_node': [1, 2, 3],
    ...     'via_node': [2, 3, 4],
    ...     'to_node': [3, 4, 5],
    ...     'modes': ['c', 'c', 'c'],
    ...     'penalty': [None, 15.0, 30.0],  # None or +inf = prohibited
    ... })

    >>> restriction_ids = turns.insert_from(restrictions_df)

Setting Global U-Turn Permission
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The U-turn setting is managed through the project's ``about`` table:

.. code-block:: python

    >>> project.about.allow_uturns = '1'  # Allow U-turns
    >>> project.about.write_back()

    # Rebuild graphs to apply the change
    >>> project.network.build_graphs()

Building Graphs with Turn Restrictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you call ``build_graphs()``, turn restrictions are automatically loaded
from the database and applied to all generated graphs:

.. code-block:: python

    >>> project.network.build_graphs()

    # Check if a graph has turn restrictions
    >>> graph = project.network.graphs['c']
    >>> print(graph.has_turn_restrictions)
    True

Manual Turn Restriction Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also manually set turn restrictions on a graph:

.. code-block:: python

    >>> import pandas as pd
    >>> from aequilibrae.paths import Graph

    >>> graph = Graph()
    >>> graph.network = network_data
    >>> graph.prepare_graph(centroids)

    # Create turn restrictions DataFrame
    >>> turn_df = pd.DataFrame({
    ...     'from_node': [1, 2],
    ...     'via_node': [2, 3],
    ...     'to_node': [3, 4],
    ...     'penalty': [None, 20.0]
    ... })

    # Apply turn restrictions
    >>> graph.set_turn_restrictions(turn_df, allow_path_uturns=False)

    # Clear turn restrictions
    >>> graph.clear_turn_restrictions()

Traffic Assignment with Turn Restrictions
-----------------------------------------

Turn restrictions are automatically considered during traffic assignment when
they are present in the graph. The turn penalties are applied as additional
time costs during the equilibration process.

.. code-block:: python

    >>> from aequilibrae.paths import TrafficAssignment, TrafficClass

    >>> project.network.build_graphs()
    >>> graph = project.network.graphs['c']
    >>> graph.set_graph('free_flow_time')

    # Turn restrictions are automatically included
    >>> traffic_class = TrafficClass('car', graph, demand_matrix)

    >>> assignment = TrafficAssignment()
    >>> assignment.set_classes([traffic_class])
    >>> assignment.set_vdf('BPR')
    >>> assignment.set_vdf_parameters({'alpha': 0.15, 'beta': 4.0})
    >>> assignment.set_capacity_field('capacity')
    >>> assignment.set_time_field('free_flow_time')
    >>> assignment.set_algorithm('bfw')
    >>> assignment.execute()

Performance Considerations
--------------------------

The arc-based Dijkstra algorithm required for turn restrictions has higher memory
requirements than the standard node-based algorithm (arrays are indexed by number
of links rather than number of nodes). For this reason:

1. Arc-based path finding is only activated when turn restrictions are present
2. If no turn restrictions exist, the standard (faster) algorithm is used
3. Graph compression is adjusted to preserve nodes where U-turns could occur

Turn Penalty Units
------------------

Turn penalties must be specified in the same time unit as the graph's cost field
(typically the ``time`` field). For example, if ``time`` is
in seconds, turn penalties should also be in seconds.

Limitations
-----------

- A* path finding is not supported with turn restrictions. Requesting both raises a runtime error.
- Turn restrictions are defined by 3-node movement sequences (no lane-level control)
- Complex turn restrictions involving multiple via-nodes are not supported

