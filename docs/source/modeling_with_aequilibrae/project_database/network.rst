.. _network:

Network
~~~~~~~

The objectives of developing a network format for AequilibraE are to provide the
users a seamless integration between network data and transportation modeling
algorithms and to allow users to easily edit such networks in any GIS platform
they'd like, while ensuring consistency between network components, namely links
and nodes. As the network is composed by two tables, **links** and **nodes**,
maintaining this consistency is not a trivial task.

As mentioned in other sections of this documentation, the links and a nodes
layers are kept consistent with each other through the use of database triggers,
and the network can therefore be edited in any GIS platform or
programmatically in any fashion, as these triggers will ensure that
the two layers are kept compatible with each other by either making
other changes to the layers or preventing the changes.

**We cannot stress enough how impactful this set of spatial triggers was to**
**the transportation modelling practice, as this is the first time a**
transportation network can be edited without specialized software that 
**requires the editing to be done inside such software.**

.. note::
   AequilibraE does not currently support turn penalties and/or bans. Their
   implementation requires a complete overahaul of the path-building code, so
   that is still a long-term goal, barred specific development efforts.

Importing and exporting the network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently AequilibraE can import links and nodes from a network from OpenStreetMaps, 
GMNS, and from link layers. AequilibraE can also export the existing network
into GMNS format. There is some valuable information on these topics in the following
pages:

* :ref:`Importing files in GMNS format <importing_from_gmns>` 
* :ref:`Importing from OpenStreetMaps <importing_from_osm>`
* :ref:`Importing from link layers <project_from_link_layer>`
* :ref:`Exporting AequilibraE model to GMNS format <exporting_to_gmns>`

Dealing with Geometries
^^^^^^^^^^^^^^^^^^^^^^^
Geometry is a key feature when dealing with transportation infrastructure and
actual travel. For this reason, all datasets in AequilibraE that correspond to
elements with physical GIS representation, links and nodes in particular, are
geo-enabled.

This also means that the AequilibraE API needs to provide an interface to
manipulate each element's geometry in a convenient way. This is done using the
standard `Shapely <https://shapely.readthedocs.io/>`_, and we urge you to study
its comprehensive API before attempting to edit a feature's geometry in memory.

As we mentioned in other sections of the documentation, the user is also welcome
to use its powerful tools to manipulate your model's geometries, although that
is not recommended, as the "training wheels are off".

Data consistency
^^^^^^^^^^^^^^^^

Data consistency is not achieved as a monolithic piece, but rather through the
*treatment* of specific changes to each aspect of all the objects being
considered (i.e. nodes and links) and the expected consequence to other
tables/elements. To this effect, AequilibraE has triggers covering a
comprehensive set of possible operations for links and nodes, covering both
spatial and tabular aspects of the data.

Although the behaviour of these trigger is expected to be mostly intuitive
to anybody used to editing transportation networks within commercial modeling
platforms, we have detailed the behaviour for all different network changes in
:ref:`net_section.1` .

This implementation choice is not, however, free of caveats. Due to
technological limitations of SQLite, some of the desired behaviors identified in
:ref:`net_section.1` cannot be implemented, but such caveats do not impact the
usefulness of this implementation or its robustness in face of minimally careful
use of the tool.


.. note::
  This documentation, as well as the SQL code it referes to, comes from the
  seminal work done in `TranspoNet <http://github.com/AequilibraE/TranspoNet/>`_
  by `Pedro <https://au.linkedin.com/in/pedrocamargo>`_ and
  `Andrew <https://au.linkedin.com/in/andrew-o-brien-5a8bb486>`_.

.. _network_triggers_behaviour:

Network consistency behaviour
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order for the implementation of this standard to be successful, it is
necessary to map all the possible user-driven changes to the underlying data and
the behavior the SQLite database needs to demonstrate in order to maintain
consistency of the data. The detailed expected behavior is detailed below.
As each item in the network is edited, a series of checks and changes to other
components are necessary in order to keep the network as a whole consistent. In
this section we list all the possible physical (geometrical) changes to each
element of the network and what behavior (consequences) we expect from each one
of these changes.
Our implementation, in the form of a SQLite database, will be referred to as
network from this point on.

Ensuring data consistency as each portion of the data is edited is a two part
problem:

1. Knowing what to do when a certain edit is attempted by the user
2. Automatically applying the tests and consistency checks (and changes)
   required on one

.. _net_section.1:

Change behavior
^^^^^^^^^^^^^^^

In this section we present the mapping of all meaningful operations that a user
can do to links and nodes, and you can use the table below to navigate between
each of the changes to see how they are treated through triggers.

.. table::
   :align: center

+--------------------------------------+-----------------------------------+
| Nodes                                |     Links                         |
+======================================+===================================+
| :ref:`net_creating_nodes`            | :ref:`net_deleting_link`          |
+--------------------------------------+-----------------------------------+
| :ref:`net_deleting_nodes`            | :ref:`net_moving_link_extremity`  |
+--------------------------------------+-----------------------------------+
| :ref:`net_moving_node`               | :ref:`net_reshaping_link`         |
+--------------------------------------+-----------------------------------+
| :ref:`net_add_node_field`            | :ref:`net_deleting_reqfield_link` |
+--------------------------------------+-----------------------------------+
| :ref:`net_deleting_node_field`       |                                   |
+--------------------------------------+-----------------------------------+
| :ref:`net_modifying_node_data_entry` |                                   |
+--------------------------------------+-----------------------------------+

.. _net_section.1.1:

Node layer changes and expected behavior
''''''''''''''''''''''''''''''''''''''''

There are 6 possible changes envisioned for the network nodes layer, being 3 of
geographic nature and 3 of data-only nature. The possible variations for each
change are also discussed, and all the points where alternative behavior is
conceivable are also explored.

.. _net_creating_nodes:

Creating a node
```````````````

There are only three situations when a node is to be created:

- Placement of a link extremity (new or moved) at a position where no node
  already exists

- Splitting a link in the middle

- Creation of a centroid for later connection to the network

In all cases a unique node ID needs to be generated for the new node, and all
other node fields should be empty.

An alternative behavior would be to allow the user to create nodes with no
attached links. Although this would not result in inconsistent networks for
traffic and transit assignments, this behavior would not be considered valid.
All other edits that result in the creation of unconnected nodes or that result
in such case should result in an error that prevents such operation

Behavior regarding the fields regarding modes and link types is discussed in
their respective table descriptions

.. _net_deleting_nodes:

Deleting a node
```````````````

Deleting a node is only allowed in two situations:

- No link is connected to such node (in this case, the deletion of the node
  should be handled automatically when no link is left connected to such node)

- When only two links are connected to such node. In this case, those two links
  will be merged, and a standard operation for computing the value of each field
  will be applied.

For simplicity, the operations are: Weighted average for all numeric fields,
copying the fields from the longest link for all non-numeric fields. Length is
to be recomputed in the native distance measure of distance for the projection
being used.

A node can only be eliminated as a consequence of all links that terminated/
originated at it being eliminated. If the user tries to delete a node, the
network should return an error and not perform such operation.

Behavior regarding the fields regarding modes and link types is discussed in
their respective table descriptions

.. _net_moving_node:

Moving a node
`````````````

There are two possibilities for moving a node: Moving to an empty space, and
moving on top of another node.

- **If a node is moved to an empty space**

All links originated/ending at that node will have its shape altered to conform
to that new node position and keep the network connected. The alteration of the
link happens only by changing the Latitude and Longitude of the link extremity
associated with that node.

- **If a node is moved on top of another node**

All the links that connected to the node on the bottom have their extremities
switched to the node on top
The node on the bottom gets eliminated as a consequence of the behavior listed
on :ref:`net_deleting_nodes`

Behavior regarding the fields regarding modes and link types is discussed in
their respective table descriptions

.. _net_add_node_field:

Adding a data field
```````````````````

No consistency check is needed other than ensuring that no repeated data field
names exist

.. _net_deleting_node_field:

Deleting a data field
`````````````````````

If the data field whose attempted deletion is mandatory, the network should
return an error and not perform such operation. Otherwise the operation can be
performed.

.. _net_modifying_node_data_entry:

Modifying a data entry
``````````````````````

If the field being edited is the node_id field, then all the related tables need
to be edited as well (e.g. a_b and b_node in the link layer, the node_id tagged
to turn restrictions and to transit stops)

.. _net_section.1.2:

Link layer changes and expected behavior
''''''''''''''''''''''''''''''''''''''''

Network links layer also has some possible changes of geographic and data-only nature.

.. _net_deleting_link:

Deleting a link
`````````````````

In case a link is deleted, it is necessary to check for orphan nodes, and deal
with them as prescribed in :ref:`net_deleting_nodes`. In case one of the link
extremities is a centroid (i.e. field *is_centroid*=1), then the node should not
be deleted even if orphaned.

Behavior regarding the fields regarding modes and link types is discussed in
their respective table descriptions.

.. _net_moving_link_extremity:

Moving a link extremity
```````````````````````

This change can happen in two different forms:

- **The link extremity is moved to an empty space**

In this case, a new node needs to be created, according to the behavior
described in :ref:`net_creating_nodes` . The information of node ID (A or B
node, depending on the extremity) needs to be updated according to the ID for
the new node created.

- **The link extremity is moved from one node to another**

The information of node ID (A or B node, depending on the extremity) needs to be
updated according to the ID for the node the link now terminates in.

Behavior regarding the fields regarding modes and link types is discussed in
their respective table descriptions.

.. _net_reshaping_link:

Re-shaping a link
`````````````````

When reshaping a link, the only thing other than we expect to be updated in the
link database is their length (or distance, in AequilibraE's field structure).
As of now, distance in AequilibraE is **ALWAYS** measured in meters.

.. _net_deleting_reqfield_link:

Deleting a required field
`````````````````````````
Unfortunately, SQLite does not have the resources to prevent a user to remove a
data field from the table. For this reason, if the user removes a required
field, they will most likely corrupt the project.


.. _net_section.1.3:

Field-specific data consistency
'''''''''''''''''''''''''''''''
Some data fields are specially sensitive to user changes.

.. _net_change_link_distance:

Link distance
`````````````

Link distance cannot be changed by the user, as it is automatically recalculated
using the Spatialite function *GeodesicLength*, which always returns distances
in meters.

.. _net_change_link_direc:

Link direction
``````````````

Triggers enforce link direction to be -1, 0 or 1, and any other value results in
an SQL exception.

.. _net_change_link_modes:

*modes* field (Links and Nodes layers)
``````````````````````````````````````
A serious of triggers are associated with the modes field, and they are all
described in the :ref:`tables_modes`.

.. _net_change_link_ltypes:
*link_type* field (Links layer) & *link_types* field (Nodes layer)
``````````````````````````````````````````````````````````````````
A serious of triggers are associated with the modes field, and they are all
described in the :ref:`tables_link_types`.

.. _net_change_link_node_ids:
a_node and b_node
`````````````````
The user should not change the a_node and b_node fields, as they are controlled
by the triggers that govern the consistency between links and nodes. It is not
possible to enforce that users do not change these two fields, as it is not
possible to choose the trigger application sequence in SQLite
