.. _network:

Network
~~~~~~~

The network is composed by two tables, **links** and **nodes** that are
connected through both their geometries and specific fields, hence
why they are documented under network and not individually.

.. note::
  This documentation, as well as the SQL code it referes to, comes from the
  seminal work done in `TranspoNet <http://github.com/AequilibraE/TranspoNet/>`_
  by `Pedro <https://au.linkedin.com/in/pedrocamargo>`_ and
  `Andrew <https://au.linkedin.com/in/andrew-o-brien-5a8bb486>`_.

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

As AequilibraE is based on Spatialite, the user is also welcome to use its
powerful tools to manipulate your model's geometries, although that is not
recommended, as the "training wheels are off".

The objectives of developing a network format for AequilibraE are to provide the
users a seamless integration between network data and transportation modeling
algorithms and to allow users to easily edit such networks in any GIS platform
they'd like, while ensuring consistency between network components, namely links
and nodes.

As mentioned in other sections of this documentation, the AequilibraE
network file is composed by a links and a nodes layer that are kept
consistent with each other through the use of database triggers, and
the network can therefore be edited in any GIS platform or
programmatically in any fashion, as these triggers will ensure that
the two layers are kept compatible with each other by either making
other changes to the layers or preventing the changes.

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
   AequilibraE does not currently support turn penalties and/or bans. Their
   implementation requires a complete overahaul of the path-building code, so
   that is still a long-term goal, barred specific developed efforts.

Currently, AequilibraE can create networks from OpenStreetMaps and GMNS, and
also export this network to GMNS format. You can check out more information about
these features in the following pages:

- :ref:`exporting_to_gmns`
- :ref:`importing_from_gmns`
- :ref:`importing_from_osm`


Data consistency
^^^^^^^^^^^^^^^^

One of the key characteristics of any modeling platform is the ability of the
supporting software to maintain internal data consistency. Network data
consistency is surely the most critical and complex aspect of overall data
consistency, which has been introduced in the AequilibraE framework with
`TranspoNET <https://www.github.com/aequilibrae/transponet>`_,  where
`Andrew O'Brien <https://www.linkedin.com/in/andrew-o-brien-5a8bb486/>`_
implemented link-node consistency infrastructure in the form of spatialite
triggers.

**We cannot stress enough how impactful this set of spatial triggers was to**
**the industry, as this is the first time a transportation network can be**
**edited without specialized software that requires the editing to be done**
**inside such software.**

Further data consistency, especially for tabular data, is also necessary. This
need has been largely addressed in version 0.7, but more triggers will most
likely be added as AequilibraE's capabilities grow in complexity and Spatialite
brings more capabilities.

All consistency triggers/procedures are discussed in parallel with the
features they implement.


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

In this section we present the mapping of all meaningful changes that a user can
do to each part of the transportation network, doing so for each element of the
transportation network. You can use the table below to navigate between each of the
meaningful changes documented for nodes and links of your network.

.. table::
   :align: center

+------------------------------+--------------------------+
| Nodes                        |     Links                |
+==============================+==========================+
| :ref:`net_section.1.1.1`     | :ref:`net_section.1.2.1` |
+------------------------------+--------------------------+
| :ref:`net_section.1.1.2`     | :ref:`net_section.1.2.2` |
+------------------------------+--------------------------+
| :ref:`net_section.1.1.3`     | :ref:`net_section.1.2.3` |
+------------------------------+--------------------------+
| :ref:`net_section.1.1.4`     | :ref:`net_section.1.2.6` |
+------------------------------+--------------------------+
| :ref:`net_section.1.1.5`     |                          |
+------------------------------+--------------------------+
| :ref:`net_section.1.1.6`     |                          |
+------------------------------+--------------------------+


.. _net_section.1.1:

Node layer changes and expected behavior
''''''''''''''''''''''''''''''''''''''''

There are 6 possible changes envisioned for the network nodes layer, being 3 of
geographic nature and 3 of data-only nature. The possible variations for each
change are also discussed, and all the points where alternative behavior is
conceivable are also explored.

.. _net_section.1.1.1:

Creating a node
```````````````

There are only three situations when a node is to be created:
- Placement of a link extremity (new or moved) at a position where no node
already exists
- Spliting a link in the middle
- Creation of a centroid for later connection to the network

In both cases a unique node ID needs to be generated for the new node, and all
other node fields should be empty
An alternative behavior would be to allow the user to create nodes with no
attached links. Although this would not result in inconsistent networks for
traffic and transit assignments, this behavior would not be considered valid.
All other edits that result in the creation of un-connected nodes or that result
in such case should result in an error that prevents such operation

Behavior regarding the fields regarding modes and link types is discussed in
their respective table descriptions

.. _net_section.1.1.2:

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

.. _net_section.1.1.3:

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
on :ref:`net_section.1.1.2`

Behavior regarding the fields regarding modes and link types is discussed in
their respective table descriptions

.. _net_section.1.1.4:

Adding a data field
```````````````````

No consistency check is needed other than ensuring that no repeated data field
names exist

.. _net_section.1.1.5:

Deleting a data field
`````````````````````

If the data field whose attempted deletion is mandatory, the network should
return an error and not perform such operation. Otherwise the operation can be
performed.

.. _net_section.1.1.6:

Modifying a data entry
``````````````````````

If the field being edited is the node_id field, then all the related tables need
to be edited as well (e.g. a_b and b_node in the link layer, the node_id tagged
to turn restrictions and to transit stops)

.. _net_section.1.2:

Link layer changes and expected behavior
''''''''''''''''''''''''''''''''''''''''

Network links layer also has some possible changes of geographic and data-only nature.

.. note::
   AequilibraE's link layer manipulation has other geographic nature
   changes to be implemented.

.. _net_section.1.2.1:

Deleting a link
`````````````````

In case a link is deleted, it is necessary to check for orphan nodes, and deal
with them as prescribed in :ref:`net_section.1.1.2`

Behavior regarding the fields regarding modes and link types is discussed in
their respective table descriptions.


.. _net_section.1.2.2:

Moving a link extremity
```````````````````````

This change can happen in two different forms:

- **The link extremity is moved to an empty space**

In this case, a new node needs to be created, according to the behavior
described in :ref:`net_section.1.1.1` . The information of node ID (A or B
node, depending on the extremity) needs to be updated according to the ID for
the new node created.

- **The link extremity is moved from one node to another**

The information of node ID (A or B node, depending on the extremity) needs to be
updated according to the ID for the node the link now terminates in.

Behavior regarding the fields regarding modes and link types is discussed in
their respective table descriptions.

.. _net_section.1.2.3:

Re-shaping a link
`````````````````

When reshaping a link, the only thing other than we expect to be updated in the
link database is their length (or distance, in AequilibraE's field structure).
As of now, distance in AequilibraE is **ALWAYS** measured in meters.

.. _net_section.1.2.6:

Deleting a required field
`````````````````````````
Unfortunately, SQLite does not have the resources to prevent a user to remove a
data field from the table. For this reason, if the user removes a required
field, they will most likely corrupt the project.


.. _net_section.1.3:

Field-specific data consistency
'''''''''''''''''''''''''''''''
Some data fields are specially sensitive to user changes.

.. _net_section.1.3.1:

Link distance
`````````````

Link distance cannot be changed by the user, as it is automatically recalculated
using the Spatialite function *GeodesicLength*, which always returns distances
in meters.

.. _net_section.1.3.2:

Link direction
``````````````

Triggers enforce link direction to be -1, 0 or 1, and any other value results in
an SQL exception.

.. _net_section.1.3.3:

*modes* field (Links and Nodes layers)
``````````````````````````````````````
A serious of triggers are associated with the modes field, and they are all
described in the :ref:`tables_modes`.

.. _net_section.1.3.4:
*link_type* field (Links layer) & *link_types* field (Nodes layer)
``````````````````````````````````````````````````````````````````
A serious of triggers are associated with the modes field, and they are all
described in the :ref:`tables_link_types`.

.. _net_section.1.3.5:
a_node and b_node
`````````````````
The user should not change the a_node and b_node fields, as they are controlled
by the triggers that govern the consistency between links and nodes. It is not
possible to enforce that users do not change these two fields, as it is not
possible to choose the trigger application sequence in SQLite
