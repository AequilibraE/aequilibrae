.. _network:

=======
Network
=======

.. note::
  This documentation, as well as the SQL code it referred to, comes from the
  seminal work done in `TranspoNet <http://github.com/AequilibraE/TranspoNet/>`_
  by `Pedro <https://au.linkedin.com/in/pedrocamargo>`_ and
  `Andrew <https://au.linkedin.com/in/andrew-o-brien-5a8bb486>`_.

The objectives of developing a network format for AequilibraE are to provide the
users a seamless integration between network data and  transportation modeling
algorithms and to allow users to easily edit such networks in any GIS platform
they'd like, while ensuring consistency between network components, namely links
and nodes.

As mentioned in other sections of this documentation, the AequilibraE
network file is composed by a links and a nodes layer that are kept
consistent with each other through the use of database triggers, and
the network can therefore be edited in any GIS platform or
programatically in any fashion, while these triggers will ensure that
the two layers are kept compatible with each other by either making
other changes to the layers or preventing the changes.

Although the behaviour of these trigger is expected to be mostly intuitive
to anybody used to editing transportation networks within commercial modeling
platforms, we have detailed the behaviour for all different network changes in
:ref:`net_section.1` .

This implementation choice is not, however, free of caveats. Due to
technological limitations of SQLite, some of the desired behaviors identified in
:ref:`net_section.1` cannot be implemented, but such caveats do not impact the
usefulness of this implementation or its robustness in face proper use of the
tool.

.. note::
   AequilibraE does not currently support turn penalties and/or bans. Their
   implementation requires a complete overahaul of the path-building code, so
   that is still a long-term goal, barred specific developed efforts.

Network Fields
--------------

As described in the :ref:`project` the AequilibraE network is composed of two layers (links
and nodes), detailed below.

Links
~~~~~

Network links are defined by geographic elements of type LineString (No
MultiLineString allowed) and a series of mandatory fields, as well a series of
other optional fields that might be required for documentation and display
purposes (e.g. street names) or by specific applications (e.g. parameters for
Volume-Delay functions, hazardous vehicles restrictions, etc.).

**The mandatory fields are the following**

+-------------+-----------------------------------------------------------------------+-------------------------+
|  Field name |                           Field Description                           |        Data Type        |
+=============+=======================================================================+=========================+
| link_id     | Unique identifier                                                     | Integer (32/64 bits)    |
+-------------+-----------------------------------------------------------------------+-------------------------+
| a_node      | node_id of the first (topologically) node of the link                 | Integer (32/64 bits)    |
+-------------+-----------------------------------------------------------------------+-------------------------+
| b_node      | node_id of the last (topologically) node of the link                  | Integer (32/64 bits)    |
+-------------+-----------------------------------------------------------------------+-------------------------+
| direction   | Direction of flow allowed for the link (A-->B: 1, B-->A:-1, Both:0)   | Integer 8 bits          |
+-------------+-----------------------------------------------------------------------+-------------------------+
| distance    | Length of the link in meters                                          | Float 64 bits           |
+-------------+-----------------------------------------------------------------------+-------------------------+
| modes       | Modes allowed in this link. (Concatenation of mode ids)               | String                  |
+-------------+-----------------------------------------------------------------------+-------------------------+
| link_type   | Link type classification. Can be the highway tag for OSM or other     | String                  |
+-------------+-----------------------------------------------------------------------+-------------------------+


**The optional fields may include, but are not limited to the following:**

+-----------------------+------------------------------------------------------------------+----------------+
| Field name            | Field description                                                | Data type      |
+=======================+==================================================================+================+
| Street name           | Cadastre name of the street                                      | String         |
+-----------------------+------------------------------------------------------------------+----------------+
| capacity_ab           | Modeling capacity of the link for the direction A --> B          | Float 32 bits  |
+-----------------------+------------------------------------------------------------------+----------------+
| capacity_ba           | Modeling capacity of the link for the direction B --> A          | Float 32 bits  |
+-----------------------+------------------------------------------------------------------+----------------+
| speed_ab              | Modeling (Free flow) speed for the link in the A --> B direction | Float 32 Bits  |
+-----------------------+------------------------------------------------------------------+----------------+
| speed_ab              | Modeling (Free flow) speed for the link in the B --> A direction | Float 32 bits  |
+-----------------------+------------------------------------------------------------------+----------------+
| volume delay function | Type of volume delay function to be used on that link            | String         |
+-----------------------+------------------------------------------------------------------+----------------+
| alfa_ab               | Alfa parameter for the BPR for the A->B direction of link        | Float 32 bits  |
+-----------------------+------------------------------------------------------------------+----------------+
| alfa_ba               | Alfa parameter for the BPR for the B->A direction of link        | Float 32 bits  |
+-----------------------+------------------------------------------------------------------+----------------+
| beta_ab               | Beta parameter for the BPR for the A->B direction of link        | Float 32 bits  |
+-----------------------+------------------------------------------------------------------+----------------+
| beta_ba               | Beta parameter for the BPR for the B->A direction of link        | Float 32 bits  |
+-----------------------+------------------------------------------------------------------+----------------+
| lanes_ba              | Number of lanes of the link for the direction A->B               | Integer 8 bits |
+-----------------------+------------------------------------------------------------------+----------------+
| lanes_ba              | Number of lanes of the link for the direction B->A               | Integer 8 bits |
+-----------------------+------------------------------------------------------------------+----------------+
| ...                   | ...                                                              | ...            |
+-----------------------+------------------------------------------------------------------+----------------+

Nodes
~~~~~

The nodes table only has four mandatory fields as of now: *node_id*, which are
directly linked to *a_node* and *b_node* in the links table through a series of
database triggers, *is_centroid*, which is a binary 1/0 value identifying nodes
as centroids (1) or not (0).

The fields for **mode** and **link_types** are linked to the **modes** and
**link_type** fields from the links layer through a series of triggers, and
cannot be safely edited by the user (nor there is reason for such).

+-------------+-----------------------------------------------------------------------+-------------------------+
|  Field name |                           Field Description                           |        Data Type        |
+=============+=======================================================================+=========================+
| node_id     | Unique identifier. Tied to the link table's a_node & b_node           | Integer (32/64 bits)    |
+-------------+-----------------------------------------------------------------------+-------------------------+
| is_centroid | node_id of the first (topologically) node of the link                 | Integer (32/64 bits)    |
+-------------+-----------------------------------------------------------------------+-------------------------+
| modes       | Concatenation of all mode_ids of all links connected to the node      | String                  |
+-------------+-----------------------------------------------------------------------+-------------------------+
| link_types  | Concatenation of all link_type_ids of all links connected to the node | String                  |
+-------------+-----------------------------------------------------------------------+-------------------------+

**The optional fields may include, but are not limited to the following:**

+-------------+-----------------------------------------------------------------------+-------------------------+
|  Field name |                           Field Description                           |        Data Type        |
+=============+=======================================================================+=========================+
| taz         | Zone in which the zone is located                                     | Integer (32/64 bits)    |
+-------------+-----------------------------------------------------------------------+-------------------------+
| ...         | ...                                                                   | ...                     |
+-------------+-----------------------------------------------------------------------+-------------------------+

It is good practice when working with the sqlite to keep all field names without
spaces and all lowercase.

Future components
~~~~~~~~~~~~~~~~~

3.	Turn penalties/restrictions

4.	Transit routes

5.	Transit stops

.. _importing_from_osm:

Importing from Open Street Maps
-------------------------------

Please review the information :ref:`parameters`

.. note::

   **ALL links that cannot be imported due to errors in the SQL insert**
   **statements are written to the log file with error message AND the SQL**
   **statement itself, and therefore errors in import can be analyzed for**
   **re-downloading or fixed by re-running the failed SQL statements after**
   **manual fixing**

.. _sqlite_python_limitations:

Python limitations
~~~~~~~~~~~~~~~~~~
As it happens in other cases, Python's usual implementation of SQLite is
incomplete, and does not include R-Tree, a key extension used by Spatialite for
GIS operations.

For this reason, AequilibraE's default option when importing a network from OSM
is to **NOT create spatial indices**, which renders the network consistency
triggers useless.

If you are using a vanilla Python installation (your case if you are not sure),
you can import the network without creating indices, as shown below.

::

  from aequilibrae.project import Project

  p = Project()
  p.new('path/to/project/new/folder')
  p.network.create_from_osm(place_name='my favorite place')
  p.conn.close()

And then manually add the spatial index on QGIS by adding both links and nodes
layers to the canvas, and selecting properties and clicking on *create spatial*
*index* for each layer at a time. This action automatically saves the spatial
indices to the sqlite database.

.. image:: images/qgis_creating_spatial_indices.png
    :width: 1383
    :align: center
    :alt: Adding Spatial indices with QGIS

If you are an expert user and made sure your Python installation was compiled
against a complete SQLite set of extensions, then go ahead an import the network
with the option for creating such indices.

::

  from aequilibrae.project import Project

  p = Project()
  p.new('path/to/project/new/folder/')
  p.network.create_from_osm(place_name='my favorite place', spatial_index=True)
  p.conn.close()

If you want to learn a little more about this topic, you can access this
`blog post <https://pythongisandstuff.wordpress.com/2015/11/11/python-and-spatialite-32-bit-on-64-bit-windows/>`_
or the SQLite page on `R-Tree <https://www.sqlite.org/rtree.html>`_.

Please also note that the network consistency triggers will NOT work before
spatial indices have been created and/or if the editing is being done on a
platform that does not support both RTree and Spatialite.

.. _network_triggers_behaviour:

Network consistency behaviour
-----------------------------

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
~~~~~~~~~~~~~~~

In this section we present the mapping of all meaningful changes that a user can
do to each part of the transportation network, doing so for each element of the
transportation network.

.. _net_section.1.1:

Node layer changes and expected behavior
++++++++++++++++++++++++++++++++++++++++

There are 6 possible changes envisioned for the network nodes layer, being 3 of
geographic nature and 3 of data-only nature. The possible variations for each
change are also discussed, and all the points where alternative behavior is
conceivable are also explored.

.. _net_section.1.1.1:

Creating a node
^^^^^^^^^^^^^^^

There are only two situations when a node is to be created:
- Placement of a link extremity (new or moved) at a position where no node
already exists
- Spliting a link in the middle

In both cases a unique node ID needs to be generated for the new node, and all
other node fields should be empty
An alternative behavior would be to allow the user to create nodes with no
attached links. Although this would not result in inconsistent networks for
traffic and transit assignments, this behavior would not be considered valid.
All other edits that result in the creation of un-connected nodes or that result
 in such case should result in an error that prevents such operation

.. _net_section.1.1.2:

Deleting a node
^^^^^^^^^^^^^^^

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

.. _net_section.1.1.3:

Moving a node
^^^^^^^^^^^^^

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

.. _net_section.1.1.4:

Adding a data field
^^^^^^^^^^^^^^^^^^^

No consistency check is needed other than ensuring that no repeated data field
names exist

.. _net_section.1.1.5:

Deleting a data field
^^^^^^^^^^^^^^^^^^^^^

If the data field whose attempted deletion is mandatory, the network should
return an error and not perform such operation. Otherwise the operation can be
performed.

.. _net_section.1.1.6:

Modifying a data entry
^^^^^^^^^^^^^^^^^^^^^^

If the field being edited is the node_id field, then all the related tables need
to be edited as well (e.g. a_b and b_node in the link layer, the node_id tagged
to turn restrictions and to transit stops)

.. _net_section.1.2:

Link layer changes and expected behavior
++++++++++++++++++++++++++++++++++++++++

There are 8 possible changes envisioned for the network links layer, being 5 of
geographic nature and 3 of data-only nature.

.. _net_section.1.2.1:

Deleting a link
^^^^^^^^^^^^^^^
A link cannot be deleted if there are other elements associated with it. These
elements are:

* Transit routes
* turn penalties

In case a link is deleted, it is necessary to check for orfan nodes, and deal
with them as prescribed in :ref:`net_section.1.1.2`

.. _net_section.1.2.2:

Moving a link extremity
^^^^^^^^^^^^^^^^^^^^^^^

This change can happen in two different forms:

- **The link extremity is moved to an empty space**

In this case, a new node needs to be created, according to the behavior
described in :ref:`net_section.1.1.1` . The information of node ID (A or B
node, depending on the extremity) needs to be updated according to the ID for
the new node created.

- **The link extremity is moved from one node to another**

The information of node ID (A or B node, depending on the extremety) needs to be
updated according to the ID for the node the link now terminates in.

.. _net_section.1.2.3:

Re-shaping a link
^^^^^^^^^^^^^^^^^

When reshaping a link, the only thing other than we expect to be updated in the
link database is their length (or distance, in AequilibraE's field structure).
As of now, distance in AequilibraE is **ALWAYS** measured in meters.

.. .. _net_section.1.2.4:

.. Splitting a link
.. ^^^^^^^^^^^^^^^^
.. *To come*

.. _net_section.1.2.5:

.. Merging two links
.. ^^^^^^^^^^^^^^^^^
.. *To come*

.. _net_section.1.2.6:

Deleting a required field
^^^^^^^^^^^^^^^^^^^^^^^^^
Unfortunately, SQLite does not have the resources to prevent a user to remove a
data field from the table. For this reason, if the user removes a required
field, they will most likely corrupt the project.


.. _net_section.1.3:

Field-specific data consistency
++++++++++++++++++++++++++++++
 Some data fields are specially


.. _net_section.1.3.1:

Link distance
^^^^^^^^^^^^^

Link distance cannot be changed by the user, as it is automatically recalculated
using the Spatialite function *GeodesicLength*, which always returns distances
in meters.

.. _net_section.1.3.2:

Link direction
^^^^^^^^^^^^^^

Triggers enforce link direction to be -1, 0 or 1, and any other value results in
an SQL exception.


.. _net_section.1.3.3:

*modes* field
^^^^^^^^^^^^^
Editing of the modes field will only be allowed to contain a string of mode_ids
that exist in the *modes* table, and an error will be thrown if the user
attempts to leave the field empty or to insert a non-existing mode_id.



# 4	References
http://tfresource.org/Category:Transportation_networks

# 5	Authors

## Pedro Camargo
- www.xl-optim.com
-