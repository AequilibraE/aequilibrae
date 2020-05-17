.. _tables_modes:

==============
Modes table
==============

The **modes** table exists to list all the modes available in the model's network,
and its main role is to support the creation of graphs directly from the SQLite
project.

The modes table has three fields, being the *mode_name*, *description* and
*mode_id*, where *mode_id* is a single letter that is used to codify mode
permissions in the network, as further discussed in :ref:`network`.

An example of what the contents of the mode table look like is below:

.. image:: ../images/modes_table.png
    :width: 750
    :align: center
    :alt: Link examples

Consistency triggers
--------------------
As it happens with the links and nodes table (:ref:`network_triggers_behaviour`),
the modes table is kept consistent with the links table through the use of
database triggers

.. _tables_section6.1.1.1:

Changing the modes allowed in a certain link
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Whenever we change the modes allowed on a link, we need to check for two
conditions:

* At least one mode is allowed on that link
* All links allowed on that link exist in the modes table

For each condition, a specific trigger was built, and if any of the checks
fails, the transaction will fail.

Having successfully changed the modes allowed in a link, we need to
update the nodes that are accessible to each of the nodes which are the
extremities of this link. For this purpose, a further trigger is created
to update the modes field in the nodes table for both of the link's a_node and
b_node.

Directly changing the modes field in the nodes table
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A trigger guarantees that the value being inserted in the field is according to
the values found in the associated links' modes field.
THIS TRIGGER IS STILL MISSING


.. _tables_section6.1.1.2:

Adding a new link
^^^^^^^^^^^^^^^^^
The exact same behaviour as for :ref:`tables_section6.1.1.1` applies in this
case, but it requires specific new triggers on the **creation** of the link.

.. _tables_section6.1.1.3:

Editing a mode in the modes table
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Whenever we want to edit a mode in the modes table, we need to check for two
conditions:

* The new mode_id is exactly one character long
* The old mode_id is not still in use on the network

For each condition, a specific trigger was built, and if any of the checks
fails, the transaction will fail.

The requirements for uniqueness and non-absent values are guaranteed during the
construction of the modes table by using the keys **UNIQUE** and **NOT NULL**.


.. _tables_section6.1.1.4:

Adding a new mode to the modes table
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In this case, only the first behaviour mentioned above on
:ref:`tables_section6.1.1.3` applies, the verification that the mode_id is
exactly one character long. Therefore only one new trigger is required.

.. _tables_section6.1.1.5:

Removing a mode from the modes table
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In counterpoint, only the second behaviour mentioned above on
:ref:`tables_section6.1.1.3` applies in this case, the verification that the old
mode_id is not still in use by the network. Therefore only one new trigger is
required.