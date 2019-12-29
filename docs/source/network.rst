.. index:: network

Network
=======

As described in the project_ the AequilibraE network is composed of two layers (links
and nodes), ...

The parameters file has quite a few controls for the creation of AequilibraE networks,
and it is important to discuss them


Network Fields
--------------

Links
~~~~~

There is a list of fields unique per direction (e.g. street name) and a list for direction-
specific data (e.g. number of lanes)

A few standard fields.

Nodes
~~~~~


Importing from Open Street Maps
-------------------------------

List of key tags we will import for each mode. Description of tags can be found on
`Open-Street Maps <https://wiki:openstreetmap:org/wiki/Key:highway:>`_, and we recommend
not changing the standard parameters unless you are exactly sure of what you are doing.

For each mode to be imported there is also a mode filter to control for non-default
behaviour. For example, in some cities pedestrians a generally allowed on cycleways, but
they might be forbidden in specific links, which would be tagged as **pedestrian:no**.
This feature is stored under the key *mode_filter* under each mode to be imported.

There is also the possibility that not all keywords for link types for the region being
imported, and therefore unknown link type tags are treated as a special case for each
mode, and that is controlled by the key *unknown_tags* in the parameters file.
