.. _accessing_project_data:

Accessing Project Data
======================

An AequilibraE project helds geometric information that can be accessed by the user in 
three different classes: ``Links``, ``Nodes``, and ``Zoning``. In this section, we'll
cover the main points regarding them.

``project.network.links``
-------------------------

This method allows you to access the API resources to manipulate the 'links' table.
Each item in the 'links' table is a ``Link`` object.

.. code-block:: python

    from aequilibrae.utils.create_example import create_example

    project = create_example("/path_to_my_folder", "coquimbo")

    project_links = project.network.links

    # Let's add a new field to our 'links' table
    project_links.fields.add("my_field", "This is an example", "TEXT")
    # To save this modification, we must refresh the table
    project_links.refresh_fields()

    # Let's add a new link to our project 
    new_link = project_links.new()
    new_link.geometry = LineString([(-71.304754, -29.955233), (-71.304863, -29.954049)])
    new_link.modes = "bctw"
    # To add a new link, it must be explicitly saved
    new_link.save()

    # The 'links' table has three fields which cannot be empty (i.e. with `NULL` values):
    # `link_id`, `direction`, and `modes`. When we create a node, `new` automatically
    # creates a `link_id`, and sets the default value (0) for direction. Thus, the modes
    # information should be added, otherwise, it will raise an error.

    # To delete one link from the project, you can use one of the following
    other_link = project_links.get(21332)
    other_link.delete()
    # or
    project_links.delete(21332)

    # The `copy_link` function creates a copy of a specified link
    # It is very helpful case you want to split a link. 
    # You can check out in one of the usage examples.
    link_copy = project_links.copy_link(10972)

    # Don't forget to save the modifications to the links layer
    project_links.save()

    # And refresh the links in memory for usage
    project_links.refresh()

.. seealso::
    
    * :func:`aequilibrae.project.network.Links`
        Class documentation

    * :ref:`project_from_link_layer` | :ref:`editing_network_splitting_link`
        Usage examples

``project.network.nodes``
-------------------------

This method allows you to access the API resources to manipulate the 'nodes' table.
Each item in the 'nodes' table is a ``Node`` object.

.. code-block:: python

    from aequilibrae.utils.create_example import create_example
    from shapely.geometry import Point

    project = create_example("/path_to_my_folder", "coquimbo")

    project_nodes = project.network.nodes

    # To get one 'Node' object
    node = project_nodes.get(10070)

    # We can check the existing fields for each node in the 'nodes' table
    node.data_fields()

    # Let's renumber this node and save it
    node.renumber(1000)
    node.save()

    # A node can also be used to add a special generator
    # `new_centroid` returns a `Node` object that we can edit
    centroid = project_nodes.new_centroid(2000)

    # Don't forget to add a geometry to your centroid if it's a new node
    # This centroid corresponds to the Port of Coquimbo!
    centroid.geometry = Point(-71.32, -29.94)

    # As this centroid is not associated with a zone, we must tell AequilibraE the initial area around
    # the centroid to look for candidate nodes to which the centroid can connect.
    centroid.connect_mode(area=centroid.geometry.buffer(0.01), mode_id="c")

    # Don't forget to update these changes to the nodes in memory
    project_nodes.refresh()

    # And save them into your project
    project_nodes.save()

    # Last but not less important, you can check your project nodes
    # `project_nodes.data` returns a geopandas GeoDataFrame.
    project_nodes.data

    # or if you want to check the coordinate of each node in the shape of
    # a Pandas DataFrame
    project_nodes.lonlat

.. seealso::

    * :func:`aequilibrae.project.network.Nodes`
        Class documentation

.. _project_zoning:

``project.zoning``
------------------

This method allows you to access the API resources to manipulate the 'zones' table.
Each item in the 'zones' table is a ``Zone`` object.

.. code-block:: python

    from aequilibrae.utils.create_example import create_example
    from shapely.geometry import Polygon, Point

    project = create_example("/path_to_my_folder", "coquimbo")

    project_zones = project.zoning

    # Let's start this example by adding a new field to the 'zones' table
    project_zones.fields.add("parking_spots", "Number of public parking spots", "INTEGER")

    # We can check if the new field was indeed created
    project_zones.fields.all_fields()

    # Now let's get a zone and modifiy it
    zone = project_zones.get(40)
    # By disconnecting the transit mode
    zone.disconnect_mode("t")
    # Connecting the bicycle mode
    zone.connect_mode("b")
    # And adding the number of public parking spots in the field we just created
    zone.parking_spots = 30
    # You can save this modifications if you want
    zone.save()

    # The changes connecting / disconnecting modes reflect in the zone centroids
    # and can be seen in the 'nodes' table.

    # If you want to delete a zone
    other_zone = project_zones.get(38)
    other_zone.delete()

    # Or to add a new one
    zone_extent = Polygon([(-71.3325, -29.9473), (-71.3283, -29.9473), (-71.3283, -29.9539), (-71.3325, -29.9539)])

    new_zone = project_zones.new(38)
    new_zone.geometry = zone_extent
    # We can add a centroid to the zone we just created by specifying its location or
    # pass `None` to use the geometric center of the zone 
    new_zone.add_centroid(Point(-71.33, -29.95))

    # Let's refresh our fields
    project_zones.refresh_geo_index()

    # And save the new changes in the project
    project_zones.save()

    # Finally, to return a geopandas GeoDataFrame with the project zones
    project_zones.data

    # To return a dictionary with all 'Zone' objects in the model
    project_zones.all_zones()

    # To get a Shapely Polygon or Multipolygon with the entire zoning coverage
    project_zones.all_zones()

    # And to get the nearest zone to giver geometry
    project_zones.get_closest_zone(Point(-71.3336, -29.9490))

.. seealso::

    * :func:`aequilibrae.project.Zoning`
        Class documentation

    * :ref:`create_zones`
        Usage example
