Components
===========

An AequilibraE project helds geometric information that can be accessed by the user in 
three different classes: ``Links``, ``Nodes``, and ``Zoning``. We'll first cover these classes, and
then we'll go over the project components without geo-spatial information.

``project.network.links``
-------------------------

This method allows you to access the API resources to manipulate the 'links' table.
Each item in the 'links' table is a ``Link`` object.

.. code-block:: python

    >>> from shapely.geometry import LineString

    >>> project = create_example(project_path, "coquimbo")

    >>> project_links = project.network.links

    # Let's add a new field to our 'links' table
    >>> project_links.fields.add("my_field", "This is an example", "TEXT")
    
    # To save this modification, we must refresh the table
    >>> project_links.refresh_fields()

    # Let's add a new link to our project 
    >>> new_link = project_links.new()
    >>> new_link.geometry = LineString([(-71.304754, -29.955233), (-71.304863, -29.954049)])
    >>> new_link.modes = "bctw"
    
    # To add a new link, it must be explicitly saved
    >>> new_link.save()

    # The 'links' table has three fields which cannot be empty (i.e. with `NULL` values):
    # `link_id`, `direction`, and `modes`. When we create a node, `new` automatically
    # creates a `link_id`, and sets the default value (0) for direction. Thus, the modes
    # information should be added, otherwise, it will raise an error.

    # To delete one link from the project, you can use one of the following
    >>> other_link = project_links.get(21332)
    >>> other_link.delete()
    
    # or
    >>> project_links.delete(21337)

    # The `copy_link` function creates a copy of a specified link
    # It is very helpful case you want to split a link. 
    # You can check out in one of the usage examples.
    >>> link_copy = project_links.copy_link(10972)

    # Don't forget to save the modifications to the links layer
    >>> project_links.save()

    # And refresh the links in memory for usage
    >>> project_links.refresh()

.. admonition:: References

    * :ref:`modifications_on_links_layer`

.. seealso::
    
    * :func:`aequilibrae.project.network.Links`
        Class documentation 
    * :ref:`project_from_link_layer`
        Usage example
    * :ref:`editing_network_splitting_link`
        Usage example

``project.network.nodes``
-------------------------

This method allows you to access the API resources to manipulate the 'nodes' table.
Each item in the 'nodes' table is a ``Node`` object.

.. code-block:: python

    >>> from shapely.geometry import Point

    >>> project_nodes = project.network.nodes

    # To get one 'Node' object
    >>> node = project_nodes.get(10070)

    # We can check the existing fields for each node in the 'nodes' table
    >>> node.data_fields()
    ['node_id', 'is_centroid', 'modes', 'link_types', 'geometry', 'osm_id']

    # Let's renumber this node and save it
    >>> node.renumber(1000)
    >>> node.save()

    # A node can also be used to add a special generator
    # `new_centroid` returns a `Node` object that we can edit
    >>> centroid = project_nodes.new_centroid(2000)

    # Don't forget to add a geometry to your centroid if it's a new node
    # This centroid corresponds to the Port of Coquimbo!
    >>> centroid.geometry = Point(-71.32, -29.94)

    # As this centroid is not associated with a zone, we must tell AequilibraE the initial area around
    # the centroid to look for candidate nodes to which the centroid can connect.
    >>> centroid.connect_mode(area=centroid.geometry.buffer(0.01), mode_id="c")

    # Don't forget to update these changes to the nodes in memory
    >>> project_nodes.refresh()

    # And save them into your project
    >>> project_nodes.save()

    # Last but not less important, you can check your project nodes
    # `project_nodes.data` returns a geopandas GeoDataFrame.
    >>> nodes_data = project_nodes.data

    >>> # or if you want to check the coordinate of each node in the shape of
    >>> # a Pandas DataFrame
    >>> coords = project_nodes.lonlat
    >>> coords.head(3) # doctest: +NORMALIZE_WHITESPACE
      node_id        lon        lat
    0   10037 -71.315117 -29.996804
    1   10064 -71.336604 -29.949050
    2   10065 -71.336517 -29.949062

.. admonition:: References

    * :ref:`modifications_on_nodes_layer`

.. seealso::

    * :func:`aequilibrae.project.network.Nodes`
        Class documentation
    * :ref:`editing_network_nodes`
        Usage example


.. _project_zoning:

``project.zoning``
------------------

This method allows you to access the API resources to manipulate the 'zones' table.
Each item in the 'zones' table is a ``Zone`` object.

.. code-block:: python

    >>> from shapely.geometry import Polygon

    >>> project_zones = project.zoning

    # Let's start this example by adding a new field to the 'zones' table
    >>> project_zones.fields.add("parking_spots", "Number of public parking spots", "INTEGER")

    # We can check if the new field was indeed created
    >>> project_zones.fields.all_fields() # doctest: +ELLIPSIS
    ['area', 'employment', 'geometry', 'name', 'parking_spots', 'population', 'zone_id']

    # Now let's get a zone and modifiy it
    >>> zone = project_zones.get(40)
    
    # By disconnecting the transit mode
    >>> zone.disconnect_mode("t")
    
    # Connecting the bicycle mode
    >>> zone.connect_mode("b")
    
    # And adding the number of public parking spots in the field we just created
    >>> zone.parking_spots = 30
    
    # You can save this changes if you want
    >>> zone.save()

    # The changes connecting / disconnecting modes reflect in the zone centroids
    # and can be seen in the 'nodes' table.

    # To return a dictionary with all 'Zone' objects in the model
    >>> project_zones.all_zones() # doctest: +ELLIPSIS
    {1: ..., ..., 133: ...}

    # If you want to delete a zone
    >>> other_zone = project_zones.get(38)
    >>> other_zone.delete()

    # Or to add a new one
    >>> zone_extent = Polygon([(-71.3325, -29.9473), (-71.3283, -29.9473), (-71.3283, -29.9539), (-71.3325, -29.9539)])

    >>> new_zone = project_zones.new(38)
    >>> new_zone.geometry = zone_extent

    # We can add a centroid to the zone we just created by specifying its location or
    # pass `None` to use the geometric center of the zone 
    >>> new_zone.add_centroid(Point(-71.33, -29.95))

    # Let's refresh our fields
    >>> project_zones.refresh_geo_index()

    # And save the new changes in the project
    >>> project_zones.save()

    # Finally, to return a geopandas GeoDataFrame with the project zones
    >>> zones = project_zones.data

    # To get a Shapely Polygon or Multipolygon with the entire zoning coverage
    >>> boundaries = project_zones.coverage()

    # And to get the nearest zone to a given geometry
    >>> project_zones.get_closest_zone(Point(-71.3336, -29.9490))
    57

    >>> project.close()

.. seealso::

    * :func:`aequilibrae.project.Zoning`
        Class documentation
    * :ref:`create_zones`
        Usage example


``project.about``
-----------------

This class provides an interface for editing the 'about' table of a project. We can add new fields or
edit the existing ones as necessary, but everytime you add or modify a field, you have to write back
this information, otherwise it will be lost.

.. doctest::

    >>> project = Project()
    >>> project.open("/tmp/accessing_sfalls_data")

    >>> project.about.add_info_field("my_new_field")
    >>> project.about.my_new_field = "add some useful information about the field"
    
    # We can add data to an existing field
    >>> project.about.author = "Your Name" 

    # And save our modifications
    >>> project.about.write_back()

    # To assert if 'my_new_field' was added to the 'about' table, we can check the characteristics 
    # stored in the table by returning a list with all characteristics in the 'about' table
    >>> project.about.list_fields() # doctest: +ELLIPSIS
    ['model_name', ..., 'my_new_field']

    # The 'about' table is created automatically when a project is created, but if you're 
    # loading a project created with an older AequilibraE version that didn't contain it, 
    # it is possible to create one too.
    >>> project.about.create()

    >>> project.close()

.. seealso::

    * :func:`aequilibrae.project.About`
        Class documentation
    * :ref:`tables_about`
        Table documentation

``project.FieldEditor``
-----------------------

The ``FieldEditor`` allows the user to edit the project data tables, and it has two different purposes:

* Managing data tables, through the addition/deletion of fields
* Editing the tables' metadata (aka the description of each field)

This class is directly accessed from within the corresponding module one wants to edit.

.. doctest::

    >>> project = Project()
    >>> project.open("/tmp/accessing_nauru_data")

    # We'll edit the fields in the 'nodes' table
    >>> node_fields = project.network.nodes.fields

    # To add a new field to the 'nodes' table
    >>> node_fields.add("my_new_field", "this is an example of AequilibraE's funcionalities", "TEXT")

    # Don't forget to save these modifications
    >>> node_fields.save()

    # To edit the description of a field
    >>> node_fields.osm_id = "number of the osm node_id"

    # Or just to access the description of a field
    >>> node_fields.modes
    'Modes connected to the node'

    # One can also check all the fields in the 'nodes' table.
    >>> node_fields.all_fields() # doctest: +ELLIPSIS
    ['is_centroid', ..., 'my_new_field']

    >>> project.close()

All field descriptions are kept in the table 'attributes_documentation'.

.. seealso::

    *  :func:`aequilibrae.project.FieldEditor`
        Class documentation

``project.log``
---------------

Every AequilibraE project contains a log file that holds information on all the project procedures.
It is possible to access the log file contents, as presented in the next code block.

.. doctest::

    >>> project = Project()
    >>> project.open("/tmp/accessing_nauru_data")

    >>> project_log = project.log()

    # Returns a list with all entires in the log file.
    >>> print(project_log.contents()) # doctest: +ELLIPSIS
    ['2021-01-01 15:52:03,945;aequilibrae;INFO ; Created project on D:/release/Sample models/nauru', ...]

    # If your project's log is getting cluttered, it is possible to clear it. 
    # Use this option wiesly once the deletion of data in the log file can't be undone.
    >>> project_log.clear()

    >>> project.close()

.. seealso::
    
    * :func:`aequilibrae.project.Log`
        Class documentation
    * :ref:`useful-log-tips`
        Usage example
    
``project.matrices``
--------------------

This method ia a gateway to all the matrices available in the model, which allows us to update the
records in the 'matrices' table. Each item in the 'matrices' table  is a ``MatrixRecord`` object.

.. doctest::

    >>> project = Project()
    >>> project.open("/tmp/accessing_sfalls_data")

    >>> matrices = project.matrices

    # One can also check all the project matrices as a Pandas' DataFrame
    >>> matrices.list() # doctest: +SKIP

    # We can add a naw matrix
    >>> matrices.new_record() # doctest: +SKIP
    
    # To delete a matrix from the 'matrices' table, we can delete the record directly
    >>> matrices.delete_record("demand_mc")
    
    # or by selecting the matrix and deleting it
    >>> mat_record = matrices.get_record("demand_omx")
    >>> mat_record.delete()

    # If you're unsure if you have a matrix in you project, you can check if it exists
    # This function will return `True` or `False`
    >>> matrices.check_exists("my_matrix")
    False

    # If a matrix was added or deleted by an external process, you should update or clean
    # your 'matrices' table to keep your project organised.
    >>> matrices.update_database()  # in case of addition
    
    >>> matrices.clear_database()  # in case of deletion

    # To reload the existing matrices in memory once again
    >>> matrices.reload()

    # Similar to the `get_record` function, we have the `get_matrix`, which allows you to
    # get an AequilibraE matrix.
    >>> matrices.get_matrix("demand_aem") # doctest: +SKIP

    >>> project.close()

.. seealso::

    * :func:`aequilibrae.project.Matrices`
        Class documentation
    * :ref:`matrix_table`
        Table documentation

``project.network.link_types``
------------------------------

This method allows you to access the API resources to manipulate the 'link_types' table.
Each item in the 'link_types' table is a ``LinkType`` object.

.. doctest::

    >>> project = Project()
    >>> project.open("/tmp/accessing_coquimbo_data")

    >>> link_types = project.network.link_types

    >>> new_link_type = link_types.new("A")  # Create a new LinkType with ID 'A'

    # We can add information to the LinkType we just created
    >>> new_link_type.description = "This is a description"
    >>> new_link_type.speed = 35
    >>> new_link_type.link_type = "Arterial"

    # To save the modifications for `new_link_type`
    >>> new_link_type.save()

    # To create a new field in the 'link_types' table, you can call the function `fields`
    # to return a FieldEditor instance, which can be edited
    >>> link_types.fields.add("my_new_field", "this is an example of AequilibraE's funcionalities", "TEXT")

    # You can also remove a LinkType from a project using its `link_type_id`
    >>> link_types.delete("A")

    # And don't forget to save the modifications you did in the 'link_types' table
    >>> link_types.save()

    # To check all `LinkTypes` in the project as a dictionary whose keys are the `link_type_id`'s
    >>> link_types.all_types() # doctest: +SKIP
    {'z': <aequilibrae.project.network.link_type.LinkType object at 0x...>} 

    # There are two ways to get a LinkType from the 'link_types' table
    # using the `link_type_id`
    >>> get_link = link_types.get("p")

    # or using the `link_type`
    >>> get_link = link_types.get_by_name("primary")

    >>> project.close()

.. seealso::

    * :func:`aequilibrae.project.network.LinkTypes`
        Class documentation
    * :ref:`tables_link_types`
        Table documentation

``project.network.modes``
-------------------------

This method allows you to access the API resources to manipulate the 'modes' table.
Each item in 'modes' table is a ``Mode`` object.

.. doctest::

    >>> project = Project()
    >>> project.open("/tmp/accessing_coquimbo_data")

    >>> modes = project.network.modes

    # We create a new mode
    >>> new_mode = modes.new("k")
    >>> new_mode.mode_name = "flying_car"

    # And add it to the modes table
    >>> modes.add(new_mode)

    # When we add a new mode to the 'modes' table, it is automatically saved in the table
    # But we can continue editing the modes, and save them as we modify them
    >>> new_mode.description = "Like the one in the cartoons"
    >>> new_mode.save()

    # You can also remove a Mode from a project using its ``mode_id``
    >>> modes.delete("k")

    # To check all `Modes` in the project as a dictionary whose keys are the `mode_id`'s
    >>> modes.all_modes() # doctest: +SKIP
    {'b': <aequilibrae.project.network.mode.Mode object at 0x...>}

    # There are two ways to get a Mode from the 'modes' table
    # using the ``mode_id``
    >>> get_mode = modes.get("c")
    
    # or using the ``mode_name``
    >>> get_mode = modes.get_by_name("car")

    >>> project.close()

.. seealso::

    * :func:`aequilibrae.project.network.Modes`
        Class documentation
    * :ref:`tables_modes`
        Table documentation

``project.network.periods``
---------------------------

This method allows you to access the API resources to manipulate the 'periods' table.
Each item in the 'periods' table is a ``Period`` object.

.. doctest::

    >>> project = Project()
    >>> project.open("/tmp/accessing_coquimbo_data")

    >>> periods = project.network.periods

    # Let's add a new field to our 'periods' table
    >>> periods.fields.add("my_field", "This is field description", "TEXT")

    # To save this modification, we must refresh the table
    >>> periods.refresh_fields()

    # Let's get our default period and change the description for our new field
    >>> select_period = periods.get(1)
    >>> select_period.my_field = "hello world"

    # And we save this period modification
    >>> select_period.save()

    # To see all periods data as a Pandas' DataFrame
    >>> all_periods = periods.data

    # To add a new period
    >>> new_period = periods.new_period(2, 21600, 43200, "6AM to noon")

    # It is also possible to renumber a period
    >>> new_period.renumber(9)

    # And check the existing data fields for each period
    >>> new_period.data_fields()
    ['period_id', 'period_start', 'period_end', 'period_description', 'my_field']

    # Saving can be done after finishing all modifications in the table but for the sake
    # of this example, we'll save the addition of a new period to our table right away
    >>> periods.save()

    >>> project.close()

.. seealso::

    * :func:`aequilibrae.project.network.Periods`
        Class documentation
    * :ref:`tables_period`
        Table documentation
