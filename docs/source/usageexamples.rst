
Use examples
============
This page is still under development, so most of the headers are just place-holders for the actual examples

.. note::
   The examples provided here are not meant as a through description of AequilibraE's capabilities. For that, please
   look into the API documentation or email aequilibrae@googlegroups.com

Paths module
------------


::


  from aequilibrae.paths import allOrNothing
  from aequilibrae.paths import path_computation
  from aequilibrae.paths.results import AssignmentResults as asgr
  from aequilibrae.paths.results import PathResults as pthr


Building a graph
~~~~~~~~~~~~~~~~

Path computation
~~~~~~~~~~~~~~~~

Skimming
~~~~~~~~

Let's suppose you want to compute travel times between all zone on your network. In that case,
you need only a graph that you have previously built, and the list of skims you want to compute.

::

    from aequilibrae.paths.results import SkimResults as skmr
    from aequilibrae.paths import Graph
    from aequilibrae.paths import NetworkSkimming

    # We instantiate the graph and load it from disk (say you created it using the QGIS GUI
    g = Graph()
    g.load_from_disk(aeg_pth)

    # You now have to set the graph for what you want
    # In this case, we are computing fastest path (minimizing free flow time) and skimming **length** along the way
    # We are also **blocking** paths from going through centroids
    g.set_graph(cost_field='fftime', skim_fields=['length'],block_centroid_flows=True)

    # We instantiate the skim results and prepare it to have results compatible with the graph provided
    result = skmr()
    result.prepare(g)

    # We create the network skimming object and execute it
    # This is multi-threaded, so if the network is too big, prepare for a slow computer
    skm = NetworkSkimming(g, result)
    skm.execute()


If you want to use fewer cores for this computation (which also saves memory), you also can do it
You just need to use the method *set_cores* before you run the skimming. Ideally it is done before preparing it

::

    result = skmr()
    result.set_cores(3)
    result.prepare(g)

And if you want to compute skims between all nodes in the network, all you need to do is to make sure
the list of centroids in your graph is updated to include all nodes in the graph

::

    from aequilibrae.paths.results import SkimResults as skmr
    from aequilibrae.paths import Graph
    from aequilibrae.paths import NetworkSkimming

    g = Graph()
    g.load_from_disk(aeg_pth)

    # Let's keep the original list of centroids in case we want to use it again
    orig_centr = g.centroids

    # Now we set the list of centroids to include all nodes in the network
    g.prepare_graph(g.all_nodes)

    # And continue **almost** like we did before
    # We just need to remember to NOT block paths through centroids. Otherwise there will be no paths available
    g.set_graph(cost_field='fftime', skim_fields=['length'],block_centroid_flows=False)

    result = skmr()
    result.prepare(g)

    skm = NetworkSkimming(g, result)
    skm.execute()

After it is all said and done, the skim matrices are part of the result object.

You can save the results to your place of choice in AequilibraE format or export to OMX or CSV

::

    result.skims.export('path/to/desired/folder/file_name.omx')

    result.skims.export('path/to/desired/folder/file_name.csv')

    result.skims.copy('path/to/desired/folder/file_name.aem')


Traffic Assignment
~~~~~~~~~~~~~~~~~~

::

    some code

Gravity Models
--------------

::

    some code

Synthetic gravity calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    some code

Synthetic gravity application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    some code

Iterative Proportional Fitting (IPF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    some code

Transit
-------
We only have import for now, and it is likely to not work on Windows if you want the geometries

GTFS import
~~~~~~~~~~~

::

    some code

Matrix computation
------------------

::

    some code