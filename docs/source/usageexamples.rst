Use examples
============
This page is still under development, so if you have developed interesting use
cases, please consider contributing them.

.. note::
   The examples provided here are not meant as a through description of
   AequilibraE's capabilities. For that, please look into the API documentation
   or email aequilibrae@googlegroups.com

Sample Data
-----------

We have compiled two very distinct example datasets imported from the
`TNTP instances <https://github.com/bstabler/TransportationNetworks/>`_.

* `Sioux Falls <http://www.aequilibrae.com/data/SiouxFalls.7z>`_
* `Chicago Regional <http://www.aequilibrae.com/data/Chicago.7z>`_

While the Sioux Falls network is probably the most traditional example network
available for evaluating network algorithms, the Chicago Regional model is a
good example of a real-world sized model, with roughly 1,800 zones.

Each instance contains the following folder structure and contents:

0_tntp_data:

* Data imported from https://github.com/bstabler/TransportationNetworks/
* matrices in openmatrix and AequilibraE formats
* vectors computed from the matrix in question and in AequilibraE format
* No alterations made to the data

1_project

* AequilibraE project result of the import of the links and nodes layers

2_skim_results:

* Skim results for distance and free_flow_travel_time computed by minimizing
  free_flow_travel_time
* Result matrices in openmatrix and AequilibraE formats

3_desire_lines

* Layers for desire lines and delaunay lines,  each one in a separate
  geopackage file
* Desire lines flow map
* Delaunay Lines flow map

4_assignment_results

* Outputs from traffic assignment to a relative gap of 1e-5 and with skimming
  enabled
* Link flows in csv and AequilibraE formats
* Skim matrices in openmatrix and AequilibraE formats
* Assignment flow map in png format

5_distribution_results

* Models calibrated for inverse power and negative exponential deterrence
  functions
* Convergence logs for the calibration of each model
* Trip length frequency distribution chart for original matrix
* Trip length frequency distribution chart for model with negative exponential
  deterrence function
* Trip length frequency distribution chart for model with inverse power
  deterrence function
* Inputs are the original demand matrix and the skim for TIME (final iteration)
  from the ASSIGNMENT

6_forecast

* Synthetic future vectors generated with a random growth from 0 to 10% in each
  cell on top of the original matrix vectors
* Application of both gravity models calibrated plus IPF to the synthetic
  future vectors

7_future_year_assignment

* Traffic assignment

    - Outputs from traffic assignment to a relative gap of 1e-5 and with
      skimming enabled
    - Link flows in csv and AequilibraE formats
    - Skim matrices in openmatrix and AequilibraE formats
* Scenario comparison flow map of absolute differences
* Composite scenario comparison flow map (gray is flow maintained in both
  scenarios, red is flow growth and green is flow decline)


Comprehensive example
---------------------

The process of generating the data provided in the sample data above from the
data downloaded from the TNTP instances was similar than a natural workflow one
would find in a traditional model, and it was developed as a Jupyter notebook,
which is available on
`Github <https://github.com/AequilibraE/aequilibrae/blob/master/docs/source/SiouxFalls.ipynb>`_

Below we have that same workflow as a single script

::

    import sys
    from os.path import join
    import numpy as np
    from math import log10, floor
    import matplotlib.pyplot as plt
    from aequilibrae.distribution import GravityCalibration, Ipf, GravityApplication, SyntheticGravityModel
    from aequilibrae import Parameters
    from aequilibrae.project import Project
    from aequilibrae.paths import PathResults, SkimResults
    from aequilibrae.matrix import AequilibraeData, AequilibraeMatrix
    from aequilibrae import logger
    from aequilibrae.paths import TrafficAssignment, TrafficClass

    import logging

    ######### FILES AND FOLDER #########

    fldr = 'D:/release/Sample models/sioux_falls_2020_02_15'
    proj_name = 'SiouxFalls.sqlite'

    # remove the comments for the lines below to run the Chicago model example instead
    # fldr = 'D:/release/Sample models/Chicago_2020_02_15'
    # proj_name = 'chicagomodel.sqlite'

    dt_fldr = '0_tntp_data'
    prj_fldr = '1_project'
    skm_fldr = '2_skim_results'
    assg_fldr = '4_assignment_results'
    dstr_fldr = '5_distribution_results'
    frcst_fldr = '6_forecast'
    ftr_fldr = '7_future_year_assignment'

    ########### LOGGING #################

    p = Parameters()
    p.parameters['system']['logging_directory'] = fldr
    p.write_back()
    # To make sure the logging will go where it should, stop the script here and
    # re-run it

    # Because assignment takes a long time, we want the log to be shown here
    stdout_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s;%(name)s;%(levelname)s ; %(message)s")
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    ########### PROJECT #################

    project = Project()
    project.load(join(fldr, prj_fldr, proj_name))

    ########### PATH COMPUTATION #################

    # we build all graphs
    project.network.build_graphs()
    # We get warnings that several fields in the project are filled with NaNs.  Which is true, but we won't
    # use those fields

    # we grab the graph for cars
    graph = project.network.graphs['c']

    # let's say we want to minimize distance
    graph.set_graph('distance')

    # And will skim time and distance while we are at it
    graph.set_skimming(['free_flow_time', 'distance'])

    # And we will allow paths to be compute going through other centroids/centroid connectors
    # required for the Sioux Falls network, as all nodes are centroids
    graph.set_blocked_centroid_flows(False)

    # instantiate a path results object and prepare it to work with the graph
    res = PathResults()
    res.prepare(graph)

    # compute a path from node 2 to 13
    res.compute_path(2, 13)

    # We can get the sequence of nodes we traverse
    res.path_nodes

    # We can get the link sequence we traverse
    res.path

    # We can get the mileposts for our sequence of nodes
    res.milepost

    # And We can the skims for our tree
    res.skims

    # If we want to compute the path for a different destination and same origin, we can just do this
    # It is way faster when you have large networks
    res.update_trace(4)

    ########## SKIMMING ###################


    # setup the object result
    res = SkimResults()
    res.prepare(graph)

    # And run the skimming
    res.compute_skims()

    # The result is an AequilibraEMatrix object
    skims = res.skims

    # We can export to AEM and OMX
    skims.export(join(fldr, skm_fldr, 'skimming_on_time.aem'))
    skims.export(join(fldr, skm_fldr, 'skimming_on_time.omx'))

    ######### TRAFFIC ASSIGNMENT WITH SKIMMING

    demand = AequilibraeMatrix()
    demand.load(join(fldr, dt_fldr, 'demand.omx'))
    demand.computational_view(['matrix'])  # We will only assign one user class stored as 'matrix' inside the OMX file

    assig = TrafficAssignment()

    # Creates the assignment class
    assigclass = TrafficClass(graph, demand)

    # The first thing to do is to add at list of traffic classes to be assigned
    assig.set_classes([assigclass])

    assig.set_vdf("BPR")  # This is not case-sensitive # Then we set the volume delay function

    assig.set_vdf_parameters({"alpha": "b", "beta": "power"})  # And its parameters

    assig.set_capacity_field("capacity")  # The capacity and free flow travel times as they exist in the graph
    assig.set_time_field("free_flow_time")

    # And the algorithm we want to use to assign
    assig.set_algorithm('bfw')

    # since I haven't checked the parameters file, let's make sure convergence criteria is good
    assig.max_iter = 1000
    assig.rgap_target = 0.00001

    assig.execute()  # we then execute the assignment

    # Convergence report is easy to see
    import pandas as pd
    convergence_report = pd.DataFrame(assig.assignment.convergence_report)
    convergence_report.head()

    # The link flows are easy to export.
    # we do so for csv and AequilibraEData
    assigclass.results.save_to_disk(join(fldr, assg_fldr, 'link_flows_c.csv'), output="loads")
    assigclass.results.save_to_disk(join(fldr, assg_fldr, 'link_flows_c.aed'), output="loads")

    # the skims are easy to get.

    # The blended one are here
    avg_skims = assigclass.results.skims

    # The ones for the last iteration are here
    last_skims = assigclass._aon_results.skims

    # Assembling a single final skim file can be done like this
    # We will want only the time for the last iteration and the distance averaged out for all iterations
    kwargs = {'file_name': join(fldr, assg_fldr, 'skims.aem'),
              'zones': graph.num_zones,
              'matrix_names': ['time_final', 'distance_blended']}

    # Create the matrix file
    out_skims = AequilibraeMatrix()
    out_skims.create_empty(**kwargs)
    out_skims.index[:] = avg_skims.index[:]

    # Transfer the data
    # The names of the skims are the name of the fields
    out_skims.matrix['time_final'][:, :] = last_skims.matrix['free_flow_time'][:, :]
    # It is CRITICAL to assign the matrix values using the [:,:]
    out_skims.matrix['distance_blended'][:, :] = avg_skims.matrix['distance'][:, :]

    out_skims.matrices.flush()  # Make sure that all data went to the disk

    # Export to OMX as well
    out_skims.export(join(fldr, assg_fldr, 'skims.omx'))

    #############    TRIP DISTRIBUTION #################

    # The demand is already in memory

    # Need the skims
    imped = AequilibraeMatrix()
    imped.load(join(fldr, assg_fldr, 'skims.aem'))

    # But before using the data, let's get some impedance for the intrazonals
    # Let's assume it is 75% of the closest zone

    # If we run the code below more than once, we will be overwriting the diagonal values with non-sensical data
    # so let's zero it first
    np.fill_diagonal(imped.matrix['time_final'], 0)

    # We compute it with a little bit of NumPy magic
    intrazonals = np.amin(imped.matrix['time_final'], where=imped.matrix['time_final'] > 0,
                          initial=imped.matrix['time_final'].max(), axis=1)
    intrazonals *= 0.75

    # Then we fill in the impedance matrix
    np.fill_diagonal(imped.matrix['time_final'], intrazonals)

    # We set the matrices for use in computation
    imped.computational_view(['time_final'])
    demand.computational_view(['matrix'])


    # Little function to plot TLFDs
    def plot_tlfd(demand, skim, name):
        # No science here. Just found it works well for Sioux Falls & Chicago
        b = floor(log10(skim.shape[0]) * 10)
        n, bins, patches = plt.hist(np.nan_to_num(skim.flatten(), 0), bins=b,
                                    weights=np.nan_to_num(demand.flatten()),
                                    density=False, facecolor='g', alpha=0.75)

        plt.xlabel('Trip length')
        plt.ylabel('Probability')
        plt.title('Trip-length frequency distribution')
        plt.savefig(name, format="png")
        plt.clf()


    # Calibrate models with the two functional forms
    for function in ['power', 'expo']:
        model = GravityCalibration(matrix=demand, impedance=imped, function=function, nan_as_zero=True)
        model.calibrate()

        # we save the model
        model.model.save(join(fldr, dstr_fldr, f'{function}_model.mod'))

        # We save a trip length frequency distribution image
        plot_tlfd(model.result_matrix.matrix_view, imped.matrix_view,
                  join(fldr, dstr_fldr, f'{function}_tfld.png'))

        # We can save the result of applying the model as well
        # we can also save the calibration report
        with open(join(fldr, dstr_fldr, f'{function}_convergence.log'), 'w') as otp:
            for r in model.report:
                otp.write(r + '\n')

    # We save a trip length frequency distribution image
    plot_tlfd(demand.matrix_view, imped.matrix_view, join(fldr, dstr_fldr, 'demand_tfld.png'))

    ################  FORECAST #############################

    # We compute the vectors from our matrix
    mat = AequilibraeMatrix()

    mat.load(join(fldr, dt_fldr, 'demand.omx'))
    mat.computational_view()
    origins = np.sum(mat.matrix_view, axis=1)
    destinations = np.sum(mat.matrix_view, axis=0)

    args = {'file_path':join(fldr,  frcst_fldr, 'synthetic_future_vector.aed'),
            "entries": mat.zones,
            "field_names": ["origins", "destinations"],
        "data_types": [np.float64, np.float64],
            "memory_mode": False}

    vectors = AequilibraeData()
    vectors.create_empty(**args)

    vectors.index[:] =mat.index[:]

    # Then grow them with some random growth between 0 and 10% - Plus balance them
    vectors.origins[:] = origins * (1+ np.random.rand(vectors.entries)/10)
    vectors.destinations[:] = destinations * (1+ np.random.rand(vectors.entries)/10)
    vectors.destinations *= vectors.origins.sum()/vectors.destinations.sum()

    # Impedance matrix is already in memory

    # We want the main diagonal to be zero, as the original matrix does
    # not have intrazonal trips
    np.fill_diagonal(imped.matrix_view, np.nan)

    # Apply the gravity models
    for function in ['power', 'expo']:
        model = SyntheticGravityModel()
        model.load(join(fldr, dstr_fldr, f'{function}_model.mod'))

        outmatrix = join(fldr,frcst_fldr, f'demand_{function}_model.aem')
        apply = GravityApplication()
        args = {"impedance": imped,
                "rows": vectors,
                "row_field": "origins",
                "model": model,
                "columns": vectors,
                "column_field": "destinations",
                "output": outmatrix,
                "nan_as_zero":True
                }

        gravity = GravityApplication(**args)
        gravity.apply()

        #We get the output matrix and save it to OMX too
        resm = AequilibraeMatrix()
        resm.load(outmatrix)
        resm.export(join(fldr,frcst_fldr, f'demand_{function}_model.omx'))

    # APPLY IPF
    demand = AequilibraeMatrix()
    demand.load(join(fldr, dt_fldr, 'demand.omx'))
    demand.computational_view()

    args = {'matrix': demand,
            'rows': vectors,
            'columns': vectors,
            'column_field': "destinations",
            'row_field': "origins",
            'nan_as_zero': True}

    ipf = Ipf(**args)
    ipf.fit()

    output = AequilibraeMatrix()
    output.load(ipf.output.file_path)

    output.export(join(fldr,frcst_fldr, 'demand_ipf.aem'))
    output.export(join(fldr,frcst_fldr, 'demand_ipf.omx'))


    logger.info('\n\n\n TRAFFIC ASSIGNMENT FOR FUTURE YEAR')

    # Let's use the IPF matrix
    demand = AequilibraeMatrix()
    demand.load(join(fldr, frcst_fldr, 'demand_ipf.omx'))
    demand.computational_view() # There is only one matrix there, so don;t even worry about its core name

    assig = TrafficAssignment()

    # Creates the assignment class
    assigclass = TrafficClass(graph, demand)

    # The first thing to do is to add at list of traffic classes to be assigned
    assig.set_classes([assigclass])

    assig.set_vdf("BPR")  # This is not case-sensitive # Then we set the volume delay function

    assig.set_vdf_parameters({"alpha": "b", "beta": "power"}) # And its parameters

    assig.set_capacity_field("capacity") # The capacity and free flow travel times as they exist in the graph
    assig.set_time_field("free_flow_time")

    # And the algorithm we want to use to assign
    assig.set_algorithm('bfw')

    # since I haven't checked the parameters file, let's make sure convergence criteria is good
    assig.max_iter = 1000
    assig.rgap_target = 0.00001

    assig.execute() # we then execute the assignment


.. _example_logging:

Logging
-------
AequilibraE uses Python's standard logging library to a file called
*aequilibrae.log*, but the output folder for this logging can be changed to a
custom system folder by altering the parameter **system --> logging_directory** on
the parameters file.

As an example, one could do programatically change the output folder to
*'D:/myProject/logs'* by doing the following:

::

  from aequilibrae import Parameters

  fldr = 'D:/myProject/logs'

  p = Parameters()
  p.parameters['system']['logging_directory'] =  fldr
  p.write_back()

The other useful resource, especially during model debugging it to also show
all log messages directly on the screen. Doing that requires a little knowledge
of the Python Logging library, but it is just as easy:

::

  from aequilibrae import logger
  import logging

  stdout_handler = logging.StreamHandler(sys.stdout)
  logger.addHandler(stdout_handler)

.. _example_usage_parameters:

Parameters module
-----------------
Several examples on how to manipulate the parameters within AequilibraE have
been shown in other parts of this tutorial.

However, in case you ever have trouble with parameter changes you have made,
you can always revert them to their default values. But remember, **ALL**
**CHANGES WILL BE LOST**.

::

  from aequilibrae import Parameters

  fldr = 'D:/myProject/logs'

  p = Parameters()
  p.reset_default()


.. _example_usage_matrix:

Matrix module
-------------

Let's see two cases where we work with the matrix module

Extracting vectors
~~~~~~~~~~~~~~~~~~

Let's extract the vectors for total origins and destinations for the Chicago
model demand matrix:

::

    from aequilibrae.matrix import AequilibraeData, AequilibraeMatrix
    import numpy as np

    mat = AequilibraeMatrix()
    mat.load("D:/release/Sample models/Chicago_2020_02_15/demand.omx")
    m = mat.get_matrix("matrix")

    vectors = "D:/release/Sample models/Chicago_2020_02_15/vectors.aed"
    args = {
        "file_path": vectors,
        "entries": vec_1.shape[0],
        "field_names": ["origins", "destinations"],
        "data_types": [np.float64, np.float64],
    }
    dataset = AequilibraeData()
    dataset.create_empty(**args)

    # Transfer the data
    dataset.index[:] =mat.index[:]
    dataset.origins[:] = np.sum(m, axis=1)[:]
    dataset.destinations[:] = np.sum(m, axis=0)[:]

Comprehensive example
~~~~~~~~~~~~~~~~~~~~~

Lets say we want to Import the freight matrices provided with FAF into AequilibraE's matrix format
in order to create some Delaunay Lines in QGIS or to perform traffic assignment

Required data
+++++++++++++

* `FAF Matrices <https://faf.ornl.gov/fafweb/Data/FAF4.4_HiLoForecasts.zip>`__
* `Zones System <http://www.census.gov/econ/cfs/AboutGeographyFiles/CFS_AREA_shapefile_010215.zip>`__

Useful Information
++++++++++++++++++

* `FAF overview <https://faf.ornl.gov/fafweb/>`__
* `FAF User Guide <https://faf.ornl.gov/fafweb/data/FAF4%20User%20Guide.pdf>`__
* `The blog post (with data) <http://www.xl-optim.com/matrix-api-and-multi-class-assignment>`__

The code
++++++++

We import all libraries we will need, including the AequilibraE

::

    import pandas as pd
    import numpy as np
    import os
    from aequilibrae.matrix import AequilibraeMatrix
    from scipy.sparse import coo_matrix

Now we set all the paths for files and parameters we need and import the matrices into a Pandas DataFrame

::

    data_folder = 'Y:/ALL DATA/DATA/Pedro/Professional/Data/USA/FAF/4.4'
    data_file = 'FAF4.4_HiLoForecasts.csv'
    sctg_names_file = 'sctg_codes.csv'  # Simplified to 50 characters, which is AequilibraE's limit
    output_folder = data_folder

    matrices = pd.read_csv(os.path.join(data_folder, data_file), low_memory=False)

We import the sctg codes

::

    sctg_names = pd.read_csv(os.path.join(data_folder, sctg_names_file), low_memory=False)
    sctg_names.set_index('Code', inplace=True)
    sctg_descr = list(sctg_names['Commodity Description'])


We now process the matrices to collect all the data we need, such as:

* List of zones
* CSTG codes
* Matrices/scenarios we are importing

::

    all_zones = np.array(sorted(list(set( list(matrices.dms_orig.unique()) + list(matrices.dms_dest.unique())))))

    # Count them and create a 0-based index
    num_zones = all_zones.shape[0]
    idx = np.arange(num_zones)

    # Creates the indexing dataframes
    origs = pd.DataFrame({"from_index": all_zones, "from":idx})
    dests = pd.DataFrame({"to_index": all_zones, "to":idx})

    # adds the new index columns to the pandas dataframe
    matrices = matrices.merge(origs, left_on='dms_orig', right_on='from_index', how='left')
    matrices = matrices.merge(dests, left_on='dms_dest', right_on='to_index', how='left')

    # Lists sctg codes and all the years/scenarios we have matrices for
    mat_years = [x for x in matrices.columns if 'tons' in x]
    sctg_codes = matrices.sctg2.unique()

We now import one matrix for each year, saving all the SCTG codes as different matrix cores in our zoning system

::

    # aggregate the matrix according to the relevant criteria
    agg_matrix = matrices.groupby(['from', 'to', 'sctg2'])[mat_years].sum()

    # returns the indices
    agg_matrix.reset_index(inplace=True)


    for y in mat_years:
        mat = AequilibraeMatrix()

        # Here it does not make sense to use OMX
        # If one wants to create an OMX from other data sources, openmatrix is
        # the library to use
        kwargs = {'file_name': os.path.join(output_folder, y + '.aem'),
                  'zones': num_zones,
                  'matrix_names': sctg_descr}

        mat.create_empty(**kwargs)
        mat.index[:] = all_zones[:]
        # for all sctg codes
        for i in sctg_names.index:
            prod_name = sctg_names['Commodity Description'][i]
            mat_filtered_sctg = agg_matrix[agg_matrix.sctg2 == i]

            m = coo_matrix((mat_filtered_sctg[y], (mat_filtered_sctg['from'], mat_filtered_sctg['to'])),
                                               shape=(num_zones, num_zones)).toarray().astype(np.float64)

            mat.matrix[prod_name][:,:] = m[:,:]

        mat.close()


.. _example_usage_project:

Project module
--------------

Let's suppose one wants to create project files for a list of 5 cities around
the world with their complete networks downloaded from
`Open Street Maps <http://www.openstreetmap.org>`_ and place them on a local
folder for analysis at a later time.


::

  from aequilibrae.project import Project

  cities = ["Darwin, Australia",
            "Karlsruhe, Germany",
            "London, UK",
            "Paris, France",
            "Auckland, New Zealand"]

  for city in cities:
      print(city)
      pth = f'd:/net_tests/{city}.sqlite'

      p = Project()
      p.new(pth)
      p.network.create_from_osm(place_name=city)
      p.conn.close()
      del p

If one wants to load a project and check some of its properties, it is easy:

::

  >>> from aequilibrae.project import Project

  >>> p = Project()
  >>> p.load('path/to_project')

  # for the modes available in the model
  >>> p.network.modes()
  ['car', 'walk', 'bicycle']

  >>> p.network.count_links()
  157926

  >>> p.network.count_nodes()
  793200


.. _example_usage_paths:

Paths module
------------

::

  from aequilibrae.paths import allOrNothing
  from aequilibrae.paths import path_computation
  from aequilibrae.paths.results import AssignmentResults as asgr
  from aequilibrae.paths.results import PathResults as pthr

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
    # In this case, we are computing fastest path (minimizing free flow time)
    g.set_graph(cost_field='fftime')

    # We are also **blocking** paths from going through centroids
    g.set_blocked_centroid_flows(block_centroid_flows=True)

    # We will be skimming for fftime **AND** distance along the way
    g.set_skimming(['fftime', 'distance'])

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
    g.set_graph(cost_field='fftime', block_centroid_flows=False)
    g.set_skimming('fftime')

    result = skmr()
    result.prepare(g)

    skm = NetworkSkimming(g, result)
    skm.execute()

Setting skimming after setting the graph is **CRITICAL**, and the skim matrices are part of the result object.

You can save the results to your place of choice in AequilibraE format or export to OMX or CSV

::

    result.skims.export('path/to/desired/folder/file_name.omx')

    result.skims.export('path/to/desired/folder/file_name.csv')

    result.skims.copy('path/to/desired/folder/file_name.aem')

.. _comprehensive_traffic_assignment_case:

Traffic assignment
~~~~~~~~~~~~~~~~~~

A comprehensive example of assignment

::

    from aequilibrae.project import Project
    from aequilibrae.paths import TrafficAssignment, TrafficClass
    from aequilibrae.matrix import AequilibraeMatrix

    assig = TrafficAssignment()

    proj = Project()
    proj.load('path/to/folder/SiouxFalls.sqlite')
    proj.network.build_graphs()
    # Mode c is car
    car_graph = proj.network.graphs['c']

    # If, for any reason, you would like to remove a set of links from the
    # graph based solely on the modes assigned to links in the project file
    # This will alter the Graph ID, but everything else (cost field, set of
    # centroids and configuration for blocking flows through centroid connectors
    #  remains unaltered
    car_graph.exclude_links([123, 451, 1, 569, 345])

    mat = AequilibraeMatrix()
    mat.load('path/to/folder/demand.omx')
    # We will only assign one user class stored as 'matrix' inside the OMX file
    mat.computational_view(['matrix'])

    # Creates the assignment class
    assigclass = TrafficClass(g, mat)

    # If you want to know which assignment algorithms are available:
    assig.algorithms_available()

    # If you want to know which Volume-Delay functions are available
    assig.vdf.functions_available()

    # The first thing to do is to add at list of traffic classes to be assigned
    assig.set_classes([assigclass])

    # Then we set the volume delay function
    assig.set_vdf("BPR")  # This is not case-sensitive

    # And its parameters
    assig.set_vdf_parameters({"alpha": "alpha", "beta": "beta"})

    # If you don't have parameters in the network, but rather global ones
    # assig.set_vdf_parameters({"alpha": 0.15, "beta": 4})

    # The capacity and free flow travel times as they exist in the graph
    assig.set_capacity_field("capacity")
    assig.set_time_field("free_flow_time")

    # And the algorithm we want to use to assign
    assig.set_algorithm('bfw')

    # To overwrite the number of iterations and the relative gap intended
    assig.max_iter = 250
    assig.rgap_target = 0.0001

    # To overwrite the number of CPU cores to be used
    assig.set_cores(3)

    # we then execute the assignment
    assig.execute()

Assigning traffic on TNTP instances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is a set of well known traffic assignment problems used in the literature
maintained on `GitHub <https://github.com/bstabler/TransportationNetworks/>`_
that is often used for tests, so we will use one of those problems here.

Let's suppose we want to perform traffic assignment for one of those problems
and check the results against the reference results.

The parsing and importing of those networks are not really the case here, but
there is `online code <https://gist.github.com/pedrocamargo/d565f545667fd473ea0590c7866965de>`_
available for doing that work.

::

    import os
    import sys
    import numpy as np
    import pandas as pd
    from aequilibrae.paths import TrafficAssignment
    from aequilibrae.paths import Graph
    from aequilibrae.paths.traffic_class import TrafficClass
    from aequilibrae.matrix import AequilibraeMatrix, AequilibraeData
    import matplotlib.pyplot as plt

    from aequilibrae import logger
    import logging

    # We redirect the logging output to the terminal
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stdout_handler)

    # Let's work with Sioux Falls
    os.chdir('D:/src/TransportationNetworks/SiouxFalls')
    result_file = 'SiouxFalls_flow.tntp'

    # Loads and prepares the graph
    g = Graph()
    g.load_from_disk('graph.aeg')
    g.set_graph('time')
    g.cost = np.array(g.cost, copy=True)
    g.set_skimming(['time'])
    g.set_blocked_centroid_flows(True)

    # Loads and prepares the matrix
    mat = AequilibraeMatrix()
    mat.load('demand.aem')
    mat.computational_view(['matrix'])

    # Creates the assignment class
    assigclass = TrafficClass(g, mat)

    # Instantiates the traffic assignment problem
    assig = TrafficAssignment()

    # configures it properly
    assig.set_vdf('BPR')
    assig.set_vdf_parameters(**{'alpha': 0.15, 'beta': 4.0})
    assig.set_capacity_field('capacity')
    assig.set_time_field('time')
    assig.set_classes(assigclass)
    # could be assig.set_algorithm('frank-wolfe')
    assig.set_algorithm('msa')

    # Execute the assignment
    assig.execute()

    # the results are within each traffic class only one, in this case
    assigclass.results.link_loads

.. _multiple_user_classes:

Setting multiple user classes before assignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's suppose one wants to setup a matrix for assignment that has two user
classes, *red_cars* and *blue cars* for a single traffic class. To do that, one
needs only to call the *computational_view* method with a list of the two
matrices of interest.  Both matrices need to be contained in the same file (and
to be contiguous if an *.aem instead of a *.omx file) however.

::

    mat = AequilibraeMatrix()
    mat.load('demand.aem')
    mat.computational_view(['red_cars', 'blue_cars'])


Advanced usage: Building a Graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Let's suppose now that you are interested in creating links from a bespoke procedure. For
the purpose of this example, let's say you have a sparse matrix representing a graph as
an adjacency matrix

::

    from aequilibrae.paths import Graph
    from aequilibrae.project.network import Network
    from scipy.sparse import coo_matrix

    # original_adjacency_matrix is a sparse matrix where positive values are actual links
    # where the value of the cell is the distance in that link

    # We create the sparse matrix in proper sparse matrix format
    sparse_graph = coo_matrix(original_adjacency_matrix)

    # We create the structure to create the network
    all_types = [k._Graph__integer_type,
                 k._Graph__integer_type,
                 k._Graph__integer_type,
                 np.int8,
                 k._Graph__float_type,
                 k._Graph__float_type]

    # List of all required link fields for a network
    # Network.req_link_flds

    # List of all required node fields for a network
    # Network.req_node_flds

    # List of fields that are reserved for internal workings
    # Network.protected_fields

    dt = [(t, d) for t, d in zip(all_titles, all_types)]

    # Number of links
    num_links = sparse_graph.data.shape[0]

    my_graph = Graph()
    my_graph.network = np.zeros(links, dtype=dt)

    my_graph.network['link_id'] = np.arange(links) + 1
    my_graph.network['a_node'] = sparse_graph.row
    my_graph.network['b_node'] = sparse_graph.col
    my_graph.network["distance"] = sparse_graph.data

    # If the links are directed (from A to B), direction is 1. If bi-directional, use zeros
    my_graph.network['direction'] = np.ones(links)

    # Let's say that all nodes in the network are centroids
    list_of_centroids =  np.arange(max(sparse_graph.shape[0], sparse_graph.shape[0])+ 1)
    centroids_list = np.array(list_of_centroids)

    my_graph.type_loaded = 'NETWORK'
    my_graph.status = 'OK'
    my_graph.network_ok = True
    my_graph.prepare_graph(centroids_list)

This usage is really advanced, and very rarely not-necessary. Make sure to know what you are doing
before going down this route

.. _example_usage_distribution:

Trip distribution
-----------------

The support for trip distribution in AequilibraE is not very comprehensive,
mostly because of the loss of relevance that such type of model has suffered
in the last decade.

However, it is possible to calibrate and apply synthetic gravity models and
to perform Iterative Proportional Fitting (IPF) with really high performance,
which might be of use in many applications other than traditional distribution.


.. Synthetic gravity calibration
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. ::

..    some code

Synthetic gravity application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, imagine that you have your demographic information in an
sqlite database and that you have already computed your skim matrix.

It is also important to notice that it is crucial to have consistent data, such
as same set of zones (indices) in both the demographics and the impedance
matrix.

::

    import pandas as pd
    import sqlite3

    from aequilibrae.matrix import AequilibraeMatrix
    from aequilibrae.matrix import AequilibraeData

    from aequilibrae.distribution import SyntheticGravityModel
    from aequilibrae.distribution import GravityApplication


    # We define the model we will use
    model = SyntheticGravityModel()

    # Before adding a parameter to the model, you need to define the model functional form
    model.function = "GAMMA" # "EXPO" or "POWER"

    # Only the parameter(s) applicable to the chosen functional form will have any effect
    model.alpha = 0.1
    model.beta = 0.0001

    # Or you can load the model from a file
    model.load('path/to/model/file')

    # We load the impedance matrix
    matrix = AequilibraeMatrix()
    matrix.load('path/to/impedance_matrix.aem')
    matrix.computational_view(['distance'])

    # We create the vectors we will use
    conn = sqlite3.connect('path/to/demographics/database')
    query = "SELECT zone_id, population, employment FROM demographics;"
    df = pd.read_sql_query(query,conn)

    index = df.zone_id.values[:]
    zones = index.shape[0]

    # You create the vectors you would have
    df = df.assign(production=df.population * 3.0)
    df = df.assign(attraction=df.employment * 4.0)

    # We create the vector database
    args = {"entries": zones, "field_names": ["productions", "attractions"],
        "data_types": [np.float64, np.float64], "memory_mode": True}
    vectors = AequilibraeData()
    vectors.create_empty(**args)

    # Assign the data to the vector object
    vectors.productions[:] = df.production.values[:]
    vectors.attractions[:] = df.attraction.values[:]
    vectors.index[:] = zones[:]

    # Balance the vectors
    vectors.attractions[:] *= vectors.productions.sum() / vectors.attractions.sum()

    args = {"impedance": matrix,
            "rows": vectors,
            "row_field": "productions",
            "model": model,
            "columns": vectors,
            "column_field": "attractions",
            "output": 'path/to/output/matrix.aem',
            "nan_as_zero":True
            }

    gravity = GravityApplication(**args)
    gravity.apply()

Iterative Proportional Fitting (IPF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The implementation of IPF is fully vectorized and leverages all the speed of NumPy, but it does not include the
fancy multithreading implemented in path computation.

**Please note that the AequilibraE matrix used as input is OVERWRITTEN by the IPF**

::

    import pandas as pd
    from aequilibrae.distribution import Ipf
    from aequilibrae.matrix import AequilibraeMatrix
    from aequilibrae.matrix import AequilibraeData

    matrix = AequilibraeMatrix()

    # Here we can create from OMX or load from an AequilibraE matrix.
    matrix.create_from_omx(path/to/aequilibrae_matrix, path/to/omxfile)

    # The matrix will be operated one (see the note on overwriting), so it does
    # not make sense load an OMX matrix


    source_vectors = pd.read_csv(path/to/CSVs)
    zones = source_vectors.zone.shape[0]

    args = {"entries": zones, "field_names": ["productions", "attractions"],
            "data_types": [np.float64, np.float64], "memory_mode": True}

    vectors = AequilibraEData()
    vectors.create_empty(**args)

    vectors.productions[:] = source_vectors.productions[:]
    vectors.attractions[:] = source_vectors.attractions[:]

    # We assume that the indices would be sorted and that they would match the matrix indices
    vectors.index[:] = source_vectors.zones[:]

    args = {
            "matrix": matrix, "rows": vectors, "row_field": "productions", "columns": vectors,
            "column_field": "attractions", "nan_as_zero": False}

    fratar = Ipf(**args)
    fratar.fit()

    # We can get back to our OMX matrix in the end
    matrix.export(path/to_omx/output)

.. Transit
.. -------
We only have import for now, and it is likely to not work on Windows if you want the geometries

.. _example_usage_transit:

.. GTFS import
.. ~~~~~~~~~~~

.. ::

..    some code
