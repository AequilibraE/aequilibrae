.. _create_aequilibrae_model:

========================
Create AequilibraE model
========================

.. code-block:: python
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
    project.load(join(fldr, prj_fldr))

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