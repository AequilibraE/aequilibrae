.. _numerical_study_traffic_assignment:

Traffic Assignment
==================

Similar to other complex algorthms that handle a large amount of data through
complex computations, traffic assignment procedures can always be subject to at
least one very reasonable question: Are the results right?

For this reason, we have used all equilibrium traffic assignment algorithms
available in AequilibraE to solve standard instances used in academia for
comparing algorithm results, some of which have are available with highly
converged solutions (~1e-14). Instances can be downloaded `here <https://github.com/bstabler/TransportationNetworks/>`_.

Sioux Falls
-----------
Network has:

* Links: 76
* Nodes: 24
* Zones: 24

.. image:: ../images/sioux_falls_msa-500_iter.png
    :align: center
    :width: 590
    :alt: Sioux Falls MSA 500 iterations
|
.. image:: ../images/sioux_falls_frank-wolfe-500_iter.png
    :align: center
    :width: 590
    :alt: Sioux Falls Frank-Wolfe 500 iterations
|
.. image:: ../images/sioux_falls_cfw-500_iter.png
    :align: center
    :width: 590
    :alt: Sioux Falls Conjugate Frank-Wolfe 500 iterations
|
.. image:: ../images/sioux_falls_bfw-500_iter.png
    :align: center
    :width: 590
    :alt: Sioux Falls Biconjugate Frank-Wolfe 500 iterations

Anaheim
-------
Network has:

* Links: 914
* Nodes: 416
* Zones: 38

.. image:: ../images/anaheim_msa-500_iter.png
    :align: center
    :width: 590
    :alt: Anaheim MSA 500 iterations
|
.. image:: ../images/anaheim_frank-wolfe-500_iter.png
    :align: center
    :width: 590
    :alt: Anaheim Frank-Wolfe 500 iterations
|
.. image:: ../images/anaheim_cfw-500_iter.png
    :align: center
    :width: 590
    :alt: Anaheim Conjugate Frank-Wolfe 500 iterations
|
.. image:: ../images/anaheim_bfw-500_iter.png
    :align: center
    :width: 590
    :alt: Anaheim Biconjugate Frank-Wolfe 500 iterations

Winnipeg
--------
Network has:

* Links: 914
* Nodes: 416
* Zones: 38

.. image:: ../images/winnipeg_msa-500_iter.png
    :align: center
    :width: 590
    :alt: Winnipeg MSA 500 iterations
|
.. image:: ../images/winnipeg_frank-wolfe-500_iter.png
    :align: center
    :width: 590
    :alt: Winnipeg Frank-Wolfe 500 iterations
|
.. image:: ../images/winnipeg_cfw-500_iter.png
    :align: center
    :width: 590
    :alt: Winnipeg Conjugate Frank-Wolfe 500 iterations
|
.. image:: ../images/winnipeg_bfw-500_iter.png
    :align: center
    :width: 590
    :alt: Winnipeg Biconjugate Frank-Wolfe 500 iterations

The results for Winnipeg do not seem extremely good when compared to a highly,
but we believe posting its results would suggest deeper investigation by one
of our users :-)


Barcelona
---------
Network has:

* Links: 2,522
* Nodes: 1,020
* Zones: 110

.. image:: ../images/barcelona_msa-500_iter.png
    :align: center
    :width: 590
    :alt: Barcelona MSA 500 iterations
|
.. image:: ../images/barcelona_frank-wolfe-500_iter.png
    :align: center
    :width: 590
    :alt: Barcelona Frank-Wolfe 500 iterations
|
.. image:: ../images/barcelona_cfw-500_iter.png
    :align: center
    :width: 590
    :alt: Barcelona Conjugate Frank-Wolfe 500 iterations
|
.. image:: ../images/barcelona_bfw-500_iter.png
    :align: center
    :width: 590
    :alt: Barcelona Biconjugate Frank-Wolfe 500 iterations

Chicago Regional
----------------
Network has:

* Links: 39,018
* Nodes: 12,982
* Zones: 1,790

.. image:: ../images/chicago_regional_msa-500_iter.png
    :align: center
    :width: 590
    :alt: Chicago MSA 500 iterations
|
.. image:: ../images/chicago_regional_frank-wolfe-500_iter.png
    :align: center
    :width: 590
    :alt: Chicago Frank-Wolfe 500 iterations
|
.. image:: ../images/chicago_regional_cfw-500_iter.png
    :align: center
    :width: 590
    :alt: Chicago Conjugate Frank-Wolfe 500 iterations
|
.. image:: ../images/chicago_regional_bfw-500_iter.png
    :align: center
    :width: 590
    :alt: Chicago Biconjugate Frank-Wolfe 500 iterations

Convergence Study
-----------------

Besides validating the final results from the algorithms, we have also compared
how well they converge for the largest instance we have tested (Chicago
Regional), as that instance has a comparable size to real-world models.

.. image:: ../images/convergence_comparison.png
    :align: center
    :width: 590
    :alt: Algorithm convergence comparison
|

Not surprinsingly, one can see that Frank-Wolfe far outperforms the Method of
Successive Averages for a number of iterations larger than 25, and is capable of
reaching 1.0e-04 just after 800 iterations, while MSA is still at 3.5e-4 even
after 1,000 iterations.

The actual show, however, is left for the Biconjugate Frank-Wolfe
implementation, which delivers a relative gap of under 1.0e-04 in under 200
iterations, and a relative gap of under 1.0e-05 in just over 700 iterations.

This convergence capability, allied to its computational performance described
below suggest that AequilibraE is ready to be used in large real-world
applications.

Computational performance
-------------------------
Running on a IdeaPad Gaming3i equipped with a 12 cores Intel Core i7-10750H
CPU @ 2.60 GHz, and 32GB of RAM, AequilibraE performed 1,000 iterations of 
Frank-Wolfe assignment on the Chicago Network in just under 18 minutes, 
while Biconjugate Frank Wolfe takes just under 19 minutes.

Compared with AequilibraE previous versions, we can notice a reasonable descrease
in processing time.

Noteworthy items
----------------

.. note::
   The biggest opportunity for performance in AequilibraE right now it to apply
   network contraction hierarchies to the building of the graph, but that is
   still a long-term goal

Want to run your own convergence study?
---------------------------------------

If you want to run the convergence study in your machine, with Chicago Regional instance
or any other instance presented here, check out the code block below! Please make sure
you have already imported `TNTP files <https://github.com/bstabler/TransportationNetworks>`_ 
into your machine.

In the first part of the code, we'll parse TNTP instances to a format AequilibraE can
understand, and then we'll perform the assignment.

.. _code-block-for-convergence-study:
.. code-block:: python

    # Imports
    import os
    import numpy as np
    import pandas as pd
    from aequilibrae.matrix import AequilibraeMatrix, AequilibraeData

    from aequilibrae.paths import TrafficAssignment
    from aequilibrae.paths.traffic_class import TrafficClass
    import statsmodels.api as sm

    from aequilibrae.paths import Graph
    from copy import deepcopy

    # Folders
    data_folder = 'C:/your/path/to/TransportationNetworks/chicago-regional'
    matfile = os.path.join(data_folder, 'ChicagoRegional_trips.tntp')

    # Creating the matrix
    f = open(matfile, 'r')
    all_rows = f.read()
    blocks = all_rows.split('Origin')[1:]
    matrix = {}
    for k in range(len(blocks)):
        orig = blocks[k].split('\n')
        dests = orig[1:]
        orig=int(orig[0])

        d = [eval('{'+a.replace(';',',').replace(' ','') +'}') for a in dests]
        destinations = {}
        for i in d:
            destinations = {**destinations, **i}
        matrix[orig] = destinations
    zones = max(matrix.keys())
    index = np.arange(zones) + 1
    mat = np.zeros((zones, zones))
    for i in range(zones):
        for j in range(zones):
            mat[i, j] = matrix[i+1].get(j+1,0)

    # Let's save our matrix in AequilibraE Matrix format
    aemfile = os.path.join(folder, "demand.aem")
    aem = AequilibraeMatrix()
    kwargs = {'file_name': aem_file,
            'zones': zones,
            'matrix_names': ['matrix'],
            "memory_only": False}  # in case you want to save the matrix in your machine

    aem.create_empty(**kwargs)
    aem.matrix['matrix'][:,:] = mtx[:,:]
    aem.index[:] = index[:]

    # Now let's parse the network
    net = os.path.join(data_folder, 'ChicagoRegional_net.tntp')
    net = pd.read_csv(net, skiprows=7, sep='\t')

    network = net[['init_node', 'term_node', 'free_flow_time', 'capacity', "b", "power"]]
    network.columns = ['a_node', 'b_node', 'free_flow_time', 'capacity', "b", "power"]
    network = network.assign(direction=1)
    network["link_id"] = network.index + 1

    # If you want to create an AequilibraE matrix for computation, then it follows
    g = Graph()
    g.cost = net['free_flow_time'].values
    g.capacity = net['capacity'].values
    g.free_flow_time = net['free_flow_time'].values

    g.network = network 
    g.network.loc[(g.network.power < 1), "power"] = 1
    g.network.loc[(g.network.free_flow_time == 0), "free_flow_time"] = 0.01
    g.network_ok = True
    g.status = 'OK'
    g.prepare_graph(index)
    g.set_graph("free_flow_time")
    g.set_skimming(["free_flow_time"])
    g.set_blocked_centroid_flows(True)

    # We run the traffic assignment
    for algorithm in ["bfw", "fw", "cfw", "msa"]:

        mat = AequilibraeMatrix()
        mat.load(os.path.join(data_folder, "demand.aem"))
        mat.computational_view(["matrix"])

        assigclass = TrafficClass("car", g, mat)

        assig = TrafficAssignment()

        assig.set_classes([assigclass])
        assig.set_vdf("BPR")
        assig.set_vdf_parameters({"alpha": "b", "beta": "power"})
        assig.set_capacity_field("capacity")
        assig.set_time_field("free_flow_time")
        assig.max_iter = 1000
        assig.rgap_target = 1e-10
        assig.set_algorithm(algorithm)

        assig.execute()
        assigclass.results.save_to_disk(
            os.path.join(data_folder, f"convergence_study/results-1000.aed"))

        assig.report().to_csv(os.path.join(data_folder, f"{algorithm}_computational_results.csv"))

As we've exported the assignment's results into CSV files, we can use Pandas to read the files,
and plot a graph just :ref:`like the one above <Algorithm convergence comparison>`.