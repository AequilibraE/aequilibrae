.. _numerical_study_traffic_assignment:

Traffic Assignment Validation
=============================

Similar to other complex algorithms that handle a large amount of data through
complex computations, traffic assignment procedures can always be subject to at
least one very reasonable question: Are the results right?

For this reason, we have used all equilibrium traffic assignment algorithms
available in AequilibraE to solve standard instances used in academia for
comparing algorithm results.

Instances can be downloaded `here <https://github.com/bstabler/TransportationNetworks/>`_.

All tests were performed with the AequilibraE version 1.1.0.

As shown below, the results produced by AequilibraE are within expected, although
some differences have been found, particularly for Winnipeg. We suspect that there are 
issues with the reference results and welcome further investigations.

.. tabs::

   .. tab:: Chicago

      .. tabs::

         .. tab:: Network stats

            * Links: 39,018
            * Nodes: 12,982
            * Zones: 1,790

         .. tab:: biconjugate Frank-Wolfe

            .. image:: ../_images/assig_validation/ChicagoRegional_bfw-1000_iter.png
                :align: center
                :width: 590
                :alt: Chicago Biconjugate Frank-Wolfe 1000 iterations

         .. tab:: Conjugate Frank-Wolfe

            .. image:: ../_images/assig_validation/ChicagoRegional_cfw-1000_iter.png
                :align: center
                :width: 590
                :alt: Chicago Conjugate Frank-Wolfe 1000 iterations

         .. tab:: Frank-Wolfe

            .. image:: ../_images/assig_validation/ChicagoRegional_fw-1000_iter.png
                :align: center
                :width: 590
                :alt: Chicago Frank-Wolfe 1000 iterations

         .. tab:: MSA

            .. image:: ../_images/assig_validation/ChicagoRegional_msa-1000_iter.png
                :align: center
                :width: 590
                :alt: Chicago MSA 1000 iterations

   .. tab:: Barcelona

      .. tabs::

         .. tab:: Network stats

            * Links: 2,522
            * Nodes: 1,020
            * Zones: 110

         .. tab:: biconjugate Frank-Wolfe

            .. image:: ../_images/assig_validation/Barcelona_bfw-1000_iter.png
                :align: center
                :width: 590
                :alt: Barcelona Biconjugate Frank-Wolfe 1000 iterations

         .. tab:: Conjugate Frank-Wolfe

            .. image:: ../_images/assig_validation/Barcelona_cfw-1000_iter.png
                :align: center
                :width: 590
                :alt: Barcelona Conjugate Frank-Wolfe 1000 iterations

         .. tab:: Frank-Wolfe

            .. image:: ../_images/assig_validation/Barcelona_fw-1000_iter.png
                :align: center
                :width: 590
                :alt: Barcelona Frank-Wolfe 1000 iterations

         .. tab:: MSA

            .. image:: ../_images/assig_validation/Barcelona_msa-1000_iter.png
                :align: center
                :width: 590
                :alt: Barcelona MSA 1000 iterations

   .. tab:: Winnipeg

      .. tabs::

         .. tab:: Network stats

            * Links: 914
            * Nodes: 416
            * Zones: 38

         .. tab:: biconjugate Frank-Wolfe

            .. image:: ../_images/assig_validation/Winnipeg_bfw-1000_iter.png
                :align: center
                :width: 590
                :alt: Winnipeg Biconjugate Frank-Wolfe 1000 iterations

         .. tab:: Conjugate Frank-Wolfe

            .. image:: ../_images/assig_validation/Winnipeg_cfw-1000_iter.png
                :align: center
                :width: 590
                :alt: Winnipeg Conjugate Frank-Wolfe 1000 iterations

         .. tab:: Frank-Wolfe

            .. image:: ../_images/assig_validation/Winnipeg_fw-1000_iter.png
                :align: center
                :width: 590
                :alt: Winnipeg Frank-Wolfe 1000 iterations

         .. tab:: MSA

            .. image:: ../_images/assig_validation/Winnipeg_msa-1000_iter.png
                :align: center
                :width: 590
                :alt: Winnipeg MSA 1000 iterations

   .. tab:: Anaheim

      .. tabs::

         .. tab:: Network stats

            * Links: 914
            * Nodes: 416
            * Zones: 38

         .. tab:: biconjugate Frank-Wolfe

            .. image:: ../_images/assig_validation/Anaheim_bfw-1000_iter.png
                :align: center
                :width: 590
                :alt: Anaheim Biconjugate Frank-Wolfe 1000 iterations

         .. tab:: Conjugate Frank-Wolfe

            .. image:: ../_images/assig_validation/Anaheim_cfw-1000_iter.png
                :align: center
                :width: 590
                :alt: Anaheim Conjugate Frank-Wolfe 1000 iterations

         .. tab:: Frank-Wolfe

            .. image:: ../_images/assig_validation/Anaheim_fw-1000_iter.png
                :align: center
                :width: 590
                :alt: Anaheim Frank-Wolfe 1000 iterations

         .. tab:: MSA

            .. image:: ../_images/assig_validation/Anaheim_msa-1000_iter.png
                :align: center
                :width: 590
                :alt: Anaheim MSA 1000 iterations

   .. tab:: Sioux Falls

      .. tabs::

         .. tab:: Network stats

            * Links: 76
            * Nodes: 24
            * Zones: 24

         .. tab:: biconjugate Frank-Wolfe

            .. image:: ../_images/assig_validation/SiouxFalls_bfw-1000_iter.png
                :align: center
                :width: 590
                :alt: Sioux Falls Biconjugate Frank-Wolfe 1000 iterations

         .. tab:: Conjugate Frank-Wolfe

            .. image:: ../_images/assig_validation/SiouxFalls_cfw-1000_iter.png
                :align: center
                :width: 590
                :alt: Sioux Falls Conjugate Frank-Wolfe 1000 iterations

         .. tab:: Frank-Wolfe

            .. image:: ../_images/assig_validation/SiouxFalls_fw-1000_iter.png
                :align: center
                :width: 590
                :alt: Sioux Falls Frank-Wolfe 1000 iterations

         .. tab:: MSA

            .. image:: ../_images/assig_validation/SiouxFalls_msa-1000_iter.png
                :align: center
                :width: 590
                :alt: Sioux Falls MSA 1000 iterations

Convergence Study
-----------------

Besides validating the final results from the algorithms, we have also compared
how well they converge for the largest instance we have tested (Chicago Regional), 
as that instance has a comparable size to real-world models.

.. _algorithm_convergence_comparison:

.. tabs::

   .. tab:: Chicago

      .. image:: ../_images/assig_validation/convergence_comparison_ChicagoRegional.png
          :align: center
          :width: 590
          :alt: Algorithm convergence comparison

   .. tab:: Barcelona

      .. image:: ../_images/assig_validation/convergence_comparison_Barcelona.png
          :align: center
          :width: 590
          :alt: Algorithm convergence comparison

   .. tab:: Winnipeg

      .. image:: ../_images/assig_validation/convergence_comparison_Winnipeg.png
          :align: center
          :width: 590
          :alt: Algorithm convergence comparison

   .. tab:: Anaheim

      .. image:: ../_images/assig_validation/convergence_comparison_Anaheim.png
          :align: center
          :width: 590
          :alt: Algorithm convergence comparison

   .. tab:: Sioux-Falls

      .. image:: ../_images/assig_validation/convergence_comparison_SiouxFalls.png
          :align: center
          :width: 590
          :alt: Algorithm convergence comparison

Not surprisingly, one can see that Frank-Wolfe far outperforms the Method of
Successive Averages for a number of iterations larger than 25 in the case of
Chicago, and is capable of reaching 1.0e-04 just after 800 iterations, while 
MSA is still at 3.5e-4 even after 1,000 iterations for that same case.

The actual show, however, is left for the biconjugate Frank-Wolfe
implementation, which delivers a relative gap of under 1.0e-04 in under 200
iterations, and a relative gap of under 1.0e-05 in just over 700 iterations.

This convergence capability, allied to its computational performance described
below suggest that AequilibraE is ready to be used in large real-world
applications.

Computational performance
-------------------------

All tests were run on a workstation equipped AMD Threadripper 3970X with 32 cores
(64 threads) @ 3.7 GHz (memory use is trivial for these instances).

On this machine, AequilibraE performed 1,000 iterations of
biconjugate Frank-Wolfe assignment on the Chicago Network in a little over 4 minutes,
or a little less than 0.43s per iteration.

Compared with AequilibraE previous versions, we can notice a reasonable decrease
in processing time.

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
    from pathlib import Path
    from time import perf_counter

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    from aequilibrae.matrix import AequilibraeMatrix
    from aequilibrae.paths import Graph
    from aequilibrae.paths import TrafficAssignment
    from aequilibrae.paths.traffic_class import TrafficClass

    # Helper functions
    def build_matrix(folder: Path, model_stub: str) -> AequilibraeMatrix:
        omx_name = folder / f"{model_stub}_trips.omx"
        if omx_name.exists():
            mat = AequilibraeMatrix()
            mat.load(omx_name)
            mat.computational_view()
            return mat

        matfile = str(folder / f"{model_stub}_trips.tntp")
        # Creating the matrix
        f = open(matfile, 'r')
        all_rows = f.read()
        blocks = all_rows.split('Origin')[1:]
        matrix = {}
        for k in range(len(blocks)):
            orig = blocks[k].split('\n')
            dests = orig[1:]
            orig = int(orig[0])

            d = [eval('{' + a.replace(';', ',').replace(' ', '') + '}') for a in dests]
            destinations = {}
            for i in d:
                destinations = {**destinations, **i}
            matrix[orig] = destinations
        zones = max(matrix.keys())
        index = np.arange(zones) + 1
        mat_data = np.zeros((zones, zones))
        for i in range(zones):
            for j in range(zones):
                mat_data[i, j] = matrix[i + 1].get(j + 1, 0)

        # Let's save our matrix in AequilibraE Matrix format
        mat = AequilibraeMatrix()
        mat.create_empty(zones=zones, matrix_names=['matrix'], memory_only=True)
        mat.matrix['matrix'][:, :] = mat_data[:, :]
        mat.index[:] = index[:]
        mat.computational_view(["matrix"])
        mat.export(str(omx_name))
        return mat

    # Now let's parse the network
    def build_graph(folder: Path, model_stub: str, centroids: np.array) -> Graph:
        net = pd.read_csv(folder / f"{model_stub}_net.tntp", skiprows=7, sep='\t')
        cols = ['init_node', 'term_node', 'free_flow_time', 'capacity', "b", "power"]
        if 'toll' in net.columns:
            cols.append('toll')
        network = net[cols]
        network.columns = ['a_node', 'b_node', 'free_flow_time', 'capacity', "b", "power", "toll"]
        network = network.assign(direction=1)
        network["link_id"] = network.index + 1
        network.free_flow_time = network.free_flow_time.astype(np.float64)

        # If you want to create an AequilibraE matrix for computation, then it follows
        g = Graph()
        g.cost = net['free_flow_time'].values
        g.capacity = net['capacity'].values
        g.free_flow_time = net['free_flow_time'].values

        g.network = network
        g.network.loc[(g.network.power < 1), "power"] = 1
        g.network.loc[(g.network.free_flow_time == 0), "free_flow_time"] = 0.01
        g.prepare_graph(centroids)
        g.set_graph("free_flow_time")
        g.set_skimming(["free_flow_time"])
        g.set_blocked_centroid_flows(False)
        return g

    def known_results(folder: Path, model_stub: str) -> pd.DataFrame:
        df = pd.read_csv(folder / f"{model_stub}_flow.tntp", sep='\t')
        df.columns = ["a_node", "b_node", "TNTP Solution", "cost"]
        return df

    # Let's run the assignment
    def assign(g: Graph, mat: AequilibraeMatrix, algorithm: str):
        assigclass = TrafficClass("car", g, mat)
        if "toll" in g.network.columns:
            assigclass.set_fixed_cost("toll")

        assig = TrafficAssignment()
        assig.set_classes([assigclass])
        assig.set_vdf("BPR")
        assig.set_vdf_parameters({"alpha": "b", "beta": "power"})
        assig.set_capacity_field("capacity")
        assig.set_time_field("free_flow_time")
        assig.max_iter = 1000
        assig.rgap_target = 1e-10 # Nearly guarantees that convergence won't be reached
        assig.set_algorithm(algorithm)
        assig.execute()
        return assig

    # We compare the results
    def validate(assig: TrafficAssignment, known_flows: pd.DataFrame, algorithm: str, folder: Path, model_name):
        modeled = g.network[["link_id", "a_node", "b_node"]].merge(assig.results().matrix_ab.reset_index(),
                                                                   on="link_id").rename(
            columns={"matrix_ab": "AequilibraE Solution"})
        merged = known_flows.merge(modeled, on=["a_node", "b_node"])

        # Scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=merged, x="TNTP Solution", y="AequilibraE Solution", s=30)

        # Linear regression
        X = merged["TNTP Solution"].values.reshape(-1, 1)
        y = merged["AequilibraE Solution"].values
        reg = LinearRegression(fit_intercept=False).fit(X, y)
        y_pred = reg.predict(X)
        r_squared = r2_score(y, y_pred)

        # Plot regression line
        plt.plot(merged["TNTP Solution"], y_pred, color='red', label='Linear regression')

        # Customize the plot
        plt.title(f'Comparison of Known and AequilibraE Solutions - Algorithm: {algorithm}', fontsize=16)
        plt.xlabel('Known Solution', fontsize=14)
        plt.ylabel('AequilibraE Solution (1,000 iterations)', fontsize=14)

        # Display the equation and R-squared on the plot
        equation_text = f'y = {reg.coef_[0]:.2f}x\nR-squared = {r_squared:.5f}'
        plt.text(x=merged["TNTP Solution"].max() * 0.5, y=merged["AequilibraE Solution"].max() * 0.85, s=equation_text,
                 fontsize=12)

        plt.legend()
        plt.savefig(folder / f"{model_name}_{algorithm}-1000_iter.png", dpi=300)
        plt.close()

    def assign_and_validate(g: Graph, mat: AequilibraeMatrix, folder: Path, model_stub: str):
        known_flows = known_results(folder, model_stub)
        # We run the traffic assignment
        conv = None
        for algorithm in ["bfw", "cfw", "fw", "msa"]:
            t = -perf_counter()
            assig = assign(g, mat, algorithm)
            t += perf_counter()
            print(f"{model_stub},{algorithm},{t:0.4f}")

            res = assig.report()[["iteration", "rgap"]].rename(columns={"rgap": algorithm})
            validate(assig, known_flows, algorithm, folder, model_stub)

            conv = res if conv is None else conv.merge(res, on="iteration")
        df = conv.replace(np.inf, 1).set_index("iteration")
        convergence_chart(df, data_folder, model_stub)
        df.to_csv(folder / f"{model_stub}_convergence.csv")

    def convergence_chart(df: pd.DataFrame, folder: Path, model_name):
        import matplotlib.pyplot as plt

        plt.cla()
        df = df.loc[15:, :]
        for column in df.columns:
            plt.plot(df.index, df[column], label=column)
        # Customize the plot
        plt.title('Convergence Comparison')
        plt.xlabel('Iterations')
        plt.ylabel('RGap')
        plt.yscale("log")
        plt.legend(title='Columns')
        plt.savefig(folder / f"convergence_comparison_{model_name}.png", dpi=300)

    models = {"chicago": [Path(r'D:\src\TransportationNetworks\chicago-regional'), "ChicagoRegional"],
          "sioux_falls": [Path(r'D:\src\TransportationNetworks\SiouxFalls'), "SiouxFalls"],
            "anaheim": [Path(r'D:\src\TransportationNetworks\Anaheim'), "Anaheim"],
            "winnipeg": [Path(r'D:\src\TransportationNetworks\Winnipeg'), "Winnipeg"],
            "barcelona": [Path(r'D:\src\TransportationNetworks\Barcelona'), "Barcelona"],
          }

    convergence = {}
    for model_name, (data_folder, model_stub) in models.items():
        print(model_name)
        mat = build_matrix(data_folder, model_stub)
        g = build_graph(data_folder, model_stub, mat.index)
        assign_and_validate(g, mat, data_folder, model_stub)