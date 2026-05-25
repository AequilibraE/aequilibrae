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

All tests were performed with the AequilibraE version 1.7.0.

As shown below, the results produced by AequilibraE are within expected, although
some differences have been found, particularly for Anaheim. We suspect that there are
issues with the reference results and welcome further investigations.

.. tab-set::

   .. tab-item:: Chicago

      .. tab-set::

         .. tab-item:: Network stats

            * Links: 39,018
            * Nodes: 12,982
            * Zones: 1,790

         .. tab-item:: biconjugate Frank-Wolfe

            .. image:: ../_images/assig_validation/ChicagoRegional_flow_dashboard_bfw.png
                :align: center
                :width: 590
                :alt: Chicago Biconjugate Frank-Wolfe 1000 iterations

         .. tab-item:: Conjugate Frank-Wolfe

            .. image:: ../_images/assig_validation/ChicagoRegional_flow_dashboard_cfw.png
                :align: center
                :width: 590
                :alt: Chicago Conjugate Frank-Wolfe 1000 iterations

         .. tab-item:: Frank-Wolfe

            .. image:: ../_images/assig_validation/ChicagoRegional_flow_dashboard_fw.png
                :align: center
                :width: 590
                :alt: Chicago Frank-Wolfe 1000 iterations

         .. tab-item:: MSA

            .. image:: ../_images/assig_validation/ChicagoRegional_flow_dashboard_msa.png
                :align: center
                :width: 590
                :alt: Chicago MSA 1000 iterations

   .. tab-item:: Barcelona

      .. tab-set::

         .. tab-item:: Network stats

            * Links: 2,522
            * Nodes: 1,020
            * Zones: 110

         .. tab-item:: biconjugate Frank-Wolfe

            .. image:: ../_images/assig_validation/Barcelona_convergence_bfw.png
                :align: center
                :width: 590
                :alt: Barcelona Biconjugate Frank-Wolfe 1000 iterations

         .. tab-item:: Conjugate Frank-Wolfe

            .. image:: ../_images/assig_validation/Barcelona_convergence_cfw.png
                :align: center
                :width: 590
                :alt: Barcelona Conjugate Frank-Wolfe 1000 iterations

         .. tab-item:: Frank-Wolfe

            .. image:: ../_images/assig_validation/Barcelona_convergence_fw.png
                :align: center
                :width: 590
                :alt: Barcelona Frank-Wolfe 1000 iterations

         .. tab-item:: MSA

            .. image:: ../_images/assig_validation/Barcelona_convergence_msa.png
                :align: center
                :width: 590
                :alt: Barcelona MSA 1000 iterations

   .. tab-item:: Winnipeg

      .. tab-set::

         .. tab-item:: Network stats

            * Links: 914
            * Nodes: 416
            * Zones: 38

         .. tab-item:: biconjugate Frank-Wolfe

            .. image:: ../_images/assig_validation/Winnipeg_flow_dashboard_bfw.png
                :align: center
                :width: 590
                :alt: Winnipeg Biconjugate Frank-Wolfe 1000 iterations

         .. tab-item:: Conjugate Frank-Wolfe

            .. image:: ../_images/assig_validation/Winnipeg_flow_dashboard_cfw.png
                :align: center
                :width: 590
                :alt: Winnipeg Conjugate Frank-Wolfe 1000 iterations

         .. tab-item:: Frank-Wolfe

            .. image:: ../_images/assig_validation/Winnipeg_flow_dashboard_fw.png
                :align: center
                :width: 590
                :alt: Winnipeg Frank-Wolfe 1000 iterations

         .. tab-item:: MSA

            .. image:: ../_images/assig_validation/Winnipeg_flow_dashboard_msa.png
                :align: center
                :width: 590
                :alt: Winnipeg MSA 1000 iterations

   .. tab-item:: Anaheim

      .. tab-set::

         .. tab-item:: Network stats

            * Links: 914
            * Nodes: 416
            * Zones: 38

         .. tab-item:: biconjugate Frank-Wolfe

            .. image:: ../_images/assig_validation/Anaheim_flow_dashboard_bfw.png
                :align: center
                :width: 590
                :alt: Anaheim Biconjugate Frank-Wolfe 1000 iterations

         .. tab-item:: Conjugate Frank-Wolfe

            .. image:: ../_images/assig_validation/Anaheim_flow_dashboard_cfw.png
                :align: center
                :width: 590
                :alt: Anaheim Conjugate Frank-Wolfe 1000 iterations

         .. tab-item:: Frank-Wolfe

            .. image:: ../_images/assig_validation/Anaheim_flow_dashboard_fw.png
                :align: center
                :width: 590
                :alt: Anaheim Frank-Wolfe 1000 iterations

         .. tab-item:: MSA

            .. image:: ../_images/assig_validation/Anaheim_flow_dashboard_msa.png
                :align: center
                :width: 590
                :alt: Anaheim MSA 1000 iterations

   .. tab-item:: Sioux Falls

      .. tab-set::

         .. tab-item:: Network stats

            * Links: 76
            * Nodes: 24
            * Zones: 24

         .. tab-item:: biconjugate Frank-Wolfe

            .. image:: ../_images/assig_validation/SiouxFalls_flow_dashboard_bfw.png
                :align: center
                :width: 590
                :alt: Sioux Falls Biconjugate Frank-Wolfe 1000 iterations

         .. tab-item:: Conjugate Frank-Wolfe

            .. image:: ../_images/assig_validation/SiouxFalls_flow_dashboard_cfw.png
                :align: center
                :width: 590
                :alt: Sioux Falls Conjugate Frank-Wolfe 1000 iterations

         .. tab-item:: Frank-Wolfe

            .. image:: ../_images/assig_validation/SiouxFalls_flow_dashboard_fw.png
                :align: center
                :width: 590
                :alt: Sioux Falls Frank-Wolfe 1000 iterations

         .. tab-item:: MSA

            .. image:: ../_images/assig_validation/SiouxFalls_flow_dashboard_msa.png
                :align: center
                :width: 590
                :alt: Sioux Falls MSA 1000 iterations

Convergence Study
-----------------

Besides validating the final results from the algorithms, we have also compared
how well they converge for the largest instance we have tested (Chicago Regional),
as that instance has a comparable size to real-world models.

.. _algorithm_convergence_comparison:

.. tab-set::
   .. tab-item:: Chicago

      .. image:: ../_images/assig_validation/ChicagoRegional_aeq_method_convergence_time.png
          :align: center
          :width: 590
          :alt: Algorithm convergence comparison

   .. tab-item:: Winnipeg

      .. image:: ../_images/assig_validation/Winnipeg_aeq_method_convergence_time.png
          :align: center
          :width: 590
          :alt: Algorithm convergence comparison

   .. tab-item:: Anaheim

      .. image:: ../_images/assig_validation/Anaheim_aeq_method_convergence_time.png
          :align: center
          :width: 590
          :alt: Algorithm convergence comparison

   .. tab-item:: Sioux-Falls

      .. image:: ../_images/assig_validation/SiouxFalls_aeq_method_convergence_time.png
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

All tests were run on a desktop equipped with an Intel Core Ultra 285K (**14 cores used only**)
running Windows 11 Pro.

On this machine, AequilibraE performed 1,000 iterations of
biconjugate Frank-Wolfe assignment on the Chicago Network in around 267 seconds,
or around 0.267s per iteration (other algorithms are as low as 0.230 seconds per iteration).

This performance is substantially better than seen on previous versions and is on par with
that of commercial software.

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
    from pathlib import Path
    from time import perf_counter

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy.stats import linregress
    from quetzal.model import stepmodel

    from aequilibrae.matrix import AequilibraeMatrix
    from aequilibrae.paths import Graph, TrafficAssignment
    from aequilibrae.paths.traffic_class import TrafficClass


    BASE = Path(r"C:\Users\jake\src\aequilibrae\TransportationNetworks")

    # Model list used for validation runs.
    # Sioux Falls is included and handled by a dedicated Quetzal builder branch
    # because FIRST_THRU_NODE=1 means zones are embedded in the road graph.
    MODELS = {
        "chicago": (BASE / "chicago-regional", "ChicagoRegional"),
        "anaheim": (BASE / "Anaheim", "Anaheim"),
        "winnipeg": (BASE / "Winnipeg", "Winnipeg"),
        "sioux_falls": (BASE / "SiouxFalls", "SiouxFalls"),
        "barcelona": (BASE / "Barcelona", "Barcelona"),
    }

    # Methods to test: msa, fw, cfw, bfw
    # Note: Quetzal does not support cfw, so it will be skipped for Quetzal runs.
    METHODS = ["msa", "fw", "cfw", "bfw"]
    MAX_ITER = 1000
    RGAP_TARGET = 1e-10
    NUM_CORES = 14

    VDF = {"bpr": "time * (1 + alpha * (flow / capacity) ** beta)"}
    SEGMENTS = ["car"]


    def parse_tntp_header(folder: Path, model_stub: str) -> dict:
        """Returns the integer values from a TNTP metadata header."""
        result = {}
        with open(folder / f"{model_stub}_net.tntp") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("<") and ">" in line:
                    key, _, val = line[1:].partition(">")
                    try:
                        result[key.strip()] = int(val.strip())
                    except ValueError:
                        pass
                elif line.startswith("~"):
                    break
        return result


    def known_results(folder: Path, model_stub: str) -> pd.DataFrame:
        path = folder / f"{model_stub}_flow.tntp"
        with open(path) as fh:
            first_line = fh.readline().strip()
        skiprows = 8 if first_line.startswith("<") else 0
        df = pd.read_csv(path, skiprows=skiprows, sep=r"\s+", engine="python")
        # Drop any trailing semicolon column
        df = df.loc[:, ~df.columns.str.strip().isin([";", ""])]
        df.columns = [c.strip() for c in df.columns]
        # Normalise column names to a_node / b_node / TNTP Solution / cost
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if cl in ("tail", "from"):
                col_map[c] = "a_node"
            elif cl in ("head", "to"):
                col_map[c] = "b_node"
            elif cl in ("volume",):
                col_map[c] = "TNTP Solution"
            elif cl in ("cost",):
                col_map[c] = "cost"
        df = df.rename(columns=col_map)
        return df[["a_node", "b_node", "TNTP Solution"]].dropna()


    def build_matrix(folder: Path, model_stub: str) -> AequilibraeMatrix:
        omx_name = folder / f"{model_stub}_trips.omx"
        if omx_name.exists():
            mat = AequilibraeMatrix()
            mat.load(omx_name)
            mat.computational_view()
            return mat

        matfile = str(folder / f"{model_stub}_trips.tntp")
        with open(matfile, "r") as fh:
            all_rows = fh.read()
        blocks = all_rows.split("Origin")[1:]
        matrix = {}
        for k in range(len(blocks)):
            orig = blocks[k].split("\n")
            dests = orig[1:]
            orig = int(orig[0])
            d = [eval("{" + a.replace(";", ",").replace(" ", "") + "}") for a in dests]
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

        mat = AequilibraeMatrix()
        mat.create_empty(zones=zones, matrix_names=["matrix"], memory_only=True)
        mat.matrix["matrix"][:, :] = mat_data[:, :]
        mat.index[:] = index[:]
        mat.computational_view(["matrix"])
        mat.export(str(omx_name))
        return mat


    def build_graph(folder: Path, model_stub: str, centroids: np.ndarray) -> Graph:
        header = parse_tntp_header(folder, model_stub)
        first_thru_node = header.get("FIRST THRU NODE", 2)

        net = pd.read_csv(folder / f"{model_stub}_net.tntp", skiprows=7, sep="\t")
        cols = [
            "init_node",
            "term_node",
            "free_flow_time",
            "capacity",
            "b",
            "power",
            "length",
        ]
        if "toll" in net.columns:
            cols.append("toll")
        network = net[cols].copy()
        new_cols = [
            "a_node",
            "b_node",
            "free_flow_time",
            "capacity",
            "b",
            "power",
            "length",
        ]
        if "toll" in net.columns:
            new_cols.append("toll")
        network.columns = new_cols
        network = network.assign(direction=1)
        network["link_id"] = network.index + 1
        network["free_flow_time"] = network["free_flow_time"].astype(np.float64)

        g = Graph()
        g.cost = net["free_flow_time"].values
        g.capacity = net["capacity"].values
        g.free_flow_time = net["free_flow_time"].values

        g.network = network
        g.network.loc[g.network["power"] < 1, "power"] = 1
        g.network.loc[g.network["free_flow_time"] == 0, "free_flow_time"] = 0.01
        g.prepare_graph(centroids)
        g.set_graph("free_flow_time")
        g.set_skimming(["free_flow_time"])
        g.set_blocked_centroid_flows(first_thru_node > 1)
        return g


    def _base_quetzal_links(network: pd.DataFrame) -> pd.DataFrame:
        """Build the common Quetzal link table from AequilibraE network links."""
        q = network[
            [
                "link_id",
                "a_node",
                "b_node",
                "free_flow_time",
                "capacity",
                "b",
                "power",
                "length",
            ]
        ].copy()
        q = q.rename(
            columns={
                "a_node": "a",
                "b_node": "b",
                "free_flow_time": "time",
                "b": "alpha",
                "power": "beta",
            }
        )
        q["segments"] = [{"car"} for _ in range(len(q))]
        q["vdf"] = "bpr"
        q = q.set_index("link_id")
        q.index.name = "index"
        return q


    def _build_quetzal_model_separated_zones(
        network: pd.DataFrame,
        mat: AequilibraeMatrix,
    ) -> stepmodel.StepModel:
        """
        Standard TNTP case: zone nodes are separate from road nodes.
        Build zone_to_road from centroid-touching links.
        """
        sm = stepmodel.StepModel()
        centroids = mat.index

        q = _base_quetzal_links(network)

        # Centroid connectors: any link touching a centroid node.
        connector_mask = q["a"].isin(centroids) | q["b"].isin(centroids)

        sm.road_links = q[~connector_mask].drop(columns=["segments"]).copy()
        sm.road_links["segments"] = [{"car"} for _ in range(len(sm.road_links))]

        zone_to_road = q[connector_mask].copy()
        zone_to_road["direction"] = "egress"
        zone_to_road.loc[zone_to_road["a"].isin(centroids), "direction"] = "access"
        sm.zone_to_road = zone_to_road

        centroid_set = set(centroids.tolist())
        all_nodes = set(q["a"].tolist()) | set(q["b"].tolist())
        road_node_ids = sorted(all_nodes - centroid_set)
        sm.road_nodes = pd.DataFrame(index=road_node_ids)
        sm.road_nodes.index.name = "node_id"

        sm.zones = pd.DataFrame(index=centroids)
        sm.zones.index.name = "index"
        _set_quetzal_volumes(sm, mat, centroids)
        return sm


    def build_quetzal_model(
        network: pd.DataFrame,
        first_thru_node: int,
        mat: AequilibraeMatrix,
    ) -> stepmodel.StepModel:
        """
        Build a Quetzal StepModel from a TNTP network DataFrame.

        Branches:
        - first_thru_node > 1: standard separated zone/road coding
        - first_thru_node <= 1: embedded-zone coding (Sioux Falls style)
        """
        if first_thru_node <= 1:
            return _build_quetzal_model_embedded_zones(network, mat)
        return _build_quetzal_model_separated_zones(network, mat)


    def _set_quetzal_volumes(
        sm: stepmodel.StepModel, mat: AequilibraeMatrix, zone_ids: np.ndarray
    ) -> None:
        """Assign OD volumes to the StepModel using provided zone IDs."""
        n = len(zone_ids)
        sm.volumes = pd.DataFrame(
            {
                "origin": np.repeat(zone_ids, n),
                "destination": np.tile(zone_ids, n),
                "car": mat.matrix_view.flatten(),
            }
        )
        sm.volumes = sm.volumes[sm.volumes["car"] != 0.0].reset_index(drop=True)


    def _build_quetzal_model_embedded_zones(
        network: pd.DataFrame,
        mat: AequilibraeMatrix,
    ) -> stepmodel.StepModel:
        """
        Sioux Falls style case (FIRST_THRU_NODE=1): zones are embedded in the road graph.

        We keep all original links as road_links and create synthetic zone nodes with
        zero-time connectors to/from their corresponding road node so Quetzal still has
        a clean zone/road separation.
        """
        sm = stepmodel.StepModel()
        centroids = mat.index.astype(int)

        q = _base_quetzal_links(network)
        sm.road_links = q.copy()

        all_nodes = sorted(set(q["a"].tolist()) | set(q["b"].tolist()))
        sm.road_nodes = pd.DataFrame(index=all_nodes)
        sm.road_nodes.index.name = "node_id"

        # Create synthetic zone IDs to avoid colliding with road node IDs.
        max_node = int(max(all_nodes)) if all_nodes else int(np.max(centroids))
        zone_ids = (centroids + max_node).astype(int)
        zone_map = dict(zip(centroids.tolist(), zone_ids.tolist()))

        sm.zones = pd.DataFrame(index=zone_ids)
        sm.zones.index.name = "index"

        # Build access/egress connectors (zone <-> corresponding road node).
        connectors = []
        for c in centroids:
            z = zone_map[int(c)]
            connectors.append(
                {
                    "a": z,
                    "b": int(c),
                    "time": 0.0,
                    "capacity": 1e12,
                    "alpha": 0.15,
                    "beta": 4.0,
                    "length": 0.0,
                    "vdf": "bpr",
                    "segments": {"car"},
                    "direction": "access",
                }
            )
            connectors.append(
                {
                    "a": int(c),
                    "b": z,
                    "time": 0.0,
                    "capacity": 1e12,
                    "alpha": 0.15,
                    "beta": 4.0,
                    "length": 0.0,
                    "vdf": "bpr",
                    "segments": {"car"},
                    "direction": "egress",
                }
            )

        sm.zone_to_road = pd.DataFrame(connectors)
        sm.zone_to_road.index.name = "index"

        _set_quetzal_volumes(sm, mat, zone_ids)
        return sm


    def assign_aeq(g: Graph, mat: AequilibraeMatrix, method: str) -> TrafficAssignment:
        assig_class = TrafficClass("car", g, mat)
        if "toll" in g.network.columns:
            assig_class.set_fixed_cost("toll")
            assig_class.set_vot(1.0)

        assig = TrafficAssignment()
        assig.set_classes([assig_class])
        assig.set_vdf("BPR")
        assig.set_vdf_parameters({"alpha": "b", "beta": "power"})
        assig.set_capacity_field("capacity")
        assig.set_time_field("free_flow_time")
        assig.max_iter = MAX_ITER
        assig.rgap_target = RGAP_TARGET
        assig.set_algorithm(method)
        assig.set_cores(NUM_CORES)
        assig.execute()
        return assig


    def assign_quetzal(sm: stepmodel.StepModel, method: str) -> stepmodel.StepModel:
        # Quetzal's tolerance is a percentage; scale accordingly
        sm.step_road_pathfinder(
            method=method,
            maxiters=MAX_ITER,
            tolerance=RGAP_TARGET * 100.0,
            segments=SEGMENTS,
            vdf=VDF,
            turn_penalties=None,
            num_cores=NUM_CORES,
            return_car_los=False,
            log=True,
        )
        return sm


    def plot_convergence(
        qtl_convergence: pd.DataFrame | None,
        aeq_report: pd.DataFrame,
        name: str,
        method: str,
        tolerance: float,
        plot_time: bool = True,
        save_path: Path | None = None,
    ):
        """
        Convergence plot showing Quetzal and AequilibraE relative gaps.

        qtl_convergence is a DataFrame with columns 'relgap' (as percentage) and 'time'.
        Quetzal stores relgap as a percentage; divide by 100 to match AequilibraE's scale.
        rgap_direction was removed from AequilibraE; only the AoN rgap is plotted.
        """
        sns.set_theme(style="whitegrid", context="paper")
        palette = sns.color_palette()

        aeq_len = len(aeq_report)
        qtl_len = len(qtl_convergence) if qtl_convergence is not None else 0
        n_iters = max(qtl_len, aeq_len)
        markevery = max(1, n_iters // 20)

        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

        if plot_time:
            x_label = "Time (s)"
            x_aeq = aeq_report["time"]
        else:
            x_label = "Iterations"
            x_aeq = aeq_report.index

        if qtl_convergence is not None:
            x_sm = qtl_convergence["time"] if plot_time else range(len(qtl_convergence))
            sns.lineplot(
                x=x_sm,
                y=qtl_convergence["relgap"] / 100.0,
                label="Quetzal (relative gap)",
                ax=ax,
                marker="X",
                markevery=markevery,
                markersize=6,
                color=palette[0],
                linewidth=2,
            )

        sns.lineplot(
            x=x_aeq,
            y=aeq_report["rgap"],
            label="AequilibraE (AoN relative gap)",
            ax=ax,
            marker="^",
            markevery=markevery,
            markersize=7,
            color=palette[1],
            linewidth=2,
            linestyle=":",
            dash_capstyle="round",
        )

        ax.set_xlim(left=0)
        ax.set_yscale("log")
        ax.grid(True, which="minor", axis="y", linewidth=0.7, alpha=0.3)
        ax.set_xlabel(x_label, labelpad=10)
        ax.set_ylabel("Relative Gap", labelpad=10)
        ax.set_title(
            f"Convergence - {name}\n{method.upper()}, Target rgap: {tolerance}",
            pad=14,
            fontweight="bold",
            fontsize=14,
        )
        ax.text(
            0.02,
            0.03,
            f"Markers every {markevery} iterations",
            transform=ax.transAxes,
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "alpha": 0.9,
                "edgecolor": "0.9",
            },
        )
        ax.legend(frameon=True, framealpha=0.9, edgecolor="0.8")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        sns.despine(left=False, bottom=False)
        plt.tight_layout()
        plt.draw()

        if save_path:
            plt.savefig(save_path, dpi=plt.gcf().dpi)
        plt.show()


    def plot_aeq_method_convergence_times(
        aeq_method_reports: dict[str, pd.DataFrame],
        model_name: str,
        rgap_target: float,
        save_path: Path | None = None,
    ):
        """
        AequilibraE-only convergence plot over time for all tested methods.
        Styled to match the per-method convergence plots.
        """
        sns.set_theme(style="whitegrid", context="paper")
        palette = sns.color_palette()

        method_order = [m.upper() for m in METHODS if m.upper() in aeq_method_reports]
        if not method_order:
            method_order = list(aeq_method_reports.keys())

        max_len = max(len(df) for df in aeq_method_reports.values() if not df.empty)
        markevery = max(1, max_len // 20)

        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

        for i, method in enumerate(method_order):
            report = aeq_method_reports[method]
            sns.lineplot(
                x=report["time"],
                y=report["rgap"],
                label=method,
                ax=ax,
                marker="^",
                markevery=markevery,
                markersize=6,
                color=palette[i % len(palette)],
                linewidth=2,
                linestyle="-",
                # dash_capstyle="round",
            )

        ax.set_xlim(left=0)
        ax.set_yscale("log")
        ax.grid(True, which="minor", axis="y", linewidth=0.7, alpha=0.3)
        ax.set_xlabel("Time (s)", labelpad=10)
        ax.set_ylabel("Relative Gap", labelpad=10)
        ax.set_title(
            f"Convergence - {model_name}\nAequilibraE Methods, Target rgap: {rgap_target}",
            pad=14,
            fontweight="bold",
            fontsize=14,
        )
        ax.text(
            0.02,
            0.03,
            f"Markers every {markevery} iterations",
            transform=ax.transAxes,
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "alpha": 0.9,
                "edgecolor": "0.9",
            },
        )
        ax.legend(frameon=True, framealpha=0.9, edgecolor="0.8")
        sns.despine(left=False, bottom=False)
        plt.tight_layout()
        plt.draw()

        if save_path:
            plt.savefig(save_path, dpi=plt.gcf().dpi, bbox_inches="tight")

        plt.show()


    def plot_flow_dashboard(
        aeq_with_nodes: pd.DataFrame,
        qtl_with_nodes: pd.DataFrame | None,
        qtl_road_flows: pd.Series | None,
        aeq_road_flows: pd.Series | None,
        model_name: str,
        method: str,
        rgap_target: float,
        save_path: Path | None = None,
    ):
        """
        Combined flow comparison dashboard:
        - Main panel: AequilibraE vs TNTP (largest, left)
        - Side panels: Quetzal vs TNTP (top-right) and AequilibraE vs Quetzal (bottom-right)
          (side panels omitted when Quetzal data is not available)
        """
        sns.set_theme(style="whitegrid", context="paper")

        has_quetzal = qtl_with_nodes is not None

        fig = plt.figure(figsize=(16, 8), dpi=150)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], wspace=0.10, hspace=0.35)
        ax_main = fig.add_subplot(gs[:, 0])
        ax_rt = fig.add_subplot(gs[0, 1])
        ax_rb = fig.add_subplot(gs[1, 1])

        def add_scatter(ax, x_flows, y_flows, x_label, y_label, title, marker_size, show_grid_lines):
            ax.scatter(x_flows, y_flows, alpha=0.5, s=marker_size, label="Link flows")

            x_max = float(np.max(x_flows)) * 1.02 if len(x_flows) else 1.0
            y_max = float(np.max(y_flows)) * 1.02 if len(y_flows) else 1.0
            limit = max(x_max, y_max, 1.0)

            # power = np.floor(np.log10(limit))

            # print(0, limit, power, np.ceil(limit / 10 ** power) * 10 ** power, n_ticks)
            # major_ticks = np.linspace(0, np.ceil(limit / 10 ** power) * 10 ** power, n_ticks)

            reg = linregress(x_flows, y_flows)
            x_line = np.array([0.0, limit])
            y_line = reg.intercept + reg.slope * x_line
            ax.plot(
                x_line,
                y_line,
                linestyle="--",
                linewidth=1.8,
                color="red",
                label=f"Regression  R²={reg.rvalue**2:.4f}\ny = {reg.slope:.4f}x + {reg.intercept:.4f}",
            )
            if len(x_flows) > 0 and len(y_flows) > 0:
                ax.plot(
                    [0.0, limit],
                    [0.0, limit],
                    linestyle="-",
                    color="grey",
                    alpha=0.5,
                    label="1:1",
                )

            ax.set_xlim(0.0, limit)
            ax.set_ylim(0.0, limit)
            ax.set_aspect("equal", adjustable="box")
            steps = [4, 6, 8]
            ax.xaxis.set_major_locator(MaxNLocator(steps=steps))
            ax.yaxis.set_major_locator(MaxNLocator(steps=steps))
            ax.set_anchor("W")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title, fontweight="bold", fontsize=11)
            ax.legend(
                frameon=True, framealpha=0.9, edgecolor="0.8", loc="upper left", fontsize=8
            )
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color("black")

        add_scatter(
            ax_main,
            aeq_with_nodes["TNTP Solution"],
            aeq_with_nodes["PCE_AB"],
            "TNTP Reference Flow",
            "AequilibraE Flow",
            "AequilibraE vs TNTP",
            marker_size=16,
            show_grid_lines=True
        )

        add_scatter(
            ax_rt,
            qtl_with_nodes["TNTP Solution"]
            if qtl_with_nodes is not None
            else pd.Series(dtype=int),
            qtl_with_nodes["flow"]
            if qtl_with_nodes is not None
            else pd.Series(dtype=int),
            "TNTP Reference Flow",
            "Quetzal Flow",
            "Quetzal vs TNTP",
            marker_size=10,
            show_grid_lines=False
        )

        aligned_aeq = (
            aeq_road_flows.loc[qtl_road_flows.index]
            if qtl_road_flows is not None
            else pd.Series(dtype=int)
        )
        add_scatter(
            ax_rb,
            qtl_road_flows if qtl_road_flows is not None else pd.Series(dtype=int),
            aligned_aeq if aligned_aeq is not None else pd.Series(dtype=int),
            "Quetzal Flow",
            "AequilibraE Flow",
            "AequilibraE vs Quetzal",
            marker_size=10,
            show_grid_lines=False
        )

        fig.suptitle(
            f"Flow Validation Dashboard - {model_name}\n{method.upper()}",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        if not has_quetzal:
            for ax in [ax_rt, ax_rb]:
                ax.text(
                    0.5,
                    0.5,
                    "Quetzal data not available",
                    transform=ax.transAxes,
                    fontsize=12,
                    ha="center",
                    va="center",
                    color="grey",
                )
                ax.legend().set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])

        # plt.tight_layout()
        # plt.draw()

        if save_path:
            plt.savefig(save_path, dpi=plt.gcf().dpi, bbox_inches="tight")

        plt.show()


    def run_assignments() -> None:
        """Part 1: Run AequilibraE and Quetzal assignments and save all results to disk."""
        for model_name, (data_folder, model_stub) in MODELS.items():
            print(f"\n{'=' * 60}")
            print(f"MODEL: {model_name.upper()}")
            print(f"{'=' * 60}")

            header = parse_tntp_header(data_folder, model_stub)
            first_thru_node = header["FIRST THRU NODE"]
            print(
                f"\nFIRST_THRU_NODE={first_thru_node}, N_ZONES={header['NUMBER OF ZONES']}"
            )

            if first_thru_node <= 1:
                quetzal_mode = "embedded zones (synthetic zone connectors)"
            else:
                quetzal_mode = "separated zones (native centroid connectors)"
            print(f"Quetzal build mode: {quetzal_mode}")

            mat = build_matrix(data_folder, model_stub)
            g = build_graph(data_folder, model_stub, mat.index)
            tntp = known_results(data_folder, model_stub)
            link_lookup = g.network[["link_id", "a_node", "b_node"]].set_index("link_id")

            for method in METHODS:
                print(f"\n{'-' * 60}")
                print(f"METHOD: {method.upper()}")
                print(f"{'-' * 60}")

                # --- AequilibraE ---
                print(f"  Running AequilibraE ({method})...")
                t0 = perf_counter()
                assig = assign_aeq(g, mat, method)
                t_aeq = perf_counter() - t0
                aeq_report = assig.report()
                aeq_results = assig.results()  # indexed by link_id; PCE_AB for direction=1
                print(
                    f"  AequilibraE done in {t_aeq:.1f}s, "
                    f"final rgap={aeq_report['rgap'].iloc[-1]:.2e}"
                )

                aeq_with_nodes = (
                    aeq_results[["PCE_AB"]]
                    .join(link_lookup)
                    .merge(tntp, on=["a_node", "b_node"], how="inner")
                )

                # --- Quetzal (skip cfw) ---
                if method.lower() == "cfw":
                    print(
                        f"  Skipping Quetzal ({method}) - method not supported by Quetzal"
                    )
                    qtl_convergence = None
                    qtl_with_nodes = None
                    qtl_road_flows = None
                    aeq_road_flows = None
                else:
                    print("  Building Quetzal model...")
                    sm = build_quetzal_model(g.network, first_thru_node, mat)
                    print(f"  Running Quetzal ({method})...")
                    t0 = perf_counter()
                    assign_quetzal(sm, method)
                    t_qtl = perf_counter() - t0
                    print(
                        f"  Quetzal done in {t_qtl:.1f}s, "
                        f"final rgap={sm.relgap[-1] / 100.0:.2e}"
                    )

                    qtl_convergence = pd.DataFrame({"relgap": sm.relgap, "time": sm.times})
                    road_link_ids = sm.road_links.index
                    aeq_road_flows = aeq_results.loc[road_link_ids, "PCE_AB"]
                    qtl_road_flows = sm.road_links["flow"]
                    qtl_with_nodes = (
                        sm.road_links[["flow"]]
                        .join(link_lookup)
                        .merge(tntp, on=["a_node", "b_node"], how="inner")
                    )

                    diff = qtl_road_flows - aeq_road_flows
                    print(
                        f"\n  Road-link flow difference (Quetzal - AequilibraE):"
                        f"\n{diff.describe().to_string()}\n"
                    )

                # --- Save results ---
                prefix = data_folder / model_stub
                aeq_report.to_parquet(f"{prefix}_aeq_report_{method}.parquet")
                aeq_with_nodes.to_parquet(f"{prefix}_aeq_with_nodes_{method}.parquet")
                if qtl_convergence is not None:
                    qtl_convergence.to_parquet(
                        f"{prefix}_qtl_convergence_{method}.parquet"
                    )
                if qtl_with_nodes is not None:
                    qtl_with_nodes.to_parquet(
                        f"{prefix}_qtl_with_nodes_{method}.parquet"
                    )
                if qtl_road_flows is not None:
                    qtl_road_flows.to_frame("flow").to_parquet(
                        f"{prefix}_qtl_road_flows_{method}.parquet"
                    )
                if aeq_road_flows is not None:
                    aeq_road_flows.to_frame("PCE_AB").to_parquet(
                        f"{prefix}_aeq_road_flows_{method}.parquet"
                    )
                print(f"  Saved results for {model_name}/{method}.")


    def plot_results() -> None:
        """Part 2: Load saved results from disk and generate all plots."""
        for model_name, (data_folder, model_stub) in MODELS.items():
            print(f"\n{'=' * 60}")
            print(f"MODEL: {model_name.upper()}")
            print(f"{'=' * 60}")

            prefix = data_folder / model_stub
            aeq_method_reports: dict[str, pd.DataFrame] = {}

            for method in METHODS:
                aeq_report_path = Path(f"{prefix}_aeq_report_{method}.parquet")
                if not aeq_report_path.exists():
                    print(f"  Skipping {method.upper()} - no saved results found.")
                    continue

                aeq_report = pd.read_parquet(aeq_report_path)
                aeq_with_nodes = pd.read_parquet(
                    f"{prefix}_aeq_with_nodes_{method}.parquet"
                )
                aeq_method_reports[method.upper()] = aeq_report

                qtl_path = Path(f"{prefix}_qtl_convergence_{method}.parquet")
                if qtl_path.exists():
                    qtl_convergence = pd.read_parquet(qtl_path)
                    qtl_with_nodes = pd.read_parquet(
                        f"{prefix}_qtl_with_nodes_{method}.parquet"
                    )
                    qtl_road_flows = pd.read_parquet(
                        f"{prefix}_qtl_road_flows_{method}.parquet"
                    )["flow"]
                    aeq_road_flows = pd.read_parquet(
                        f"{prefix}_aeq_road_flows_{method}.parquet"
                    )["PCE_AB"]
                else:
                    qtl_convergence = None
                    qtl_with_nodes = None
                    qtl_road_flows = None
                    aeq_road_flows = None

                # --- Convergence plot ---
                plot_convergence(
                    qtl_convergence,
                    aeq_report,
                    model_name.title(),
                    method,
                    RGAP_TARGET,
                    plot_time=True,
                    save_path=data_folder / f"{model_stub}_convergence_{method}.png",
                )

                # --- Flow dashboard ---
                plot_flow_dashboard(
                    aeq_with_nodes,
                    qtl_with_nodes,
                    qtl_road_flows,
                    aeq_road_flows,
                    model_name.title(),
                    method,
                    RGAP_TARGET,
                    save_path=data_folder / f"{model_stub}_flow_dashboard_{method}.png",
                )

            # --- AequilibraE-only method convergence comparison ---
            if aeq_method_reports:
                plot_aeq_method_convergence_times(
                    aeq_method_reports,
                    model_name.title(),
                    RGAP_TARGET,
                    save_path=data_folder
                    / f"{model_stub}_aeq_method_convergence_time.png",
                )
