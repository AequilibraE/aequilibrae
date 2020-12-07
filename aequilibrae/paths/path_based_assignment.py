import importlib.util as iutil
import numpy as np
from typing import List, Dict
from warnings import warn
from ..utils import WorkerThread
from aequilibrae.paths.traffic_class import TrafficClass
from aequilibrae.paths.results import AssignmentResults
from aequilibrae.paths.all_or_nothing import allOrNothing
from aequilibrae import logger

from aequilibrae.paths.path_based import TrafficAssignmentCy
import pandas as pd
from aequilibrae.paths.path_based.node import Node
from aequilibrae.paths.path_based.link import Link


try:
    from aequilibrae.paths.AoN import linear_combination, linear_combination_skims
    from aequilibrae.paths.AoN import triple_linear_combination, triple_linear_combination_skims
    from aequilibrae.paths.AoN import copy_one_dimension, copy_two_dimensions, copy_three_dimensions
except ImportError as ie:
    logger.warning(f"Could not import procedures from the binary. {ie.args}")

import scipy

if int(scipy.__version__.split(".")[1]) >= 3:
    from scipy.optimize import root_scalar

    recent_scipy = True
else:
    from scipy.optimize import root as root_scalar

    recent_scipy = False
    logger.warning("Using older version of Scipy. For better performance, use Scipy >= 1.4")

if False:
    from aequilibrae.paths.traffic_assignment import TrafficAssignment

spec = iutil.find_spec("PyQt5")
pyqt = spec is not None
if pyqt:
    from PyQt5.QtCore import pyqtSignal as SIGNAL


class PathBasedAssignment(WorkerThread):
    if pyqt:
        equilibration = SIGNAL(object)
        assignment = SIGNAL(object)

    def __init__(self, assig_spec, algorithm) -> None:
        WorkerThread.__init__(self, None)
        self.algorithm = algorithm
        self.rgap_target = assig_spec.rgap_target
        self.max_iter = assig_spec.max_iter
        self.cores = assig_spec.cores
        self.iteration_issue = []
        self.convergence_report = {"iteration": [], "rgap": [], "alpha": [], "warnings": []}

        self.assig = assig_spec  # type: TrafficAssignment

        if None in [
            assig_spec.classes,
            assig_spec.vdf,
            assig_spec.capacity_field,
            assig_spec.time_field,
            assig_spec.vdf_parameters,
        ]:
            all_par = "Traffic classes, VDF, VDF_parameters, capacity field & time_field"
            raise Exception(
                "Parameter missing. Setting the algorithm is the last thing to do "
                f"when assigning. Check if you have all of these: {all_par}"
            )

        self.traffic_classes = assig_spec.classes  # type: List[TrafficClass]
        self.num_classes = len(assig_spec.classes)

        self.cap_field = assig_spec.capacity_field
        self.time_field = assig_spec.time_field
        self.vdf = assig_spec.vdf
        self.vdf_parameters = assig_spec.vdf_parameters

        self.iter = 0
        self.rgap = np.inf
        self.stepsize = 1.0
        self.class_flow = 0

        # Instantiates the arrays that we will use over and over
        self.capacity = assig_spec.capacity
        self.free_flow_tt = assig_spec.free_flow_tt
        self.total_flow = assig_spec.total_flow
        self.congested_time = assig_spec.congested_time
        self.vdf_der = np.array(assig_spec.congested_time, copy=True)
        self.congested_value = np.array(assig_spec.congested_time, copy=True)

        # for c in self.traffic_classes:
        #     r = AssignmentResults()
        #     r.prepare(c.graph, c.matrix)
        #     self.step_direction[c.mode] = r

        self.t_assignment = None

    def initialise_data_structures(self):
        """Wrapper around OpenBenchmark.build_datastructure"""

        # fix to one class for now
        graph_ = pd.DataFrame(self.traffic_classes[0].graph.graph)

        # unique nodes
        x_ = set(graph_["a_node"].unique())
        x_ = x_.union(set(graph_["b_node"].unique()))
        nodes = [Node(node_id=x) for x in x_]

        # links
        def create_link(row):
            link_id = row["link_id"]
            t0 = row["time"]
            capacity = row["capacity"]
            alfa = row["alpha"]
            power = row["beta"]
            origin_node = row["a_node"]
            to_node = row["b_node"]
            return Link(
                link_id=link_id,
                t0=t0,
                capacity=capacity,
                alfa=alfa,
                beta=power,
                node_id_from=origin_node,
                node_id_to=to_node,
            )

        links = graph_.apply(create_link, axis=1).to_list()

        # ods
        num_zones = self.traffic_classes[0].matrix.zones
        mat_ = self.traffic_classes[0].matrix.get_matrix("matrix")
        ods = {(o, d): mat_[o, d] for o in range(0, num_zones) for d in range(0, num_zones)}

        # from OpenBenchmark:
        destinations = []
        origins = []
        for (origin, destination) in ods:
            if destination not in destinations:
                destinations.append(destination)
            if origin not in origins:
                origins.append(origin)

        return links, nodes, ods, destinations, origins

    def doWork(self):
        self.execute()

    def execute(self):
        for c in self.traffic_classes:
            c.graph.set_graph(self.time_field)

        logger.info(f"{self.algorithm} Assignment STATS")
        # logger.info("Iteration, RelativeGap, stepsize")

        links, nodes, ods, destinations, origins = self.initialise_data_structures()
        num_links = len(links)
        num_nodes = len(nodes)
        num_centroids = len(origins)
        logger.info(f" Initialised data structures, num nodes = {len(nodes)}, num links = {len(links)}")

        self.t_assignment = TrafficAssignmentCy.TrafficAssignmentCy(links, num_links, num_nodes, num_centroids)
        destinations_per_origin = {}
        for (o, d) in ods:
            self.t_assignment.insert_od(o, d, ods[o, d])
            if o not in destinations_per_origin:
                destinations_per_origin[o] = 0
            destinations_per_origin[o] += 1

        ### Ini solution, iter 0
        self.t_assignment.perform_initial_solution()
        costs = [self.t_assignment.get_objective_function()]
        # gaps = [1]  # initial gap, fix to arbitrary value or better compute it
        logger.info(f" 0th iteration done, cost = {costs[0]}")

    #########

    # for self.iter in range(1, self.max_iter + 1):
    #     self.iteration_issue = []
    #     if pyqt:
    #         self.equilibration.emit(["rgap", self.rgap])
    #         self.equilibration.emit(["iterations", self.iter])
    #     flows = []
    #     aon_flows = []
    #
    #     for c in self.traffic_classes:
    #         aon = allOrNothing(c.matrix, c.graph, c._aon_results)
    #         if pyqt:
    #             aon.assignment.connect(self.signal_handler)
    #         aon.execute()
    #         c._aon_results.total_flows()
    #         aon_flows.append(c._aon_results.total_link_loads * c.pce)
    #     self.aon_total_flow = np.sum(aon_flows, axis=0)
    #
    #     if self.iter == 1:
    #         for c in self.traffic_classes:
    #             copy_two_dimensions(c.results.link_loads, c._aon_results.link_loads, self.cores)
    #             c.results.total_flows()
    #             copy_one_dimension(c.results.total_link_loads, c._aon_results.total_link_loads, self.cores)
    #             if c.results.num_skims > 0:
    #                 copy_three_dimensions(c.results.skims.matrix_view, c._aon_results.skims.matrix_view, self.cores)
    #             flows.append(c.results.total_link_loads * c.pce)
    #     else:
    #         self.__calculate_step_direction()
    #         self.calculate_stepsize()
    #         for c in self.traffic_classes:
    #             stp_dir = self.step_direction[c.mode]
    #             cls_res = c.results
    #             linear_combination(
    #                 cls_res.link_loads, stp_dir.link_loads, cls_res.link_loads, self.stepsize, self.cores
    #             )
    #             if cls_res.num_skims > 0:
    #                 linear_combination_skims(
    #                     cls_res.skims.matrix_view,
    #                     stp_dir.skims.matrix_view,
    #                     cls_res.skims.matrix_view,
    #                     self.stepsize,
    #                     self.cores,
    #                 )
    #             cls_res.total_flows()
    #             flows.append(cls_res.total_link_loads * c.pce)
    #
    #     self.fw_total_flow = np.sum(flows, axis=0)
    #
    #     # Check convergence
    #     # This needs to be done with the current costs, and not the future ones
    #     converged = False
    #     if self.iter > 1:
    #         converged = self.check_convergence()
    #
    #     self.convergence_report["iteration"].append(self.iter)
    #     self.convergence_report["rgap"].append(self.rgap)
    #     self.convergence_report["warnings"].append("; ".join(self.iteration_issue))
    #     self.convergence_report["alpha"].append(self.stepsize)
    #
    #     if self.algorithm == "bfw":
    #         self.convergence_report["beta0"].append(self.betas[0])
    #         self.convergence_report["beta1"].append(self.betas[1])
    #         self.convergence_report["beta2"].append(self.betas[2])
    #
    #     logger.info(f"{self.iter},{self.rgap},{self.stepsize}")
    #     if converged:
    #         if self.steps_below >= self.steps_below_needed_to_terminate:
    #             break
    #         else:
    #             self.steps_below += 1
    #
    #     self.vdf.apply_vdf(
    #         self.congested_time, self.fw_total_flow, self.capacity, self.free_flow_tt, *self.vdf_parameters
    #     )
    #
    #     for c in self.traffic_classes:
    #         c.graph.cost = self.congested_time
    #         if self.time_field in c.graph.skim_fields:
    #             idx = c.graph.skim_fields.index(self.time_field)
    #             c.graph.skims[:, idx] = self.congested_time[:]
    #         c._aon_results.reset()
    #
    # if self.rgap > self.rgap_target:
    #     logger.error(f"Desired RGap of {self.rgap_target} was NOT reached")
    # logger.info(f"{self.algorithm} Assignment finished. {self.iter} iterations and {self.rgap} final gap")
    # if pyqt:
    #     self.equilibration.emit(["rgap", self.rgap])
    #     self.equilibration.emit(["iterations", self.iter])
    #     self.equilibration.emit(["finished_threaded_procedure"])

    def check_convergence(self):
        """Calculate relative gap and return True if it is smaller than desired precision"""
        aon_cost = np.sum(self.congested_time * self.aon_total_flow)
        current_cost = np.sum(self.congested_time * self.fw_total_flow)
        self.rgap = abs(current_cost - aon_cost) / current_cost
        if self.rgap_target >= self.rgap:
            return True
        return False

    def signal_handler(self, val):
        if pyqt:
            self.assignment.emit(val)
