import importlib.util as iutil
import numpy as np
from typing import List, Dict
from warnings import warn
import cvxopt
import array

from ..utils import WorkerThread
from aequilibrae.paths.traffic_class import TrafficClass
from aequilibrae.paths.results import AssignmentResults
from aequilibrae import logger
import pandas as pd  # temporary for data structures
from aequilibrae.paths.link import Link, Node

try:
    from aequilibrae.paths import TrafficAssignmentCy

    # temp for one to all shortest path hack
    from aequilibrae.paths.multi_threaded_aon import MultiThreadedAoN
    from aequilibrae.paths.AoN import one_to_all
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
        self.cores = assig_spec.cores
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

        for c in self.traffic_classes:
            r = AssignmentResults()
            r.prepare(c.graph, c.matrix)
        #     self.step_direction[c.mode] = r

        self.t_assignment = None

        cvxopt.solvers.options["show_progress"] = False
        # cvxopt.solvers.options['maxiters'] = 6 #
        cvxopt.solvers.options["abstol"] = 1e-11
        cvxopt.solvers.options["reltol"] = 1e-11
        cvxopt.solvers.options["feastol"] = 1e-11

    def initialise_data_structures(self):
        """Wrapper around OpenBenchmark.build_datastructure"""

        # fix to one class for now
        graph_ = pd.DataFrame(self.traffic_classes[0].graph.graph)

        # unique nodes
        x_ = set(graph_["a_node"].unique())
        x_ = x_.union(set(graph_["b_node"].unique()))
        self.nodes = [Node(node_id=x) for x in x_]

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

        self.links = graph_.apply(create_link, axis=1).to_list()

        # ods
        num_zones = self.traffic_classes[0].matrix.zones
        mat_ = self.traffic_classes[0].matrix.get_matrix("matrix")
        self.ods = {(o, d): mat_[o, d] for o in range(0, num_zones) for d in range(0, num_zones) if mat_[o, d] > 0.0}

        # from OpenBenchmark:
        destinations = []
        origins = []
        for (origin, destination) in self.ods:
            if destination not in destinations:
                destinations.append(destination)
            if origin not in origins:
                origins.append(origin)
        self.destinations = destinations
        self.origins = origins

    def initial_iteration(self):
        c = self.traffic_classes[0]
        matrix = c.matrix
        graph = c.graph
        results = c._aon_results
        for origin in range(len(self.origins)):
            aux_res = MultiThreadedAoN()
            aux_res.prepare(graph, results)
            matrix.matrix_view = c.matrix.matrix_view.reshape(
                (graph.num_zones, graph.num_zones, results.classes["number"])
            )
            th = 0  # th is thread id
            origin_aeq = origin + 1  # sort out this mess
            _ = one_to_all(origin_aeq, matrix, graph, results, aux_res, th)

            # set precedence to aux_res.predecessors[:,0]
            prec = aux_res.predecessors[:, 0]
            self.t_assignment.set_precedence(prec)
            self.t_assignment.compute_path_link_sequence_external_precedence(origin)
            self.t_assignment.set_initial_path_flows(origin)

        for origin in range(len(self.origins)):
            self.t_assignment.update_link_flows(origin)

        # c++ data structures and aequilibrae data structures are not integrated yet
        self.update_time_field_for_path_computation()

    def shortest_path_temp_wrapper(self, origin):
        c = self.traffic_classes[0]
        matrix = c.matrix
        graph = c.graph
        results = c._aon_results
        aux_res = MultiThreadedAoN()
        aux_res.prepare(graph, results)
        matrix.matrix_view = c.matrix.matrix_view.reshape((graph.num_zones, graph.num_zones, results.classes["number"]))
        th = 0  # th is thread id
        origin_aeq = origin + 1  # sort out this mess
        _ = one_to_all(origin_aeq, matrix, graph, results, aux_res, th)

        # set precedence to aux_res.predecessors[:,0]
        prec = aux_res.predecessors[:, 0]
        self.t_assignment.set_precedence(prec)
        self.t_assignment.compute_path_link_sequence_external_precedence(origin)

    def update_time_field_for_path_computation(self):
        total_flow_ = self.t_assignment.get_link_flows()
        # let's just hack it, most data structures will be integrated anyways
        self.total_flow = array.array("d", total_flow_)
        self.traffic_classes[0].results.link_loads = self.total_flow

        self.vdf.apply_vdf(
            self.congested_time, self.total_flow, self.capacity, self.free_flow_tt, *self.vdf_parameters, self.cores
        )
        c = self.traffic_classes[0]
        c.graph.cost = self.congested_time
        if self.time_field in c.graph.skim_fields:
            idx = c.graph.skim_fields.index(self.time_field)
            c.graph.skims[:, idx] = self.congested_time[:]
        c._aon_results.reset()  # not used atm, we do not assign link flows along shortest paths
        c.results.link_loads = self.total_flow

    def doWork(self):
        self.execute()

    def execute(self):
        do_parallel = False
        if "parallel" in self.algorithm:
            do_parallel = True

        if do_parallel:
            self.execute_parallel()
        else:
            self.execute_sequential()

    def execute_parallel(self):
        use_boost = False

        for c in self.traffic_classes:
            c.graph.set_graph(self.time_field)

        logger.info(f"{self.algorithm} parallel Assignment STATS")

        self.initialise_data_structures()
        num_links = len(self.links)
        num_nodes = len(self.nodes)
        num_centroids = len(self.origins)
        logger.info(f" Initialised data structures, num nodes = {num_nodes}, num links = {num_links}")

        self.t_assignment = TrafficAssignmentCy.TrafficAssignmentCy(self.links, num_links, num_nodes, num_centroids)
        destinations_per_origin = {}
        for (o, d) in self.ods:
            self.t_assignment.insert_od(o, d, self.ods[o, d])
            if o not in destinations_per_origin:
                destinations_per_origin[o] = 0
            destinations_per_origin[o] += 1

        if use_boost:
            self.t_assignment.perform_initial_solution()
        else:
            self.initial_iteration()

        logger.info(f" 0th iteration done, cost = {self.t_assignment.get_objective_function()}")

        for self.iter in range(1, self.max_iter + 1):
            self.iteration_issue = []
            if pyqt:
                self.equilibration.emit(["rgap", self.rgap])
                self.equilibration.emit(["iterations", self.iter])

            origins = destinations_per_origin.keys()

            for origin in origins:
                if use_boost:
                    self.t_assignment.compute_shortest_paths(origin)
                else:
                    self.shortest_path_temp_wrapper(origin)

                t_paths = self.t_assignment.get_total_paths(origin)
                Q, q, A, b, G, h = self.t_assignment.get_problem_data(origin, destinations_per_origin[origin])
                Am = cvxopt.matrix(A.tolist(), (t_paths, destinations_per_origin[origin]), "d")
                bm = cvxopt.matrix(b.tolist(), (destinations_per_origin[origin], 1), "d")
                Qm = cvxopt.matrix(Q.tolist(), (t_paths, t_paths), "d")
                qm = cvxopt.matrix(q.tolist(), (t_paths, 1), "d")
                Gm = cvxopt.matrix(G.tolist(), (t_paths, t_paths), "d")
                hm = cvxopt.matrix(h.tolist(), (t_paths, 1), "d")
                solution = cvxopt.solvers.qp(Qm.trans(), qm, Gm.trans(), hm, Am.trans(), bm)["x"]

                self.t_assignment.update_current_iteration_flows_by_origin(origin, solution)

            # not parallel:
            try:
                min_res = root_scalar(lambda x: self.t_assignment.objective_derivative_stepsize(x), bracket=[0, 1])
                self.stepsize = min_res.root
            except ValueError:
                left_ = self.t_assignment.objective_derivative_stepsize(0)
                right_ = self.t_assignment.objective_derivative_stepsize(1)
                logger.warning(
                    f"Error finding stepsize: left derivative is {left_:.0f}, right derivative is {right_:.0f}"
                )
                # TODO: need to decide what to do here
                if left_ < right_:
                    self.stepsize = 0.01
                else:
                    self.stepsize = 0.99

            # update solution
            self.t_assignment.update_link_flows_stepsize(self.stepsize)

            if not use_boost:
                # c++ data structures and aequilibrae data structures are not integrated yet
                self.update_time_field_for_path_computation()

            # now this can be parallel again, loop over origins in c++
            # for origin in origins:
            self.t_assignment.update_path_flows_stepsize(self.stepsize)

            this_cost = self.t_assignment.get_objective_function()

            if use_boost:
                self.traffic_classes[0].results.link_loads = self.t_assignment.get_link_flows()

            converged = self.check_convergence()

            logger.info(
                f"Iteration {self.iter}, computed gap: {self.rgap}, computed objective: {this_cost}, stepsize: {self.stepsize}"
            )
            # self.convergence_report["iteration"].append(self.iter)
            # self.convergence_report["rgap"].append(self.rgap)
            # self.convergence_report["warnings"].append("; ".join(self.iteration_issue))

            if converged:
                break

        if self.rgap > self.rgap_target:
            logger.error(f"Desired RGap of {self.rgap_target} was NOT reached")
        logger.info(f"{self.algorithm} Assignment finished. {self.iter} iterations and {self.rgap} final gap")
        if pyqt:
            self.equilibration.emit(["rgap", self.rgap])
            self.equilibration.emit(["iterations", self.iter])
            self.equilibration.emit(["finished_threaded_procedure"])

    def execute_sequential(self):
        use_boost = False

        for c in self.traffic_classes:
            c.graph.set_graph(self.time_field)

        logger.info(f"{self.algorithm} sequential Assignment STATS")
        # logger.info("Iteration, RelativeGap, stepsize")

        # links, nodes, ods, destinations, origins =
        self.initialise_data_structures()
        num_links = len(self.links)
        num_nodes = len(self.nodes)
        num_centroids = len(self.origins)
        logger.info(f" Initialised data structures, num nodes = {num_nodes}, num links = {num_links}")

        self.t_assignment = TrafficAssignmentCy.TrafficAssignmentCy(self.links, num_links, num_nodes, num_centroids)
        destinations_per_origin = {}
        for (o, d) in self.ods:
            self.t_assignment.insert_od(o, d, self.ods[o, d])
            if o not in destinations_per_origin:
                destinations_per_origin[o] = 0
            destinations_per_origin[o] += 1

        if use_boost:
            self.t_assignment.perform_initial_solution()
        else:
            self.initial_iteration()

        logger.info(f" 0th iteration done, cost = {self.t_assignment.get_objective_function()}")

        for self.iter in range(1, self.max_iter + 1):
            self.iteration_issue = []
            if pyqt:
                self.equilibration.emit(["rgap", self.rgap])
                self.equilibration.emit(["iterations", self.iter])

            origins = destinations_per_origin.keys()
            for origin in origins:

                if use_boost:
                    self.t_assignment.compute_shortest_paths(origin)
                else:
                    self.shortest_path_temp_wrapper(origin)

                t_paths = self.t_assignment.get_total_paths(origin)
                Q, q, A, b, G, h = self.t_assignment.get_problem_data(origin, destinations_per_origin[origin])
                Am = cvxopt.matrix(A.tolist(), (t_paths, destinations_per_origin[origin]), "d")
                bm = cvxopt.matrix(b.tolist(), (destinations_per_origin[origin], 1), "d")
                Qm = cvxopt.matrix(Q.tolist(), (t_paths, t_paths), "d")
                qm = cvxopt.matrix(q.tolist(), (t_paths, 1), "d")
                Gm = cvxopt.matrix(G.tolist(), (t_paths, t_paths), "d")
                hm = cvxopt.matrix(h.tolist(), (t_paths, 1), "d")
                solution = cvxopt.solvers.qp(Qm.trans(), qm, Gm.trans(), hm, Am.trans(), bm)["x"]
                self.t_assignment.update_path_flows(origin, solution)

                if not use_boost:
                    # c++ data structures and aequilibrae data structures are not integrated yet
                    self.update_time_field_for_path_computation()

            this_cost = self.t_assignment.get_objective_function()

            if use_boost:
                self.traffic_classes[0].results.link_loads = self.t_assignment.get_link_flows()

            converged = self.check_convergence()

            logger.info(f"Iteration {self.iter}, computed gap: {self.rgap}, computed objective: {this_cost}")
            # self.convergence_report["iteration"].append(self.iter)
            # self.convergence_report["rgap"].append(self.rgap)
            # self.convergence_report["warnings"].append("; ".join(self.iteration_issue))

            if converged:
                break

        if self.rgap > self.rgap_target:
            logger.error(f"Desired RGap of {self.rgap_target} was NOT reached")
        logger.info(f"{self.algorithm} Assignment finished. {self.iter} iterations and {self.rgap} final gap")
        if pyqt:
            self.equilibration.emit(["rgap", self.rgap])
            self.equilibration.emit(["iterations", self.iter])
            self.equilibration.emit(["finished_threaded_procedure"])

    def check_convergence(self):
        """Calculate relative gap and return True if it is smaller than desired precision"""

        num = 0.0
        dem = 0.0
        for (origin, destination) in self.ods:
            p_times, p_flows = self.t_assignment.get_path_times(origin, destination)
            num += sum(p_flows) * min(p_times)
            for ind in range(len(p_times)):
                dem += p_times[ind] * p_flows[ind]
        self.rgap = 1 - num / dem
        if self.rgap_target >= self.rgap:
            return True
        return False

    def signal_handler(self, val):
        if pyqt:
            self.assignment.emit(val)
