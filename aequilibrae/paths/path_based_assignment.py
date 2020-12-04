import importlib.util as iutil
import numpy as np
from typing import List, Dict
from warnings import warn
from ..utils import WorkerThread
from aequilibrae.paths.traffic_class import TrafficClass
from aequilibrae.paths.results import AssignmentResults
from aequilibrae.paths.all_or_nothing import allOrNothing
from aequilibrae import logger

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

        for c in self.traffic_classes:
            r = AssignmentResults()
            r.prepare(c.graph, c.matrix)
            self.step_direction[c.mode] = r

    def doWork(self):
        self.execute()

    def execute(self):
        for c in self.traffic_classes:
            c.graph.set_graph(self.time_field)

        logger.info(f"{self.algorithm} Assignment STATS")
        logger.info("Iteration, RelativeGap, stepsize")
        for self.iter in range(1, self.max_iter + 1):
            self.iteration_issue = []
            if pyqt:
                self.equilibration.emit(["rgap", self.rgap])
                self.equilibration.emit(["iterations", self.iter])
            flows = []
            aon_flows = []

            for c in self.traffic_classes:
                aon = allOrNothing(c.matrix, c.graph, c._aon_results)
                if pyqt:
                    aon.assignment.connect(self.signal_handler)
                aon.execute()
                c._aon_results.total_flows()
                aon_flows.append(c._aon_results.total_link_loads * c.pce)
            self.aon_total_flow = np.sum(aon_flows, axis=0)

            if self.iter == 1:
                for c in self.traffic_classes:
                    copy_two_dimensions(c.results.link_loads, c._aon_results.link_loads, self.cores)
                    c.results.total_flows()
                    copy_one_dimension(c.results.total_link_loads, c._aon_results.total_link_loads, self.cores)
                    if c.results.num_skims > 0:
                        copy_three_dimensions(c.results.skims.matrix_view, c._aon_results.skims.matrix_view, self.cores)
                    flows.append(c.results.total_link_loads * c.pce)
            else:
                self.__calculate_step_direction()
                self.calculate_stepsize()
                for c in self.traffic_classes:
                    stp_dir = self.step_direction[c.mode]
                    cls_res = c.results
                    linear_combination(
                        cls_res.link_loads, stp_dir.link_loads, cls_res.link_loads, self.stepsize, self.cores
                    )
                    if cls_res.num_skims > 0:
                        linear_combination_skims(
                            cls_res.skims.matrix_view,
                            stp_dir.skims.matrix_view,
                            cls_res.skims.matrix_view,
                            self.stepsize,
                            self.cores,
                        )
                    cls_res.total_flows()
                    flows.append(cls_res.total_link_loads * c.pce)

            self.fw_total_flow = np.sum(flows, axis=0)

            # Check convergence
            # This needs to be done with the current costs, and not the future ones
            converged = False
            if self.iter > 1:
                converged = self.check_convergence()

            self.convergence_report["iteration"].append(self.iter)
            self.convergence_report["rgap"].append(self.rgap)
            self.convergence_report["warnings"].append("; ".join(self.iteration_issue))
            self.convergence_report["alpha"].append(self.stepsize)

            if self.algorithm == "bfw":
                self.convergence_report["beta0"].append(self.betas[0])
                self.convergence_report["beta1"].append(self.betas[1])
                self.convergence_report["beta2"].append(self.betas[2])

            logger.info(f"{self.iter},{self.rgap},{self.stepsize}")
            if converged:
                if self.steps_below >= self.steps_below_needed_to_terminate:
                    break
                else:
                    self.steps_below += 1

            self.vdf.apply_vdf(
                self.congested_time, self.fw_total_flow, self.capacity, self.free_flow_tt, *self.vdf_parameters
            )

            for c in self.traffic_classes:
                c.graph.cost = self.congested_time
                if self.time_field in c.graph.skim_fields:
                    idx = c.graph.skim_fields.index(self.time_field)
                    c.graph.skims[:, idx] = self.congested_time[:]
                c._aon_results.reset()

        if self.rgap > self.rgap_target:
            logger.error(f"Desired RGap of {self.rgap_target} was NOT reached")
        logger.info(f"{self.algorithm} Assignment finished. {self.iter} iterations and {self.rgap} final gap")
        if pyqt:
            self.equilibration.emit(["rgap", self.rgap])
            self.equilibration.emit(["iterations", self.iter])
            self.equilibration.emit(["finished_threaded_procedure"])

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
