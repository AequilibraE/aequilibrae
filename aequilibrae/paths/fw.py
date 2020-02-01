import numpy as np
from typing import List
from aequilibrae.paths.assignment_class import AssignmentClass
from aequilibrae.paths.all_or_nothing import allOrNothing
from aequilibrae.paths.results import AssignmentResults
from aequilibrae import Parameters
from aequilibrae.paths.vdf import VDF
from aequilibrae import logger

from scipy.optimize import root_scalar


class FW:
    def __init__(self, traffic_classes: List[AssignmentClass]):
        parameters = Parameters().parameters["assignment"]["equilibrium"]
        self.rgap_target = parameters["rgap"]
        self.max_iter = parameters["maximum_iterations"]

        # A single class for now
        self.graph = traffic_classes[0].graph
        self.matrix = traffic_classes[0].matrix
        self.final_results = traffic_classes[0].results

        self.aon_results = AssignmentResults()
        self.aon_results.prepare(self.graph, self.matrix)
        self.iter = 0
        self.rgap = np.inf
        self.vdf = VDF()

    def execute(self):
        for self.iter in range(1, self.max_iter + 1):
            aon = allOrNothing(self.matrix, self.graph, self.aon_results)
            aon.execute()
            self.aon_class_flow = np.sum(self.aon_results.link_loads, axis=1)

            # calculate stepsize
            self.stepsize = self.calculate_stepsize()

            self.final_results.link_loads[:, :] *= 1.0 - self.stepsize
            self.final_results.link_loads[:, :] += self.aon_results.link_loads[:, :] * self.stepsize

            self.fw_class_flow = np.sum(self.final_results.link_loads, axis=1)

            self.congested_time = self.vdf.apply_vdf(
                "BPR", link_flows=self.fw_class_flow, capacity=self.graph.capacity, fftime=self.graph.free_flow_time
            )
            self.graph.cost = self.congested_time

            # Check convergence
            if self.check_convergence() and self.iter > 1:
                break
            self.aon_results.reset()

        if self.rgap > self.rgap_target:
            logger.error("Desired RGap of {} was NOT reached".format(self.rgap_target))
        logger.info("FW Assignment finished. {} iterations and {} final gap".format(self.iter, self.rgap))

    def calculate_stepsize(self):
        """Calculate optimal stepsize in gradient direction"""
        # First iteration gets 100% of shortest path
        if self.iter == 1:
            return 1.0

        def derivative_of_objective(stepsize):
            x = self.fw_class_flow + stepsize * (self.aon_class_flow - self.fw_class_flow)
            congested_value = self.vdf.apply_vdf(
                "BPR", link_flows=x, capacity=self.graph.capacity, fftime=self.graph.free_flow_time
            )
            return np.sum(congested_value * (self.aon_class_flow - self.fw_class_flow))

        min_res = root_scalar(derivative_of_objective, bracket=(0, 1))
        print(min_res)
        stepsize = min_res.root
        assert 0 <= stepsize <= 1.0
        if not min_res.converged:
            logger.warn("Frank Wolfe stepsize finder is not converged")
        return stepsize

    def check_convergence(self):
        aon_cost = np.sum(self.congested_time * self.aon_class_flow)
        fw_cost = np.sum(self.congested_time * self.fw_class_flow)
        self.rgap = abs(fw_cost - aon_cost) / fw_cost
        print(self.iter, self.rgap)
        if self.rgap_target >= self.rgap:
            return True
        return False
