import numpy as np
from scipy.optimize import root_scalar
from typing import List

from aequilibrae.paths.assignment_class import AssignmentClass
from aequilibrae.paths.all_or_nothing import allOrNothing
from aequilibrae.paths.results import AssignmentResults
from aequilibrae import Parameters
from aequilibrae.paths.vdf import VDF
from aequilibrae import logger


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
        self.stepsize = 1.0

        # rgap can be a bit wiggly, specifying how many times we need to be below target rgap is a quick way to
        # ensure a better result. We could also demand that the solution is that many consecutive times below,
        # but we are talking about small oscillations so not really necessary.
        self.steps_below = 0
        self.steps_below_needed_to_terminate = 1

    def execute(self):
        for self.iter in range(1, self.max_iter + 1):
            aon = allOrNothing(self.matrix, self.graph, self.aon_results)
            aon.execute()
            self.aon_class_flow = np.sum(self.aon_results.link_loads, axis=1)

            _ = self.calculate_stepsize()
            # self.stepsize = 1. / float(self.iter)
            # print(self.stepsize)
            self.final_results.link_loads[:, :] = self.final_results.link_loads[:, :] * (1.0 - self.stepsize)
            self.final_results.link_loads[:, :] += self.aon_results.link_loads[:, :] * self.stepsize

            self.fw_class_flow = np.sum(self.final_results.link_loads, axis=1)

            self.congested_time = self.vdf.apply_vdf(
                "BPR", link_flows=self.fw_class_flow, capacity=self.graph.capacity, fftime=self.graph.free_flow_time
            )
            self.graph.cost = self.congested_time

            # Check convergence
            if self.check_convergence() and self.iter > 1:
                if self.steps_below >= self.steps_below_needed_to_terminate:
                    break
                else:
                    self.steps_below += 1

            self.aon_results.reset()

        if self.rgap > self.rgap_target:
            logger.error("Desired RGap of {} was NOT reached".format(self.rgap_target))
        logger.info("FW Assignment finished. {} iterations and {} final gap".format(self.iter, self.rgap))
        # print("FW Assignment finished. {} iterations and {} final gap".format(self.iter, self.rgap))

    def calculate_stepsize(self):
        """Calculate optimal stepsize in gradient direction"""
        # First iteration gets 100% of shortest path
        if self.iter == 1:
            self.stepsize = 1.0
            return True

        def derivative_of_objective(stepsize):
            x = self.fw_class_flow + stepsize * (
                self.aon_class_flow - self.fw_class_flow
            )  # fw_class_flow was calculated on last iteration
            congested_value = self.vdf.apply_vdf(
                "BPR", link_flows=x, capacity=self.graph.capacity, fftime=self.graph.free_flow_time
            )
            return np.sum(congested_value * (self.aon_class_flow - self.fw_class_flow))

        min_res = root_scalar(derivative_of_objective, bracket=(0, 1))
        # print(min_res)
        self.stepsize = min_res.root
        assert 0 <= self.stepsize <= 1.0
        if not min_res.converged:
            logger.warn("Frank Wolfe stepsize finder is not converged")
            return False
        return True

    def check_convergence(self):
        """Calculate relative gap and return True if it is smaller than desired precision"""
        aon_cost = np.sum(self.congested_time * self.aon_class_flow)
        current_cost = np.sum(self.congested_time * self.fw_class_flow)
        self.rgap = abs(current_cost - aon_cost) / current_cost
        # print("Iter {}: rgap = {}, stepsize = {}".format(self.iter, self.rgap, self.stepsize))
        if self.rgap_target >= self.rgap:
            return True
        return False
