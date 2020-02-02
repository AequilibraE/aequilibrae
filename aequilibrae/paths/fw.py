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
        self.steps_below_needed_to_terminate = 2

    def execute(self):
        logger.info("Frank-Wolfe Assignment STATS")
        logger.info("Iteration,RelativeGap,Fran-WolfeStep")
        for self.iter in range(1, self.max_iter + 1):
            aon = allOrNothing(self.matrix, self.graph, self.aon_results)
            aon.execute()
            self.aon_class_flow = np.sum(self.aon_results.link_loads, axis=1)

            if self.iter == 1:
                self.final_results.link_loads[:, :] = self.aon_results.link_loads[:, :].copy()
            else:
                self.calculate_stepsize()
                # self.stepsize = 1. / float(self.iter)
                self.final_results.link_loads[:, :] = self.final_results.link_loads[:, :] * (1.0 - self.stepsize)
                self.final_results.link_loads[:, :] += self.aon_results.link_loads[:, :] * self.stepsize

            self.fw_class_flow = np.sum(self.final_results.link_loads, axis=1)

            self.congested_time = self.vdf.apply_vdf(
                "BPR", link_flows=self.fw_class_flow, capacity=self.graph.capacity, fftime=self.graph.free_flow_time
            )
            self.graph.cost = self.congested_time

            # Check convergence
            if self.check_convergence() and self.iter > 100:
                if self.steps_below >= self.steps_below_needed_to_terminate:
                    break
                else:
                    self.steps_below += 1

            self.aon_results.reset()
            logger.info("{},{},{}".format(self.iter, self.rgap, self.stepsize))

        if self.rgap > self.rgap_target:
            logger.error("Desired RGap of {} was NOT reached".format(self.rgap_target))
        logger.info("FW Assignment finished. {} iterations and {} final gap".format(self.iter, self.rgap))
        # print("FW Assignment finished. {} iterations and {} final gap".format(self.iter, self.rgap))

    def derivative_of_objective(self, stepsize):
        x = (1.0 - stepsize) * self.fw_class_flow + stepsize * self.aon_class_flow
        congested_value = self.vdf.apply_vdf(
            "BPR", link_flows=x, capacity=self.graph.capacity, fftime=self.graph.free_flow_time
        )
        return np.sum(congested_value * (self.aon_class_flow - self.fw_class_flow))

    def calculate_stepsize(self):
        """Calculate optimal stepsize in gradient direction"""
        # First iteration gets 100% of shortest path
        if self.iter == 1:
            self.stepsize = 1.0
            return True
        try:
            min_res = root_scalar(self.derivative_of_objective, bracket=[0, 1], method="brentq")
        except ValueError:
            # We see not strictly monotone functions in practice, scipy cannot deal with this
            print("function not convex, need to take either 0 or 1")
            f_0 = self.derivative_of_objective(0.0)
            f_1 = self.derivative_of_objective(1.0)
            print(f_0, f_1)
            if f_0 < f_1:
                # prevent from stalling by making stepsize slightly non-zero
                # TODO: this should depend on the iteration number
                self.stepsize = 1e-5
                return False
            else:
                # Do we actually want this in practice? We throw away everything so far
                # for new solution. I guess that's reasonable
                self.stepsize = 1.0
                return False

        self.stepsize = min_res.root
        assert 0 <= self.stepsize <= 1.0
        if not min_res.converged:
            logger.warn("Frank Wolfe stepsize finder is not converged")
            return False
        return True

    def check_convergence(self):
        """Calculate relative gap and return True if it is smaller than desired precision"""
        aon_class_flow = np.sum(self.aon_results.link_loads, axis=1)
        aon_cost = np.sum(self.congested_time * aon_class_flow)
        # aon_cost = np.sum(self.congested_time * self.aon_class_flow)
        current_cost = np.sum(self.congested_time * self.fw_class_flow)
        self.rgap = abs(current_cost - aon_cost) / current_cost
        print(
            "Iter {}: rgap = {}, stepsize = {}, {:.2f}, {:.2f}".format(
                self.iter, self.rgap, self.stepsize, current_cost, aon_cost
            )
        )
        if self.rgap_target >= self.rgap:
            return True
        return False
