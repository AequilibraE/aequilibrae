import numpy as np
from scipy.optimize import root_scalar
from typing import List

from aequilibrae.paths.traffic_class import TrafficClass
from aequilibrae.paths.all_or_nothing import allOrNothing
from aequilibrae import Parameters
from aequilibrae import logger

if False:
    from aequilibrae.paths.traffic_assignment import TrafficAssignment


class FW:
    def __init__(self, assig_spec) -> None:
        parameters = Parameters().parameters["assignment"]["equilibrium"]
        self.rgap_target = parameters["rgap"]
        self.max_iter = parameters["maximum_iterations"]

        self.assig = assig_spec  # type: TrafficAssignment

        if None in [
            assig_spec.classes,
            assig_spec.vdf,
            assig_spec.capacity_field,
            assig_spec.time_field,
            assig_spec.vdf_parameters,
        ]:
            raise Exception("Parameters missing. Setting the algorithm is the last thing to do when assigning")

        self.traffic_classes = assig_spec.classes  # type: List[TrafficClass]
        self.num_classes = len(assig_spec.classes)

        self.cap_field = assig_spec.capacity_field
        self.time_field = assig_spec.time_field
        self.vdf = assig_spec.vdf

        self.capacity = self.traffic_classes[0].graph.graph[self.cap_field]
        self.free_flow_time = self.traffic_classes[0].graph.graph[self.time_field]

        self.vdf_parameters = {}
        for k, v in assig_spec.vdf_parameters.items():
            if isinstance(v, str):
                self.vdf_parameters[k] = assig_spec.classes[0].graph.graph[k]
            else:
                self.vdf_parameters[k] = v

        self.iter = 0
        self.rgap = np.inf
        self.stepsize = 1.0
        self.fw_class_flow = 0
        # rgap can be a bit wiggly, specifying how many times we need to be below target rgap is a quick way to
        # ensure a better result. We could also demand that the solution is that many consecutive times below,
        # but we are talking about small oscillations so not really necessary.
        self.steps_below = 0
        self.steps_below_needed_to_terminate = 1
        self.all_aons = []
        self.all_times = []

    def execute(self):
        logger.info("Frank-Wolfe Assignment STATS")
        logger.info("Iteration,RelativeGap,Frank-WolfeStep")
        for self.iter in range(1, self.max_iter + 1):
            flows = []
            aon_flows = []

            for c in self.traffic_classes:
                aon = allOrNothing(c.matrix, c.graph, c._aon_results)
                aon.execute()

                aon_flows.append(np.sum(c._aon_results.link_loads, axis=1) * c.pce)

            self.aon_total_flow = np.sum(aon_flows, axis=0)
            _ = self.calculate_stepsize()

            for c in self.traffic_classes:
                c.results.link_loads[:, :] = c.results.link_loads[:, :] * (1.0 - self.stepsize)
                c.results.link_loads[:, :] += c._aon_results.link_loads[:, :] * self.stepsize

                # We already get the total traffic class, in PCEs, corresponding to the total for the user classes
                flows.append(np.sum(c.results.link_loads, axis=1) * c.pce)

            self.fw_total_flow = np.sum(flows, axis=0)

            pars = {"link_flows": self.fw_total_flow, "capacity": self.capacity, "fftime": self.free_flow_time}

            self.congested_time = self.vdf.apply_vdf(**{**pars, **self.vdf_parameters})

            for c in self.traffic_classes:
                c.graph.cost = self.congested_time

            self.all_aons.append(self.aon_total_flow)
            self.all_times.append(self.congested_time)

            # Check convergence
            if self.check_convergence() and self.iter > 1:
                if self.steps_below >= self.steps_below_needed_to_terminate:
                    break
                else:
                    self.steps_below += 1

            for c in self.traffic_classes:
                c._aon_results.reset()
            logger.info("{},{},{}".format(self.iter, self.rgap, self.stepsize))

        if self.rgap > self.rgap_target:
            logger.error("Desired RGap of {} was NOT reached".format(self.rgap_target))
        logger.info("FW Assignment finished. {} iterations and {} final gap".format(self.iter, self.rgap))

    def calculate_stepsize(self):
        # First iteration gets 100% of shortest path
        if self.iter == 1:
            self.stepsize = 1.0
            return True

        """Calculate optimal stepsize in gradient direction"""
        def derivative_of_objective(stepsize):
            x = self.fw_total_flow + stepsize * (self.aon_total_flow - self.fw_total_flow)
            # fw_total_flow was calculated on last iteration
            pars = {'link_flows': x, 'capacity': self.capacity, 'fftime': self.free_flow_time}
            congested_value = self.vdf.apply_vdf(**{**pars, **self.vdf_parameters})
            return np.sum(congested_value * (self.aon_total_flow - self.fw_total_flow))

        min_res = root_scalar(derivative_of_objective, bracket=(0, 1))
        self.stepsize = min_res.root
        assert 0 <= self.stepsize <= 1.0
        if not min_res.converged:
            logger.warn("Frank Wolfe stepsize finder is not converged")
            return False
        return True

    def check_convergence(self):
        """Calculate relative gap and return True if it is smaller than desired precision"""
        aon_cost = np.sum(self.congested_time * self.aon_total_flow)
        current_cost = np.sum(self.congested_time * self.fw_total_flow)
        self.rgap = abs(current_cost - aon_cost) / current_cost
        if self.rgap_target >= self.rgap:
            return True
        return False