import numpy as np
from scipy.optimize import root_scalar
from typing import List

from aequilibrae.paths.traffic_class import TrafficClass
from aequilibrae.paths.all_or_nothing import allOrNothing
from aequilibrae import Parameters
from aequilibrae import logger

if False:
    from aequilibrae.paths.traffic_assignment import TrafficAssignment


class BFW:
    def __init__(self, assig_spec, algorithm) -> None:
        self.algorithm = algorithm
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
        self.conjugate_stepsize = 0.0
        self.fw_class_flow = 0
        # rgap can be a bit wiggly, specifying how many times we need to be below target rgap is a quick way to
        # ensure a better result. We might want to demand that the solution is that many consecutive times below.
        self.steps_below_needed_to_terminate = 1
        self.steps_below = 0

        self.previous_step_direction = {}
        self.step_direction = {}
        # if this is one, we do not have a new direction and will get stuck. Make it 1.
        self.conjugate_direction_max = 0.99999

        # if FW stepsize is zero, we set it to the corresponding MSA stepsize and then need to not make
        # the step direction conjugate to the previous direction.
        self.do_fw_step = False
        self.do_conjugate_step = False

        # BFW specific stuff
        self.betas = np.array([1.0, 0.0, 0.0])

    def calculate_conjugate_stepsize(self):
        pars = {"link_flows": self.fw_total_flow, "capacity": self.capacity, "fftime": self.free_flow_time}
        vdf_der = self.vdf.apply_derivative(**{**pars, **self.vdf_parameters})

        prev_dir_minus_current_sol = {}
        aon_minus_current_sol = {}
        aon_minus_prev_dir = {}
        for c in self.traffic_classes:
            prev_dir_minus_current_sol[c] = self.step_direction[c][:, :] - c.results.link_loads[:, :]
            aon_minus_current_sol[c] = c._aon_results.link_loads[:, :] - c.results.link_loads[:, :]
            aon_minus_prev_dir[c] = c._aon_results.link_loads[:, :] - self.step_direction[c][:, :]

        # TODO: This should be a sum over all supernetwork links, it's not tested for multi-class yet
        # if we can assume that all links appear in the subnetworks, then this is correct, otherwise
        # this needs more work
        numerator = 0.0
        denominator = 0.0
        for c in self.traffic_classes:
            for cp in self.traffic_classes:
                numerator += prev_dir_minus_current_sol[c] * aon_minus_current_sol[cp]
                denominator += prev_dir_minus_current_sol[c] * aon_minus_prev_dir[cp]

        numerator = np.sum(numerator * vdf_der)
        denominator = np.sum(denominator * vdf_der)

        alpha = numerator / denominator
        if alpha < 0.0:
            self.stepdirection = 0.0
        elif alpha > self.conjugate_direction_max:
            self.stepdirection = self.conjugate_direction_max
        else:
            self.conjugate_stepsize = alpha

    def calculate_biconjugate_direction(self):
        pars = {"link_flows": self.fw_total_flow, "capacity": self.capacity, "fftime": self.free_flow_time}
        vdf_der = self.vdf.apply_derivative(**{**pars, **self.vdf_parameters})

        prevs_minus_cur = {}
        aon_minus_cur = {}
        pre_pre_minus_pre = {}
        previous_minus_cur = {}

        for c in self.traffic_classes:
            prevs_minus_cur[c] = (
                self.step_direction[c][:, :] * self.stepsize
                + self.previous_step_direction[c][:, :] * (1.0 - self.stepsize)
                - c.results.link_loads[:, :]
            )
            aon_minus_cur[c] = c._aon_results.link_loads[:, :] - c.results.link_loads[:, :]
            pre_pre_minus_pre[c] = self.previous_step_direction[c] - self.step_direction[c][:, :]
            previous_minus_cur[c] = self.step_direction[c][:, :] - c.results.link_loads[:, :]

        # TODO: This should be a sum over all supernetwork links, it's not tested for multi-class yet
        # if we can assume that all links appear in the subnetworks, then this is correct, otherwise
        # this needs more work
        mu_numerator = 0.0
        mu_denominator = 0.0
        nu_nom = 0.0
        nu_denom = 0.0
        for c in self.traffic_classes:
            for cp in self.traffic_classes:
                mu_numerator += prevs_minus_cur[c] * aon_minus_cur[cp]
                mu_denominator += prevs_minus_cur[c] * pre_pre_minus_pre[cp]
                nu_nom += previous_minus_cur[c] * aon_minus_cur[cp]
                nu_denom += previous_minus_cur[c] * previous_minus_cur[cp]

        mu_numerator = np.sum(mu_numerator * vdf_der)
        mu_denominator = np.sum(mu_denominator * vdf_der)
        if mu_denominator == 0.0:
            mu = 0.0
        else:
            mu = -mu_numerator / mu_denominator
            # logger.info("mu before max = {}".format(mu))
            mu = max(0.0, mu)

        nu_nom = np.sum(nu_nom * vdf_der)
        nu_denom = np.sum(nu_denom * vdf_der)
        if nu_denom == 0.0:
            nu = 0.0
        else:
            nu = -(nu_nom / nu_denom) + mu * self.stepsize / (1.0 - self.stepsize)
            # logger.info("nu before max = {}".format(nu))
            nu = max(0.0, nu)

        self.betas[0] = 1.0 / (1.0 + nu + mu)
        self.betas[1] = nu * self.betas[0]
        self.betas[2] = mu * self.betas[0]

    def calculate_step_direction(self):
        """Caculate step direction depending on the method."""
        # current load: c.results.link_loads[:, :]
        # aon load: c._aon_results.link_loads[:, :]
        sd_flows = []

        # 2nd iteration is a fw step. if the previous step replaced the aggregated
        # solution so far, we need to start anew.
        if (
            (self.iter == 2)
            or (self.stepsize == 1.0)
            or (self.do_fw_step)
            or (self.algorithm == "frank-wolfe")
            or (self.algorithm == "msa")
        ):
            logger.info("FW step")
            self.do_fw_step = False
            self.do_conjugate_step = True
            self.conjugate_stepsize = 0.0
            for c in self.traffic_classes:
                self.step_direction[c] = c._aon_results.link_loads[:, :]
                sd_flows.append(np.sum(self.step_direction[c], axis=1) * c.pce)
        # 3rd iteration is cfw. also, if we had to reset direction search we need a cfw step before bfw
        elif (self.iter == 3) or (self.do_conjugate_step) or (self.algorithm == "cfw"):
            self.do_conjugate_step = False
            self.calculate_conjugate_stepsize()
            logger.info("CFW step, conjugate stepsize = {}".format(self.conjugate_stepsize))
            for c in self.traffic_classes:
                self.previous_step_direction[c] = self.step_direction[c].copy()  # save for bfw
                self.step_direction[c] *= self.conjugate_stepsize
                self.step_direction[c] += c._aon_results.link_loads[:, :] * (1.0 - self.conjugate_stepsize)
                sd_flows.append(np.sum(self.step_direction[c], axis=1) * c.pce)
        # biconjugate
        else:
            self.calculate_biconjugate_direction()
            # deep copy because we overwrite step_direction but need it on next iteration
            previous_step_dir_temp_copy = {}
            for c in self.traffic_classes:
                previous_step_dir_temp_copy[c] = self.step_direction[c].copy()
                self.step_direction[c] = (
                    c._aon_results.link_loads[:, :] * self.betas[0]
                    + self.step_direction[c] * self.betas[1]
                    + self.previous_step_direction[c] * self.betas[2]
                )
                sd_flows.append(np.sum(self.step_direction[c], axis=1) * c.pce)

                self.previous_step_direction[c] = previous_step_dir_temp_copy[c]

        self.step_direction_flow = np.sum(sd_flows, axis=0)

    def execute(self):
        logger.info("{} Assignment STATS".format(self.algorithm))
        logger.info("Iteration, RelativeGap, stepsize")
        for self.iter in range(1, self.max_iter + 1):
            flows = []
            aon_flows = []

            for c in self.traffic_classes:
                aon = allOrNothing(c.matrix, c.graph, c._aon_results)
                aon.execute()
                aon_flows.append(np.sum(c._aon_results.link_loads, axis=1) * c.pce)
            self.aon_total_flow = np.sum(aon_flows, axis=0)

            if self.iter == 1:
                for c in self.traffic_classes:
                    c.results.link_loads[:, :] = c._aon_results.link_loads[:, :]
                    flows.append(np.sum(c.results.link_loads, axis=1) * c.pce)
            else:
                self.calculate_step_direction()
                self.calculate_stepsize()
                for c in self.traffic_classes:
                    c.results.link_loads[:, :] = c.results.link_loads[:, :] * (1.0 - self.stepsize)
                    c.results.link_loads[:, :] += self.step_direction[c] * self.stepsize
                    # We already get the total traffic class, in PCEs, corresponding to the total for the user classes
                    flows.append(np.sum(c.results.link_loads, axis=1) * c.pce)

            self.fw_total_flow = np.sum(flows, axis=0)

            # Check convergence
            # This needs ot be done with the current costs, and not the future ones
            if self.iter > 1:
                if self.check_convergence():
                    if self.steps_below >= self.steps_below_needed_to_terminate:
                        break
                    else:
                        self.steps_below += 1

            pars = {"link_flows": self.fw_total_flow, "capacity": self.capacity, "fftime": self.free_flow_time}
            self.congested_time = self.vdf.apply_vdf(**{**pars, **self.vdf_parameters})

            for c in self.traffic_classes:
                c.graph.cost = self.congested_time
                c._aon_results.reset()
            logger.info("{},{},{}".format(self.iter, self.rgap, self.stepsize))

        if self.rgap > self.rgap_target:
            logger.error("Desired RGap of {} was NOT reached".format(self.rgap_target))
        logger.info(
            "{} Assignment finished. {} iterations and {} final gap".format(self.algorithm, self.iter, self.rgap)
        )

    def calculate_stepsize(self):
        """Calculate optimal stepsize in descent direction"""
        if self.algorithm == "msa":
            self.stepsize = 1.0 / self.iter
            return

        def derivative_of_objective(stepsize):
            x = self.fw_total_flow + stepsize * (self.step_direction_flow - self.fw_total_flow)
            # fw_total_flow was calculated on last iteration
            pars = {"link_flows": x, "capacity": self.capacity, "fftime": self.free_flow_time}
            congested_value = self.vdf.apply_vdf(**{**pars, **self.vdf_parameters})
            return np.sum(congested_value * (self.step_direction_flow - self.fw_total_flow))

        try:
            min_res = root_scalar(derivative_of_objective, bracket=[0, 1])
            self.stepsize = min_res.root
            if not min_res.converged:
                logger.warn("Descent direction stepsize finder is not converged")
        except ValueError:
            # We can have iterations where the objective function is not *strictly* convex, but the scipy method cannot deal
            # with this. Stepsize is then either given by 1 or 0, depending on where the objective function is smaller.
            # However, using zero would mean the overall solution would not get updated, and therefore we assert the stepsize
            # in order to add a small fraction of the AoN. A heuristic value equal to the corresponding MSA step size
            # seems to work well in practice.
            heuristic_stepsize_at_zero = 1.0 / self.iter
            if derivative_of_objective(0.0) < derivative_of_objective(1.0):
                logger.warn("# Alert: Adding {} to stepsize to make it non-zero".format(heuristic_stepsize_at_zero))
                self.stepsize = heuristic_stepsize_at_zero
                # need to reset conjugate / bi-conjugate direction search
                self.do_fw_step = True
            else:
                # Do we want to keep some of the old solution, or just throw away everything?
                self.stepsize = 1.0

        assert 0 <= self.stepsize <= 1.0

    def check_convergence(self):
        """Calculate relative gap and return True if it is smaller than desired precision"""
        aon_cost = np.sum(self.congested_time * self.aon_total_flow)
        current_cost = np.sum(self.congested_time * self.fw_total_flow)
        self.rgap = abs(current_cost - aon_cost) / current_cost
        if self.rgap_target >= self.rgap:
            return True
        return False
