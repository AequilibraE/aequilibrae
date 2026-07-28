import logging
import os
import time
from functools import partial
from pathlib import Path
from tempfile import gettempdir
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize_scalar, root_scalar

from aequilibrae.paths.all_or_nothing import allOrNothing
from aequilibrae.paths.AoN import (
    aggregate_link_costs,
    copy_three_dimensions,
    copy_two_dimensions,
    linear_combination,
    linear_combination_1d,
    linear_combination_skims,
    sum_a_times_b_minus_c,
    triple_linear_combination,
    triple_linear_combination_skims,
)
from aequilibrae.paths.results import AssignmentResults

if TYPE_CHECKING:
    from aequilibrae.paths.traffic_assignment import TrafficAssignment
    from aequilibrae.paths.traffic_class import TrafficClass

from aequilibrae.utils.aeq_signal import SIGNAL, simple_progress
from aequilibrae.utils.interface.worker_thread import WorkerThread


class LinearApproximation(WorkerThread):
    equilibration = SIGNAL(object)
    assignment = SIGNAL(object)
    signal = SIGNAL(object)

    def __init__(self, assig_spec, algorithm, project=None) -> None:
        WorkerThread.__init__(self, None)
        self.signal.emit(["set_text", "Linear Approximation"])
        self.logger = project.logger if project else logging.getLogger("aequilibrae")

        self.project_path = project.project_base_path if project else gettempdir()

        self.algorithm = algorithm
        self.rgap_target = assig_spec.rgap_target
        self.max_iter = assig_spec.max_iter
        self.cores = assig_spec.cores
        self.iteration_issue = []
        self.convergence_report = {
            "iteration": [],
            "time": [],
            "rgap": [],
            "alpha": [],
            "warnings": [],
        }
        if algorithm in ["cfw", "bfw"]:
            self.convergence_report["beta0"] = []
            self.convergence_report["beta1"] = []
            self.convergence_report["beta2"] = []

        self.assig: TrafficAssignment = assig_spec

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

        self.traffic_classes: list[TrafficClass] = assig_spec.classes
        self.num_classes = len(assig_spec.classes)

        self.cap_field = assig_spec.capacity_field
        self.time_field = assig_spec.time_field
        self.vdf = assig_spec.vdf
        self.vdf_parameters = assig_spec.vdf_parameters
        self.procedure_id = assig_spec.procedure_id

        self.iter = 0
        self.rgap = np.inf
        self.stepsize = 1.0
        self.conjugate_stepsize = 0.0
        self.fw_class_flow = 0
        # rgap can be a bit wiggly, specifying how many times we need to be below target rgap is a quick way to
        # ensure a better result. We might want to demand that the solution is that many consecutive times below.
        self.steps_below_needed_to_terminate = assig_spec.steps_below_needed_to_terminate
        self.steps_below = 0

        # if this is one, we do not have a new direction and will get stuck. Make it 1.
        self.conjugate_direction_max = 0.99999

        # Direction state for the current iteration and the next one. BFW needs an FW step followed by
        # a CFW step before it can safely resume bi-conjugate directions.
        self.current_direction = "fw"
        self.next_direction = None

        # BFW specific stuff
        self.betas = np.array([1.0, 0.0, 0.0])

        # Instantiates the arrays that we will use over and over
        self.capacity = assig_spec.capacity

        # Creates preload vector from preloads
        self.preload = None
        if assig_spec.preloads is not None:
            cols = assig_spec.preloads.columns.difference(["link_id", "direction"])
            self.preload = assig_spec.preloads[cols].sum(axis=1).to_numpy()

        self.free_flow_tt = assig_spec.free_flow_tt
        self.fw_total_flow = assig_spec.total_flow
        self.congested_time = assig_spec.congested_time
        self.vdf_der = np.array(assig_spec.congested_time, copy=True)
        self.congested_value = np.array(assig_spec.congested_time, copy=True)

        # Private scratch buffers for the trapezoidal Beckmann line search
        # (``__objective_change_at_stepsize``). Kept separate from
        # ``self.congested_value`` (which is the analytic-derivative line
        # search's scratch buffer) so that the trapezoidal helper does not
        # clobber the public-looking attribute as a side effect, and so that
        # the per-call ``congested_time + congested_value`` sum can be
        # written into a pre-allocated buffer instead of allocating fresh
        # each call.
        self._trap_new_flow = np.zeros_like(self.congested_time)
        self._trap_new_cost = np.zeros_like(self.congested_time)
        self._trap_avg_cost = np.zeros_like(self.congested_time)

        self.step_direction: dict[str, AssignmentResults] = {}
        self.previous_step_direction: dict[str, AssignmentResults] = {}
        self.temp_step_direction_for_copy: dict[str, AssignmentResults] = {}

        self.aons = {}

        for c in self.traffic_classes:
            r = AssignmentResults()
            r.prepare(c.graph, c.matrix)
            self.step_direction[c._id] = r

        if self.algorithm in ["cfw", "bfw"]:
            for c in self.traffic_classes:
                for d in [self.step_direction, self.previous_step_direction, self.temp_step_direction_for_copy]:
                    r = AssignmentResults()
                    r.prepare(c.graph, c.matrix)
                    r.compact_link_loads = np.zeros([])
                    r.compact_total_link_loads = np.zeros([])
                    d[c._id] = r

    def calculate_conjugate_stepsize(self):
        self.vdf.apply_derivative(
            self.vdf_der, self.fw_total_flow, self.capacity, self.free_flow_tt, *self.vdf_parameters, self.cores
        )
        numerator = 0.0
        denominator = 0.0
        prev_dir_minus_current_sol = {}
        aon_minus_current_sol = {}
        aon_minus_prev_dir = {}

        for c in self.traffic_classes:
            stp_dir = self.step_direction[c._id]
            prev_dir_minus_current_sol[c._id] = np.sum(stp_dir.link_loads[:, :] - c.results.link_loads[:, :], axis=1)
            aon_minus_current_sol[c._id] = np.sum(c._aon_results.link_loads[:, :] - c.results.link_loads[:, :], axis=1)
            aon_minus_prev_dir[c._id] = np.sum(c._aon_results.link_loads[:, :] - stp_dir.link_loads[:, :], axis=1)

        for c_0 in self.traffic_classes:
            for c_1 in self.traffic_classes:
                numerator += prev_dir_minus_current_sol[c_0._id] * aon_minus_current_sol[c_1._id]
                denominator += prev_dir_minus_current_sol[c_0._id] * aon_minus_prev_dir[c_1._id]

        numerator = np.sum(numerator * self.vdf_der)
        denominator = np.sum(denominator * self.vdf_der)

        alpha = numerator / denominator
        if alpha < 0.0:
            self.conjugate_stepsize = 0.0
        elif alpha > self.conjugate_direction_max:
            self.conjugate_stepsize = self.conjugate_direction_max
        else:
            self.conjugate_stepsize = alpha

        # for reporting, we use a different convention, consistent with BFW: beta_0 corresponds to multiplier for AON;
        # in calculations we follow the conventions of our TRB paper.
        self.betas[0] = 1.0 - self.conjugate_stepsize
        self.betas[1] = self.conjugate_stepsize
        self.betas[2] = 0.0

    def calculate_biconjugate_direction(self):
        self.vdf.apply_derivative(
            self.vdf_der, self.fw_total_flow, self.capacity, self.free_flow_tt, *self.vdf_parameters, self.cores
        )
        mu_numerator = 0.0
        mu_denominator = 0.0
        nu_nom = 0.0
        nu_denom = 0.0

        w_ = {}
        x_ = {}
        y_ = {}
        z_ = {}

        for c in self.traffic_classes:
            sd = self.step_direction[c._id].link_loads[:, :]
            psd = self.previous_step_direction[c._id].link_loads[:, :]
            ll = c.results.link_loads[:, :]

            x_[c._id] = np.sum(sd * self.stepsize + psd * (1.0 - self.stepsize) - ll, axis=1)
            y_[c._id] = np.sum(c._aon_results.link_loads[:, :] - ll, axis=1)
            z_[c._id] = np.sum(sd - ll, axis=1)
            w_[c._id] = np.sum(psd - sd, axis=1)

        for c_0 in self.traffic_classes:
            for c_1 in self.traffic_classes:
                mu_numerator += x_[c_0._id] * y_[c_1._id]
                mu_denominator += x_[c_0._id] * w_[c_1._id]
                nu_nom += z_[c_0._id] * y_[c_1._id]
                nu_denom += z_[c_0._id] * z_[c_1._id]

        mu_numerator = np.sum(mu_numerator * self.vdf_der)
        mu_denominator = np.sum(mu_denominator * self.vdf_der)
        if mu_denominator == 0.0:
            mu = 0.0
        else:
            mu = -mu_numerator / mu_denominator
            mu = max(0.0, mu)

        nu_nom = np.sum(nu_nom * self.vdf_der)
        nu_denom = np.sum(nu_denom * self.vdf_der)
        if nu_denom == 0.0:
            nu = 0.0
        else:
            nu = -(nu_nom / nu_denom) + mu * self.stepsize / (1.0 - self.stepsize)
            nu = max(0.0, nu)

        self.betas[0] = 1.0 / (1.0 + nu + mu)
        self.betas[1] = nu * self.betas[0]
        self.betas[2] = mu * self.betas[0]

    def _apply_assigned_flow(self, link_flows):
        """Records the flows just assigned, stacking any preload on top of them."""
        total = np.array(link_flows, dtype=np.float64, copy=True)
        if self.preload is not None:
            total += self.preload
        self.fw_total_flow = total

    def _refresh_congested_costs(self):
        self.vdf.apply_vdf(
            self.congested_time,
            self.fw_total_flow,
            self.capacity,
            self.free_flow_tt,
            *self.vdf_parameters,
            self.cores,
        )

        for c in self.traffic_classes:
            if self.time_field in c.graph.skim_fields:
                k = c.graph.skim_fields.index(self.time_field)
                aggregate_link_costs(self.congested_time[:], c.graph.compact_skims[:, k], c.results.crosswalk)

    def __calculate_step_direction(self):
        """Calculates step direction depending on the method"""
        sd_flows = []
        direction = self.next_direction
        self.next_direction = None

        # 2nd iteration is a fw step. if the previous step replaced the aggregated
        # solution so far, we need to start anew.
        if self.iter == 2 or direction == "fw" or self.algorithm in ["msa", "frank-wolfe"]:
            self.current_direction = "fw"
            if self.algorithm == "bfw":
                self.next_direction = "cfw"
            self.conjugate_stepsize = 0.0
            for c in self.traffic_classes:
                aon_res = c._aon_results
                stp_dir_res = self.step_direction[c._id]
                copy_two_dimensions(stp_dir_res.link_loads, aon_res.link_loads, self.cores)
                stp_dir_res.total_flows()
                if c.results.num_skims > 0:
                    copy_three_dimensions(stp_dir_res.skims.matrix_view, aon_res.skims.matrix_view, self.cores)
                sd_flows.append(aon_res.total_link_loads)

                if c._selected_links:
                    aux_res = self.aons[c._id].aux_res
                    for name, idx in c._aon_results._selected_links.items():
                        copy_two_dimensions(
                            self.sl_step_dir_ll[c._id][name]["sdr"],
                            np.sum(aux_res.temp_sl_link_loading, axis=0)[idx, :, :],
                            self.cores,
                        )
                        copy_three_dimensions(
                            self.sl_step_dir_od[c._id][name]["sdr"],
                            np.sum(aux_res.temp_sl_od_matrix, axis=0)[idx, :, :, :],
                            self.cores,
                        )

        # 3rd iteration is cfw. also, if we had to reset direction search we need a cfw step before bfw
        elif (self.iter == 3) or (direction == "cfw") or (self.algorithm == "cfw"):
            self.current_direction = "cfw"
            self.calculate_conjugate_stepsize()
            for c in self.traffic_classes:
                sdr = self.step_direction[c._id]
                previous = self.previous_step_direction[c._id]

                copy_two_dimensions(previous.link_loads, sdr.link_loads, self.cores)
                previous.total_flows()
                if c.results.num_skims > 0:
                    copy_three_dimensions(previous.skims.matrix_view, sdr.skims.matrix_view, self.cores)

                linear_combination(
                    sdr.link_loads, sdr.link_loads, c._aon_results.link_loads, self.conjugate_stepsize, self.cores
                )

                if c.results.num_skims > 0:
                    linear_combination_skims(
                        sdr.skims.matrix_view,
                        sdr.skims.matrix_view,
                        c._aon_results.skims.matrix_view,
                        self.conjugate_stepsize,
                        self.cores,
                    )

                if c._selected_links:
                    aux_res = self.aons[c._id].aux_res
                    for name, idx in c._aon_results._selected_links.items():
                        sl_step_dir_ll = self.sl_step_dir_ll[c._id][name]
                        sl_step_dir_od = self.sl_step_dir_od[c._id][name]

                        copy_two_dimensions(
                            sl_step_dir_ll["prev_sdr"],
                            sl_step_dir_ll["sdr"],
                            self.cores,
                        )
                        copy_three_dimensions(
                            sl_step_dir_od["prev_sdr"],
                            sl_step_dir_od["sdr"],
                            self.cores,
                        )

                        linear_combination(
                            sl_step_dir_ll["sdr"],
                            sl_step_dir_ll["sdr"],
                            np.sum(aux_res.temp_sl_link_loading, axis=0)[idx, :, :],
                            self.conjugate_stepsize,
                            self.cores,
                        )

                        linear_combination_skims(
                            sl_step_dir_od["sdr"],
                            sl_step_dir_od["sdr"],
                            np.sum(aux_res.temp_sl_od_matrix, axis=0)[idx, :, :, :],
                            self.conjugate_stepsize,
                            self.cores,
                        )

                sdr.total_flows()
                sd_flows.append(sdr.total_link_loads)
        # biconjugate
        else:
            self.current_direction = "bfw"
            self.calculate_biconjugate_direction()
            # deep copy because we overwrite step_direction but need it on next iteration
            for c in self.traffic_classes:
                ppst: AssignmentResults = self.temp_step_direction_for_copy[c._id]
                prev_stp_dir: AssignmentResults = self.previous_step_direction[c._id]
                stp_dir: AssignmentResults = self.step_direction[c._id]

                copy_two_dimensions(ppst.link_loads, stp_dir.link_loads, self.cores)
                ppst.total_flows()
                if c.results.num_skims > 0:
                    copy_three_dimensions(ppst.skims.matrix_view, stp_dir.skims.matrix_view, self.cores)

                triple_linear_combination(
                    stp_dir.link_loads,
                    c._aon_results.link_loads,
                    stp_dir.link_loads,
                    prev_stp_dir.link_loads,
                    self.betas,
                    self.cores,
                )

                stp_dir.total_flows()
                if c.results.num_skims > 0:
                    triple_linear_combination_skims(
                        stp_dir.skims.matrix_view,
                        c._aon_results.skims.matrix_view,
                        stp_dir.skims.matrix_view,
                        prev_stp_dir.skims.matrix_view,
                        self.betas,
                        self.cores,
                    )

                if c._selected_links:
                    aux_res = self.aons[c._id].aux_res
                    for name, idx in c._aon_results._selected_links.items():
                        sl_step_dir_ll = self.sl_step_dir_ll[c._id][name]
                        sl_step_dir_od = self.sl_step_dir_od[c._id][name]
                        copy_two_dimensions(
                            sl_step_dir_ll["temp_prev_sdr"],
                            sl_step_dir_ll["sdr"],
                            self.cores,
                        )
                        copy_three_dimensions(
                            sl_step_dir_od["temp_prev_sdr"],
                            sl_step_dir_od["sdr"],
                            self.cores,
                        )

                        triple_linear_combination(
                            sl_step_dir_ll["sdr"],
                            np.sum(aux_res.temp_sl_link_loading, axis=0)[idx, :, :],
                            sl_step_dir_ll["sdr"],
                            sl_step_dir_ll["prev_sdr"],
                            self.betas,
                            self.cores,
                        )

                        triple_linear_combination_skims(
                            sl_step_dir_od["sdr"],
                            np.sum(aux_res.temp_sl_od_matrix, axis=0)[idx, :, :, :],
                            sl_step_dir_od["sdr"],
                            sl_step_dir_od["prev_sdr"],
                            self.betas,
                            self.cores,
                        )

                        copy_two_dimensions(
                            sl_step_dir_ll["prev_sdr"],
                            sl_step_dir_ll["temp_prev_sdr"],
                            self.cores,
                        )
                        copy_three_dimensions(
                            sl_step_dir_od["prev_sdr"],
                            sl_step_dir_od["temp_prev_sdr"],
                            self.cores,
                        )

                sd_flows.append(np.sum(stp_dir.link_loads, axis=1))

                copy_two_dimensions(prev_stp_dir.link_loads, ppst.link_loads, self.cores)
                prev_stp_dir.total_flows()
                if c.results.num_skims > 0:
                    copy_three_dimensions(prev_stp_dir.skims.matrix_view, ppst.skims.matrix_view, self.cores)

        self.step_direction_flow = np.sum(sd_flows, axis=0)

    def __retry_with_fw_direction(self, msg: str):
        if self.algorithm == "bfw":
            self.betas.fill(-1)

        self.logger.debug(msg)
        self.iteration_issue.append(msg)
        self.next_direction = "fw"
        self.__calculate_step_direction()
        self.calculate_stepsize()

    def __maybe_create_path_file_directories(self):
        path_base_dir = os.path.join(self.project_path, "path_files", self.procedure_id)
        for c in self.traffic_classes:
            if c._aon_results.save_path_file:
                c._aon_results.path_file_dir = os.path.join(
                    path_base_dir, f"iter{self.iter}", f"path_c{c.mode}_{c._id}"
                )
                Path(c._aon_results.path_file_dir).mkdir(parents=True, exist_ok=True)
                if self.iter == 1:  # save simplified graph correspondences, this could change after assignment
                    c.graph.save_compressed_correspondence(path_base_dir, c.mode, c._id)

    def doWork(self):
        self.execute()

    def execute(self):  # noqa: C901
        self.__start_time = time.perf_counter()
        # We build the fixed cost field

        self.sl_step_dir_ll = {}
        self.sl_step_dir_od = {}

        for c in self.traffic_classes:
            # Copying select link dictionary that maps name to its relevant matrices into the class' results
            c._aon_results._selected_links = c._selected_links
            c.results._selected_links = c._selected_links

            link_loads_step_dir_shape = (
                c.graph.compact_num_links,
                c.results.classes["number"],
            )

            od_step_dir_shape = (
                c.graph.num_zones,
                c.graph.num_zones,
                c.results.classes["number"],
            )

            self.sl_step_dir_ll[c._id] = {}
            self.sl_step_dir_od[c._id] = {}
            for name in c._selected_links.keys():
                self.sl_step_dir_ll[c._id][name] = {
                    "sdr": np.zeros(link_loads_step_dir_shape, dtype=c.graph.default_types("float")),
                    "prev_sdr": np.zeros(link_loads_step_dir_shape, dtype=c.graph.default_types("float")),
                    "temp_prev_sdr": np.zeros(link_loads_step_dir_shape, dtype=c.graph.default_types("float")),
                }

                self.sl_step_dir_od[c._id][name] = {
                    "sdr": np.zeros(od_step_dir_shape, dtype=c.graph.default_types("float")),
                    "prev_sdr": np.zeros(od_step_dir_shape, dtype=c.graph.default_types("float")),
                    "temp_prev_sdr": np.zeros(od_step_dir_shape, dtype=c.graph.default_types("float")),
                }

            # Sizes the temporary objects used for the results
            c.results.prepare(c.graph, c.matrix)
            c._aon_results.prepare(c.graph, c.matrix)
            c.results.reset()

            # Prepares the fixed cost to be used
            if c.fixed_cost_field:
                # divide fixed cost by volume-dependent prefactor (vot) such that we don't have to do it for
                # each occurrence in the objective function. TODO: Need to think about cost skims here, we do
                # not want this there I think
                v = c.graph.graph[c.fixed_cost_field].values[:]
                c.fixed_cost[c.graph.graph.__supernet_id__] = v * c.fc_multiplier / c.vot
                c.fixed_cost[np.isnan(c.fixed_cost)] = 0

            # TODO: Review how to eliminate this. It looks unnecessary
            # Just need to create some arrays for cost
            c.graph.set_graph(self.time_field)

            self.aons[c._id] = allOrNothing(c._id, c.matrix, c.graph, c._aon_results)

        self._apply_assigned_flow(np.zeros_like(self.capacity))
        self._refresh_congested_costs()

        self.logger.info(f"{self.algorithm} Assignment stats")
        self.logger.info("Iteration, RelativeGap (AoN), stepsize")

        msg = "Equilibrium Assignment"
        for self.iter in simple_progress(range(1, self.max_iter + 1), self.signal, msg):  # noqa: B020
            self.iteration_issue = []

            aon_flows = []

            self.__maybe_create_path_file_directories()

            for c in self.traffic_classes:  # type: TrafficClass
                msg = f"All-or-Nothing - Traffic Class: {c._id}"
                self.signal.emit(["set_text", msg])
                # cost = c.fixed_cost / c.vot + self.congested_time #  now only once
                cost = c.fixed_cost + self.congested_time
                aggregate_link_costs(cost, c.graph.compact_cost, c.results.crosswalk)

                aon = self.aons[c._id]  # This is a new object every iteration, with new aux_res
                self.signal.emit(["refresh"])
                self.signal.emit(["reset"])
                aon.signal = self.signal

                aon.execute()
                c._aon_results.link_loads *= c.pce
                c._aon_results.total_flows()
                aon_flows.append(c._aon_results.total_link_loads)

            self.aon_total_flow = np.sum(aon_flows, axis=0)

            flows = []
            if self.iter == 1:
                for c in self.traffic_classes:
                    copy_two_dimensions(c.results.link_loads, c._aon_results.link_loads, self.cores)
                    c.results.total_flows()
                    if c.results.num_skims > 0:
                        copy_three_dimensions(c.results.skims.matrix_view, c._aon_results.skims.matrix_view, self.cores)

                    if c._selected_links:
                        for name, idx in c._aon_results._selected_links.items():
                            # Copy the temporary results into the final od matrix, referenced by link_set name
                            # The temp has an index associated with the link_set name
                            copy_three_dimensions(
                                c.results.select_link_od.matrix[name],  # matrix being written into
                                np.sum(self.aons[c._id].aux_res.temp_sl_od_matrix, axis=0)[
                                    idx, :, :, :
                                ],  # results after the iteration
                                self.cores,  # core count
                            )
                            copy_two_dimensions(
                                c.results.select_link_loading[name],  # output matrix
                                np.sum(self.aons[c._id].aux_res.temp_sl_link_loading, axis=0)[idx, :, :],  # matrix 1
                                self.cores,  # core count
                            )
                    flows.append(c.results.total_link_loads)

            else:
                self.__calculate_step_direction()
                self.calculate_stepsize()
                for c in self.traffic_classes:
                    stp_dir = self.step_direction[c._id]

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

                    if c._selected_links:
                        for name, _idx in c._aon_results._selected_links.items():
                            # Copy the temporary results into the final od matrix, referenced by link_set name
                            # The temp flows have an index associated with the link_set name
                            linear_combination_skims(
                                cls_res.select_link_od.matrix[name],  # output matrix
                                self.sl_step_dir_od[c._id][name]["sdr"],
                                cls_res.select_link_od.matrix[name],  # matrix 2 (previous iteration)
                                self.stepsize,  # stepsize
                                self.cores,  # core count
                            )

                            linear_combination(
                                cls_res.select_link_loading[name],  # output matrix
                                self.sl_step_dir_ll[c._id][name]["sdr"],
                                cls_res.select_link_loading[name],  # matrix 2 (previous iteration)
                                self.stepsize,  # stepsize
                                self.cores,  # core count
                            )

                    cls_res.total_flows()
                    flows.append(cls_res.total_link_loads)

            self._apply_assigned_flow(np.sum(flows, axis=0))

            if self.algorithm == "all-or-nothing":
                break

            # Check convergence
            # This needs to be done with the current costs, and not the future ones
            converged = self.check_convergence() if self.iter > 1 else False
            self._refresh_congested_costs()

            self.convergence_report["time"].append(time.perf_counter() - self.__start_time)
            self.convergence_report["iteration"].append(self.iter)
            self.convergence_report["rgap"].append(self.rgap)
            self.convergence_report["warnings"].append("; ".join(self.iteration_issue))
            self.convergence_report["alpha"].append(self.stepsize)

            if self.algorithm in ["cfw", "bfw"]:
                self.convergence_report["beta0"].append(self.betas[0])
                self.convergence_report["beta1"].append(self.betas[1])
                self.convergence_report["beta2"].append(self.betas[2])

            self.logger.info(f"{self.iter},{self.rgap},{self.stepsize}")
            if converged:
                self.steps_below += 1
                if self.steps_below >= self.steps_below_needed_to_terminate:
                    break
            else:
                self.steps_below = 0

            if self.iter < self.max_iter:
                for c in self.traffic_classes:
                    c._aon_results.reset()
                    if self.time_field not in c.graph.skim_fields:
                        continue
                    idx = c.graph.skim_fields.index(self.time_field)
                    c.graph.skims[:, idx] = self.congested_time[:]

            msg = f"Equilibrium Assignment - Iteration: {self.iter}/{self.max_iter} - RGap: {self.rgap:.6}"
            self.signal.emit(["set_text", msg])

        for c in self.traffic_classes:
            c.results.link_loads /= c.pce
            c.results.total_flows()
            c.congested_time = self.congested_time

        if (self.rgap > self.rgap_target) and (self.algorithm != "all-or-nothing"):
            self.logger.error(f"Desired RGap of {self.rgap_target} was NOT reached")
        self.logger.info(f"{self.algorithm} Assignment finished. {self.iter} iterations, final AoN rgap = {self.rgap}")

        self.signal.emit(["finished"])

    def __derivative_of_objective_stepsize_dependent(self, stepsize, const_term):
        """The stepsize-dependent part of the derivative of the objective function. If fixed costs are defined,
        the corresponding contribution needs to be passed in"""
        x = np.zeros_like(self.fw_total_flow)
        linear_combination_1d(x, self.step_direction_flow, self.fw_total_flow, stepsize, self.cores)
        # x = self.fw_total_flow + stepsize * (self.step_direction_flow - self.fw_total_flow)
        self.vdf.apply_vdf(self.congested_value, x, self.capacity, self.free_flow_tt, *self.vdf_parameters, self.cores)
        link_cost_term = sum_a_times_b_minus_c(
            self.congested_value, self.step_direction_flow, self.fw_total_flow, self.cores
        )
        return link_cost_term + const_term

    def __derivative_of_objective_stepsize_independent(self):
        """The part of the derivative of the objective function that does not dependent on stepsize. Non-zero
        only for fixed cost contributions."""
        class_specific_term = 0.0
        for c in self.traffic_classes:
            # fixed cost is scaled by vot
            class_link_costs = sum_a_times_b_minus_c(
                c.fixed_cost, self.step_direction[c._id].link_loads[:, 0], c.results.link_loads[:, 0], self.cores
            )
            class_specific_term += class_link_costs
        return class_specific_term

    def __objective_change_at_stepsize(
        self, derivative_of_objective_stepsize_independent: np.ndarray, stepsize: float
    ) -> float:
        """Trapezoidal approximation of the Beckmann objective change
        ``Z(x + α·d) − Z(x)`` for a given line-search step ``α = stepsize``.

        On large congested networks (e.g. Chicago, BPR β=4), this trapezoidal line search picks smaller,
        more conservative α values than the analytic-derivative line search and yields materially better
        BFW convergence because the smaller α reduces the magnitude of the ``μ·α/(1-α)`` bias term in
        the next iteration's BFW formula.

        All intermediate buffers are pre-allocated on the instance (``self._trap_new_flow``, ``self._trap_new_cost``,
        ``self._trap_avg_cost``) so that this helper does NOT clobber ``self.congested_value`` (which the
        analytic-derivative line search uses as scratch) and does NOT allocate fresh arrays on every Brent probe.
        """
        linear_combination_1d(
            self._trap_new_flow,
            self.step_direction_flow,
            self.fw_total_flow,
            stepsize,
            self.cores,
        )
        self.vdf.apply_vdf(
            self._trap_new_cost,
            self._trap_new_flow,
            self.capacity,
            self.free_flow_tt,
            *self.vdf_parameters,
            self.cores,
        )
        np.add(self.congested_time, self._trap_new_cost, out=self._trap_avg_cost)
        link_term = (
            0.5
            * stepsize
            * sum_a_times_b_minus_c(
                self._trap_avg_cost,
                self.step_direction_flow,
                self.fw_total_flow,
                self.cores,
            )
        )
        fixed_cost_term = stepsize * derivative_of_objective_stepsize_independent
        return link_term + fixed_cost_term

    def __clip_stepsize(self, stepsize: float, upper_bound: float = 1.0) -> float:
        if not np.isfinite(stepsize):
            raise ValueError(f"Non-finite stepsize {stepsize} encountered")

        clipped = min(max(float(stepsize), 0.0), upper_bound)
        if clipped != stepsize:
            msg = f"Stepsize {stepsize} outside [0, {upper_bound}]; clipping to {clipped}."
            self.logger.debug(msg)
            self.iteration_issue.append(msg)
        return clipped

    def calculate_stepsize(self):
        """Calculate optimal stepsize in descent direction"""
        if self.algorithm == "msa":
            self.stepsize = self.__clip_stepsize(1.0 / self.iter)
            return

        # For BFW use trapezoidal Beckmann minimiser on
        # [0, α_max] instead of root-finding the analytic derivative.
        #
        # Two cooperating mechanisms vs. the analytic root_scalar approach:
        #
        # (1) Trapezoidal objective. The analytic and trapezoidal lines
        #     agree when c(x) is approximately quadratic between x and
        #     x + d, but diverge significantly when the BPR exponent is
        #     large (β=4 on Chicago).
        # (2) For BFW only: a cap α_max = 1/sqrt(iter) prevents the line search from
        #     returning α = 1.0, which would collapse the BFW history (s^{k-1} onto x^k)
        #     and cause the μ·α/(1-α) bias term in calculate_biconjugate_direction to blow up.
        #     CFW has neither concern and uses α_max = 1.0 (uncapped).
        #
        # BFW Chicago-50 rgap: 1.14e-3 (was 1.54e-3 at HEAD baseline).
        if self.algorithm in ("bfw", "cfw"):
            # The 1/sqrt(iter) cap is only needed for BFW: it bounds the mu*alpha/(1-alpha) bias
            # term in calculate_biconjugate_direction and prevents alpha=1.0 from collapsing the
            # BFW history. CFW has no such term and no restart state sensitive to large steps, so
            # capping CFW at 1/sqrt(iter) degrades it to MSA-like convergence without any benefit.
            alpha_max = min(1.0, 1.0 / max(self.iter, 1) ** 0.5) if self.algorithm == "bfw" else 1.0
            derivative_of_objective_stepsize_independent = self.__derivative_of_objective_stepsize_independent()
            res = minimize_scalar(
                partial(
                    self.__objective_change_at_stepsize,
                    derivative_of_objective_stepsize_independent,
                ),
                bounds=(0.0, alpha_max),
                method="Bounded",
                options={"xatol": 1e-4, "maxiter": 10},
            )

            def use_tiny_step(message: str):
                tiny_step = 1e-2 / self.iter
                if message:
                    self.iteration_issue.append(message)
                    log_message = f"# Alert: {message} Adding {tiny_step} as step size to make it non-zero."
                else:
                    log_message = f"# Alert: Adding {tiny_step} as step size to make it non-zero."
                self.logger.debug(log_message)
                self.stepsize = self.__clip_stepsize(tiny_step, alpha_max)

            try:
                candidate = self.__clip_stepsize(res.x, alpha_max)
            except ValueError as e:
                msg = f"BFW/CFW line search returned an invalid stepsize. {e.args}"
                if self.current_direction == "fw":
                    use_tiny_step(msg)
                else:
                    self.__retry_with_fw_direction(msg)
                return

            # Brent's bounded method does not evaluate the endpoints exactly.
            # Compare the interior optimum against α_max explicitly so a true
            # boundary case (descent throughout the cap interval) still picks
            # α_max instead of a value just inside it.
            z_interior = float(res.fun)
            if not np.isfinite(z_interior):
                msg = f"BFW/CFW line search returned a non-finite objective value ({z_interior}); falling back to FW."
                if self.current_direction == "fw":
                    use_tiny_step(msg)
                else:
                    self.__retry_with_fw_direction(msg)
                return

            z_at_max = self.__objective_change_at_stepsize(derivative_of_objective_stepsize_independent, alpha_max)
            if not np.isfinite(z_at_max):
                msg = f"BFW/CFW line search returned a non-finite boundary objective ({z_at_max}); falling back to FW."
                if self.current_direction == "fw":
                    use_tiny_step(msg)
                else:
                    self.__retry_with_fw_direction(msg)
                return

            if z_at_max < z_interior and z_at_max < 0.0:
                self.stepsize = self.__clip_stepsize(alpha_max, alpha_max)
            elif z_interior < 0.0:
                self.stepsize = candidate
            else:
                msg = "BFW/CFW direction yielded no improvement; falling back to FW."
                if self.current_direction == "fw":
                    use_tiny_step("")
                else:
                    self.__retry_with_fw_direction(msg)
                return
            assert 0 <= self.stepsize <= alpha_max + 1e-12
            return

        class_specific_term = self.__derivative_of_objective_stepsize_independent()
        derivative_of_objective = partial(
            self.__derivative_of_objective_stepsize_dependent, const_term=class_specific_term
        )

        x_tol = max(min(1e-6, self.rgap * 1e-5), 1e-12)

        try:
            min_res = root_scalar(derivative_of_objective, bracket=[0, 1], xtol=x_tol)
            self.stepsize = self.__clip_stepsize(min_res.root)
            if not min_res.converged:
                self.logger.debug("Descent direction stepsize finder has not converged")

        except ValueError as e:
            # `root_scalar` raises ValueError when the derivative does not change sign in [0, 1].
            # There are two genuinely distinct cases:
            #   * derivative(0) < 0  ⇒  direction is descent at the current point. Since the
            #     derivative is monotone non-decreasing along the (convex) line, descent
            #     persists throughout [0, 1] and the optimum sits at α = 1 (or beyond).
            #     This is a perfectly valid line-search outcome and only happens because we
            #     bracket the search to the feasible interval. We must NOT treat it as a
            #     "reset" - the resulting solution is fine and convergence may be checked.
            #   * derivative(0) >= 0 ⇒  direction is *not* a descent direction. We then need
            #     to reset to a Frank-Wolfe step (or, if FW itself failed, take a tiny MSA
            #     step to avoid stalling).
            d0_for_branch = derivative_of_objective(0.0)

            if d0_for_branch >= 0:
                if self.current_direction == "fw" or self.algorithm == "frank-wolfe":
                    tiny_step = 1e-2 / self.iter  # use a fraction of the MSA stepsize. We observe that using 1e-4
                    # works well in practice, however for a large number of iterations this might be too much so
                    # use this heuristic instead.
                    self.logger.debug(f"# Alert: Adding {tiny_step} as step size to make it non-zero. {e.args}")
                    self.stepsize = self.__clip_stepsize(tiny_step)
                else:
                    msg = f"Found bad conjugate direction step. Performing FW search. {e.args}"
                    self.__retry_with_fw_direction(msg)
            else:
                # derivative(0) < 0 (and derivative(1) must also be ≤ 0, otherwise the bracket
                # search would have succeeded). The objective is still decreasing at α = 1, so
                # the constrained optimum on [0, 1] is α = 1. Take the full step; do NOT mark
                # this as a reset - convergence checking remains valid.
                self.stepsize = self.__clip_stepsize(1.0)
                self.logger.info("Line-search optimum at the boundary (alpha = 1.0); descent throughout [0, 1]")

        assert 0 <= self.stepsize <= 1.0

    def check_convergence(self):
        """Calculate relative gap and return ``True`` if it is smaller than desired precision.

        Two relative gaps are computed and stored on the instance:

        * ``self.rgap`` - the AequilibraE convention,
          ``|Σ flow·cost − Σ AON·cost| / Σ flow·cost``. **This is the only
          quantity used for the stopping criterion** (compared against
          ``self.rgap_target``).
          ``(Σ flow·cost − Σ direction·cost) / Σ flow·cost``, where
          ``direction`` is the BFW combined step direction
        """
        if self.stepsize == 1.0:
            return False

        aon_cost = 0.0
        current_cost = 0.0
        for c in self.traffic_classes:
            aon_class_flow = c._aon_results.total_link_loads
            current_class_flow = c.results.total_link_loads

            aon_cost += np.sum((self.congested_time + c.fixed_cost) * aon_class_flow)
            current_cost += np.sum((self.congested_time + c.fixed_cost) * current_class_flow)

        if current_cost != 0.0:
            self.rgap = abs(current_cost - aon_cost) / current_cost
        else:
            # Nothing loaded yet, so we are converged only when the AoN solution carries no cost either
            trivially_converged = aon_cost == 0.0
            self.rgap = 0.0 if trivially_converged else np.inf
            return trivially_converged

        if self.rgap_target >= self.rgap:
            return True
        return False
