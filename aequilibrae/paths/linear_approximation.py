import logging
import os
from functools import partial
from pathlib import Path
from tempfile import gettempdir
from typing import List, Dict

import numpy as np
from aequilibrae.paths.AoN import copy_two_dimensions, copy_three_dimensions
from aequilibrae.paths.AoN import linear_combination, linear_combination_skims, aggregate_link_costs
from aequilibrae.paths.AoN import sum_a_times_b_minus_c, linear_combination_1d
from aequilibrae.paths.AoN import triple_linear_combination, triple_linear_combination_skims
from scipy.optimize import root_scalar

from aequilibrae.paths.all_or_nothing import allOrNothing
from aequilibrae.paths.results import AssignmentResults
from aequilibrae.paths.traffic_class import TrafficClass

if False:
    from aequilibrae.paths.traffic_assignment import TrafficAssignment

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
        self.convergence_report = {"iteration": [], "rgap": [], "alpha": [], "warnings": []}
        if algorithm in ["cfw", "bfw"]:
            self.convergence_report["beta0"] = []
            self.convergence_report["beta1"] = []
            self.convergence_report["beta2"] = []

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

        # if FW stepsize is zero, we set it to the corresponding MSA stepsize and then need to not make
        # the step direction conjugate to the previous direction.
        self.do_fw_step = False
        self.conjugate_failed = False
        self.do_conjugate_step = False

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

        self.step_direction = {}  # type: Dict[AssignmentResults]
        self.previous_step_direction = {}  # type: Dict[AssignmentResults]
        self.temp_step_direction_for_copy = {}  # type: Dict[AssignmentResults]

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

    def __calculate_step_direction(self):
        """Calculates step direction depending on the method"""
        sd_flows = []

        # 2nd iteration is a fw step. if the previous step replaced the aggregated
        # solution so far, we need to start anew.
        if self.iter == 2 or self.stepsize == 1.0 or self.do_fw_step or self.algorithm in ["msa", "frank-wolfe"]:
            self.do_fw_step = False
            self.do_conjugate_step = True
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
        elif (self.iter == 3) or (self.do_conjugate_step) or (self.algorithm == "cfw"):
            self.do_conjugate_step = False
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
            self.calculate_biconjugate_direction()
            # deep copy because we overwrite step_direction but need it on next iteration
            for c in self.traffic_classes:
                ppst = self.temp_step_direction_for_copy[c._id]  # type: AssignmentResults
                prev_stp_dir = self.previous_step_direction[c._id]  # type: AssignmentResults
                stp_dir = self.step_direction[c._id]  # type: AssignmentResults

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

        self.logger.info(f"{self.algorithm} Assignment STATS")
        self.logger.info("Iteration, RelativeGap, stepsize")

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
                                c.results.select_link_loading[name],  # ouput matrix
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
                        for name, idx in c._aon_results._selected_links.items():
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

            self.fw_total_flow = np.sum(flows, axis=0)
            if self.preload is not None:
                self.fw_total_flow += self.preload

            if self.algorithm == "all-or-nothing":
                break

            # Check convergence
            # This needs to be done with the current costs, and not the future ones
            converged = self.check_convergence() if self.iter > 1 else False
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
        self.logger.info(f"{self.algorithm} Assignment finished. {self.iter} iterations and {self.rgap} final gap")
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

    def calculate_stepsize(self):
        """Calculate optimal stepsize in descent direction"""
        if self.algorithm == "msa":
            self.stepsize = 1.0 / self.iter
            return

        class_specific_term = self.__derivative_of_objective_stepsize_independent()
        derivative_of_objective = partial(
            self.__derivative_of_objective_stepsize_dependent, const_term=class_specific_term
        )

        x_tol = max(min(1e-6, self.rgap * 1e-5), 1e-12)

        try:
            min_res = root_scalar(derivative_of_objective, bracket=[0, 1], xtol=x_tol)
            self.stepsize = min_res.root
            if not min_res.converged:
                self.logger.warning("Descent direction stepsize finder has not converged")

            self.conjugate_failed = False

        except ValueError as e:
            # We can have iterations where the objective function is not *strictly* convex, but the scipy method cannot deal
            # with this. Stepsize is then either given by 1 or 0, depending on where the objective function is smaller.
            # However, using zero would mean the overall solution would not get updated, and therefore we assert the stepsize
            # in order to add a small fraction of the AoN. A heuristic value equal to the corresponding MSA step size
            # seems to work well in practice.
            if self.algorithm == "bfw":
                self.betas.fill(-1)
            if derivative_of_objective(0.0) < derivative_of_objective(1.0):
                if self.algorithm == "frank-wolfe" or self.conjugate_failed:
                    tiny_step = 1e-2 / self.iter  # use a fraction of the MSA stepsize. We observe that using 1e-4
                    # works well in practice, however for a large number of iterations this might be too much so
                    # use this heuristic instead.
                    self.logger.warning(f"# Alert: Adding {tiny_step} as step size to make it non-zero. {e.args}")
                    self.stepsize = tiny_step
                else:
                    self.stepsize = 0.0
                    # need to reset conjugate / bi-conjugate direction search
                    self.do_fw_step = True
                    self.conjugate_failed = True
                    self.iteration_issue.append("Found bad conjugate direction step. Performing FW search. {e.args}")
                    # By doing it recursively, we avoid doing the same AoN again
                    self.__calculate_step_direction()
                    self.calculate_stepsize()

            else:
                # Do we want to keep some of the old solution, or just throw away everything?
                self.stepsize = 1.0

        assert 0 <= self.stepsize <= 1.0

    def check_convergence(self):
        """Calculate relative gap and return ``True`` if it is smaller than desired precision"""
        aon_cost = np.sum(self.congested_time * self.aon_total_flow)
        current_cost = np.sum(self.congested_time * self.fw_total_flow)
        self.rgap = abs(current_cost - aon_cost) / current_cost
        if self.rgap_target >= self.rgap:
            return True
        return False
