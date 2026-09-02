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
        self.line_search = getattr(assig_spec, "line_search", "trapezoidal")  # CFW, BFW only
        self.bfw_conjugacy = getattr(assig_spec, "bfw_conjugacy", "approximate")  # BFW only
        # Conjugacy diagnostics for the iteration in progress; see _record_conjugacy_diagnostics.
        self.conjugacy_prev = np.nan
        self.conjugacy_prev2 = np.nan
        self.hessian_drift = np.nan
        self.bfw_clamped = np.nan
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
        if algorithm == "bfw":
            # Per-iteration conjugacy diagnostics, for comparing the approximate and exact BFW variants.
            # NaN on iterations that did not take a BFW step.
            self.convergence_report["conjugacy_prev"] = []
            self.convergence_report["conjugacy_prev2"] = []
            self.convergence_report["hessian_drift"] = []
            self.convergence_report["bfw_clamped"] = []

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
        # The PCE transformation makes the volume-dependent cost identical across classes, so each link's Hessian
        # block is rank one, H_a = t'_a * ones(M, M). Every contraction therefore separates over the class indices,
        #     u^T H v = sum_a t'_a (sum_m u_a^m) (sum_m v_a^m),
        # so we only need the class-aggregated vectors, never the M^2 class pairs. The same identity holds with
        # absolute values inside (|t'_a u_a^m v_a^m'| factors too), which gives the cancellation scale below.
        # This would no longer be valid if the volume-dependent term regained a class-dependent weight.
        # Accumulate in float64 explicitly: an in-place ``+=`` onto a
        # narrower accumulator would silently downcast every class contribution.
        u = np.zeros(self.vdf_der.shape, dtype=np.float64)  # sum_m (s_{k-1} - x_k)^m
        v = np.zeros(self.vdf_der.shape, dtype=np.float64)  # sum_m (y_k - x_k)^m
        w = np.zeros(self.vdf_der.shape, dtype=np.float64)  # sum_m (y_k - s_{k-1})^m
        abs_u = np.zeros(self.vdf_der.shape, dtype=np.float64)
        abs_w = np.zeros(self.vdf_der.shape, dtype=np.float64)

        for c in self.traffic_classes:
            stp_dir = self.step_direction[c._id]
            prev_dir_minus_current_sol = np.sum(stp_dir.link_loads[:, :] - c.results.link_loads[:, :], axis=1)
            aon_minus_current_sol = np.sum(c._aon_results.link_loads[:, :] - c.results.link_loads[:, :], axis=1)
            aon_minus_prev_dir = np.sum(c._aon_results.link_loads[:, :] - stp_dir.link_loads[:, :], axis=1)

            u += prev_dir_minus_current_sol
            v += aon_minus_current_sol
            w += aon_minus_prev_dir
            abs_u += np.abs(prev_dir_minus_current_sol)
            abs_w += np.abs(aon_minus_prev_dir)

        numerator = np.sum(self.vdf_der * u * v)
        denominator = np.sum(self.vdf_der * u * w)
        denominator_scale = np.sum(np.abs(self.vdf_der) * abs_u * abs_w)

        tolerance = np.finfo(np.float64).eps * float(denominator_scale)
        if (
            not np.isfinite(numerator)
            or not np.isfinite(denominator)
            or not np.isfinite(denominator_scale)
            or denominator_scale == 0.0
            or abs(denominator) <= tolerance
        ):
            self._reset_conjugate_direction("Invalid CFW coefficient; using the Frank-Wolfe direction.")
            return False

        alpha = numerator / denominator
        if not np.isfinite(alpha):
            self._reset_conjugate_direction("Non-finite CFW coefficient; using the Frank-Wolfe direction.")
            return False
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
        return True

    def calculate_biconjugate_direction(self):
        self.vdf.apply_derivative(
            self.vdf_der, self.fw_total_flow, self.capacity, self.free_flow_tt, *self.vdf_parameters, self.cores
        )
        # Class-aggregated vectors; see calculate_conjugate_stepsize for why the class pairs factor out. Following
        # appendix A of Mitradjieva & Lindberg, x_ is the residual direction d_{k-2}, z_ is d_{k-1}, y_ is the
        # Frank-Wolfe direction and w_ is s_{k-2} - s_{k-1}.
        # float64 accumulators for the same reason as in calculate_conjugate_stepsize.
        x_ = np.zeros(self.vdf_der.shape, dtype=np.float64)
        y_ = np.zeros(self.vdf_der.shape, dtype=np.float64)
        z_ = np.zeros(self.vdf_der.shape, dtype=np.float64)
        w_ = np.zeros(self.vdf_der.shape, dtype=np.float64)
        abs_x = np.zeros(self.vdf_der.shape, dtype=np.float64)
        abs_z = np.zeros(self.vdf_der.shape, dtype=np.float64)
        abs_w = np.zeros(self.vdf_der.shape, dtype=np.float64)

        for c in self.traffic_classes:
            sd = self.step_direction[c._id].link_loads[:, :]
            psd = self.previous_step_direction[c._id].link_loads[:, :]
            ll = c.results.link_loads[:, :]

            class_x = np.sum(sd * self.stepsize + psd * (1.0 - self.stepsize) - ll, axis=1)
            class_z = np.sum(sd - ll, axis=1)
            class_w = np.sum(psd - sd, axis=1)

            x_ += class_x
            y_ += np.sum(c._aon_results.link_loads[:, :] - ll, axis=1)
            z_ += class_z
            w_ += class_w
            abs_x += np.abs(class_x)
            abs_z += np.abs(class_z)
            abs_w += np.abs(class_w)

        if self.bfw_conjugacy == "exact":
            coefficients = self.__exact_biconjugate_coefficients(x_, y_, z_, w_, abs_x, abs_z, abs_w)
        else:
            coefficients = self.__approximate_biconjugate_coefficients(x_, y_, z_, w_, abs_x, abs_z, abs_w)
        if coefficients is None:
            return False  # the helper has already reset the direction and recorded why
        mu, nu = coefficients

        self.betas[0] = 1.0 / (1.0 + nu + mu)
        self.betas[1] = nu * self.betas[0]
        self.betas[2] = mu * self.betas[0]
        if not np.all(np.isfinite(self.betas)) or np.any(self.betas < 0.0):
            self._reset_conjugate_direction("Invalid BFW weights; using the Frank-Wolfe direction.")
            return False
        self._record_conjugacy_diagnostics(x_, y_, z_, w_, mu, nu)
        return True

    def __approximate_biconjugate_coefficients(self, x_, y_, z_, w_, abs_x, abs_z, abs_w):
        """Appendix A of Mitradjieva & Lindberg, which assumes ``d_{k-1}^T H_k d_{k-2} = 0``.

        That assumption holds only in the limit: the two directions were made conjugate with respect to
        ``H_{k-1}``, not ``H_k``. Returns ``(mu, nu)``, or ``None`` after resetting the direction.
        """
        mu_numerator = np.sum(self.vdf_der * x_ * y_)
        mu_denominator = np.sum(self.vdf_der * x_ * w_)
        mu_denominator_scale = np.sum(np.abs(self.vdf_der) * abs_x * abs_w)
        mu_tolerance = np.finfo(np.float64).eps * float(mu_denominator_scale)
        if (
            not np.isfinite(mu_numerator)
            or not np.isfinite(mu_denominator)
            or not np.isfinite(mu_denominator_scale)
            or mu_denominator_scale == 0.0
            or abs(mu_denominator) <= mu_tolerance
        ):
            self._reset_conjugate_direction("Invalid BFW mu coefficient; using the Frank-Wolfe direction.")
            return None

        mu_unclamped = -mu_numerator / mu_denominator
        mu = max(0.0, mu_unclamped)

        nu_nom = np.sum(self.vdf_der * z_ * y_)
        # Here both factors are z, so the class aggregate is squared link by link.
        nu_denom = np.sum(self.vdf_der * z_ * z_)
        nu_denominator_scale = np.sum(np.abs(self.vdf_der) * abs_z * abs_z)
        nu_tolerance = np.finfo(np.float64).eps * float(nu_denominator_scale)
        remaining_step = 1.0 - self.stepsize
        if (
            not np.isfinite(nu_nom)
            or not np.isfinite(nu_denom)
            or not np.isfinite(nu_denominator_scale)
            or not np.isfinite(mu)
            or not np.isfinite(remaining_step)
            or nu_denominator_scale == 0.0
            or abs(nu_denom) <= nu_tolerance
            # Any positive representable 1 - stepsize is a valid denominator. Comparing with machine epsilon would
            # reject valid steps immediately below one; subsequent finiteness checks catch numerical overflow instead.
            or remaining_step <= 0.0
        ):
            self._reset_conjugate_direction("Invalid BFW nu coefficient; using the Frank-Wolfe direction.")
            return None

        nu_unclamped = -(nu_nom / nu_denom) + mu * self.stepsize / remaining_step
        nu = max(0.0, nu_unclamped)
        if not np.isfinite(nu):
            self._reset_conjugate_direction("Non-finite BFW coefficient; using the Frank-Wolfe direction.")
            return None
        self.bfw_clamped = bool(mu_unclamped < 0.0 or nu_unclamped < 0.0)
        return mu, nu

    def __exact_biconjugate_coefficients(self, x_, y_, z_, w_, abs_x, abs_z, abs_w):
        """Solve both conjugacy conditions jointly, without appendix A's ``d_{k-1}^T H_k d_{k-2} = 0`` assumption.

        Writing ``d_k / beta_0 = g + (nu + mu) A + mu W`` with ``A = d_{k-1}``, ``B = d_{k-2}``,
        ``W = s_{k-2} - s_{k-1}`` and ``g`` the Frank-Wolfe direction, conditions (9a) and (9b) become the
        2x2 system in ``s = nu + mu`` and ``m = mu``::

            [ A'HA   A'HW ] [s]     [ -A'Hg ]
            [ B'HA   B'HW ] [m]  =  [ -B'Hg ]

        Dropping the off-diagonal ``B'HA`` and substituting ``A'HW = (A'HB - A'HA) / (1 - tau)`` with
        ``A'HB = 0`` recovers the appendix formulas exactly, so the two paths differ only in that one term.
        Returns ``(mu, nu)``, or ``None`` after resetting the direction.
        """
        a = np.sum(self.vdf_der * z_ * z_)  # A'HA
        b = np.sum(self.vdf_der * z_ * w_)  # A'HW
        c = np.sum(self.vdf_der * x_ * z_)  # B'HA, the term the approximation drops
        d = np.sum(self.vdf_der * x_ * w_)  # B'HW
        e = -np.sum(self.vdf_der * z_ * y_)  # -A'Hg
        f = -np.sum(self.vdf_der * x_ * y_)  # -B'Hg

        determinant = a * d - b * c
        # Cancellation scale for the determinant, built from the same class aggregates.
        scale_a = np.sum(np.abs(self.vdf_der) * abs_z * abs_z)
        scale_b = np.sum(np.abs(self.vdf_der) * abs_z * abs_w)
        scale_c = np.sum(np.abs(self.vdf_der) * abs_x * abs_z)
        scale_d = np.sum(np.abs(self.vdf_der) * abs_x * abs_w)
        determinant_scale = scale_a * scale_d + scale_b * scale_c
        tolerance = np.finfo(np.float64).eps * float(determinant_scale)

        if (
            not np.all(np.isfinite([a, b, c, d, e, f, determinant, determinant_scale]))
            or determinant_scale == 0.0
            or abs(determinant) <= tolerance
        ):
            # Singular here means A and B are H-parallel, so there is no direction conjugate to both.
            self._reset_conjugate_direction("Singular exact BFW system; using the Frank-Wolfe direction.")
            return None

        s = (e * d - b * f) / determinant
        m = (a * f - e * c) / determinant
        if not np.isfinite(s) or not np.isfinite(m):
            self._reset_conjugate_direction("Non-finite exact BFW coefficients; using the Frank-Wolfe direction.")
            return None

        if m >= 0.0 and s - m >= 0.0:
            self.bfw_clamped = False
            return m, s - m

        # The unconstrained optimum lies outside the simplex. Clamping one coefficient while keeping its
        # jointly-solved partner would leave a direction conjugate to neither previous direction, so instead
        # drop the offending direction and re-conjugate against the one that remains.
        # This is a deliberate difference from the approximate path, which clamps in place.
        self.bfw_clamped = True
        if m < 0.0:
            # Zero weight on d_{k-2}: solve (9a) alone, which is exactly the CFW conjugacy condition.
            if abs(a) <= np.finfo(np.float64).eps * float(scale_a):
                return 0.0, 0.0  # degenerate; falls back to the plain Frank-Wolfe direction
            return 0.0, max(0.0, e / a)
        # Zero weight on d_{k-1}, i.e. nu = 0 and hence s = m: solve (9b) alone.
        denominator, denominator_scale = c + d, scale_c + scale_d
        if abs(denominator) <= np.finfo(np.float64).eps * float(denominator_scale):
            return 0.0, 0.0
        return max(0.0, f / denominator), 0.0

    def _record_conjugacy_diagnostics(self, x_, y_, z_, w_, mu, nu):
        """Measure how conjugate the accepted direction actually is, for comparing the two BFW variants.

        Stores three cosines in ``[-1, 1]`` on the instance and in the convergence report:

        * ``conjugacy_prev``  - cos angle in the H inner product between ``d_k`` and ``d_{k-1}``; the exact
          solve should drive this to zero, the approximation only approximately.
        * ``conjugacy_prev2`` - the same against ``d_{k-2}``.
        * ``hessian_drift``   - cos between ``d_{k-1}`` and ``d_{k-2}``. This is exactly the term appendix A
          assumes is zero, so it bounds how much the two variants can possibly differ.
        """
        direction = y_ + (nu + mu) * z_ + mu * w_  # d_k / beta_0
        dd = np.sum(self.vdf_der * direction * direction)
        zz = np.sum(self.vdf_der * z_ * z_)
        xx = np.sum(self.vdf_der * x_ * x_)

        def cosine(numerator, left, right):
            denominator = np.sqrt(left * right)
            if not np.isfinite(denominator) or denominator <= 0.0 or not np.isfinite(numerator):
                return np.nan
            return float(numerator / denominator)

        self.conjugacy_prev = cosine(np.sum(self.vdf_der * z_ * direction), zz, dd)
        self.conjugacy_prev2 = cosine(np.sum(self.vdf_der * x_ * direction), xx, dd)
        self.hessian_drift = cosine(np.sum(self.vdf_der * x_ * z_), xx, zz)
        self.logger.debug(
            f"BFW[{self.bfw_conjugacy}] iter={self.iter} mu={mu:.6e} nu={nu:.6e} "
            f"betas=({self.betas[0]:.6e},{self.betas[1]:.6e},{self.betas[2]:.6e}) "
            f"conjugacy_prev={self.conjugacy_prev:.3e} conjugacy_prev2={self.conjugacy_prev2:.3e} "
            f"hessian_drift={self.hessian_drift:.3e} clamped={self.bfw_clamped}"
        )

    def _reset_conjugate_direction(self, message: str):
        self.conjugate_stepsize = 0.0
        self.betas[:] = (1.0, 0.0, 0.0)
        self.current_direction = "fw"
        if self.algorithm == "bfw":
            self.next_direction = "cfw"
        self.logger.debug(message)
        self.iteration_issue.append(message)

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

    def _append_convergence_report(self, terminal: bool = False):
        self.convergence_report["time"].append(time.perf_counter() - self.__start_time)
        self.convergence_report["iteration"].append(self.iter)
        self.convergence_report["rgap"].append(self.rgap)
        self.convergence_report["warnings"].append("; ".join(self.iteration_issue))
        self.convergence_report["alpha"].append(np.nan if terminal else self.stepsize)
        if self.algorithm in ["cfw", "bfw"]:
            for key, beta in zip(("beta0", "beta1", "beta2"), self.betas, strict=True):
                self.convergence_report[key].append(np.nan if terminal else beta)
        if self.algorithm == "bfw":
            diagnostics = (self.conjugacy_prev, self.conjugacy_prev2, self.hessian_drift, self.bfw_clamped)
            keys = ("conjugacy_prev", "conjugacy_prev2", "hessian_drift", "bfw_clamped")
            for key, value in zip(keys, diagnostics, strict=True):
                self.convergence_report[key].append(np.nan if terminal else value)
        self.logger.info(f"{self.iter},{self.rgap},{'nan' if terminal else self.stepsize}")

    def __calculate_step_direction(self):  # noqa: C901
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
            if not self.calculate_conjugate_stepsize():
                self.next_direction = "fw"
                self.__calculate_step_direction()
                return
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
            if not self.calculate_biconjugate_direction():
                self.next_direction = "fw"
                self.__calculate_step_direction()
                return
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
        if self.preload is not None:
            self.step_direction_flow += self.preload

    def __retry_with_fw_direction(self, msg: str):
        self._reset_conjugate_direction(msg)
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
            # Stale diagnostics must not be reported on an iteration that ends up taking an FW or CFW step.
            self.conjugacy_prev = np.nan
            self.conjugacy_prev2 = np.nan
            self.hessian_drift = np.nan
            self.bfw_clamped = np.nan

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

            converged = self.check_convergence() if self.iter > 1 else False
            if converged:
                self.steps_below += 1
                if self.steps_below >= self.steps_below_needed_to_terminate:
                    self._append_convergence_report(terminal=True)
                    break
            else:
                self.steps_below = 0

            if self.iter == self.max_iter and self.iter > 1:
                self._append_convergence_report(terminal=True)
                break

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

            self._refresh_congested_costs()

            self._append_convergence_report()
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
                c.fixed_cost, self.step_direction[c._id].total_link_loads, c.results.total_link_loads, self.cores
            )
            class_specific_term += class_link_costs
        return class_specific_term

    def __objective_change_at_stepsize(
        self, derivative_of_objective_stepsize_independent: np.ndarray, stepsize: float
    ) -> float:
        """Heuristic trapezoidal approximation of the Beckmann objective change
        ``Z(x + α·d) − Z(x)`` for a given line-search step ``α = stepsize``.

        This one-panel approximation is not the exact Beckmann integral except for affine link costs.
        However, experiments suggest that
        on large congested networks (e.g. Chicago, BPR β=4), this trapezoidal line search picks smaller,
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

        # With line_search == "trapezoidal", CFW and BFW use a heuristic bounded minimization of a one-panel
        # trapezoidal approximation to the Beckmann objective change instead of root-finding the exact
        # directional derivative. Selected via TrafficAssignment.set_line_search; "exact" falls through to the
        # root_scalar branch below, which is the line search the conjugate-direction theory assumes.
        #
        # Two cooperating mechanisms vs. the analytic root_scalar approach:
        #
        # (1) The trapezoidal objective is exact for affine link costs and approximate otherwise;
        #     the two diverge significantly when the BPR exponent is large (β=4 on Chicago test network).
        # (2) For BFW only: a cap α_max = 1/sqrt(iter) prevents the line search from
        #     returning α = 1.0, which would collapse the BFW history (s^{k-1} onto x^k)
        #     and cause the μ·α/(1-α) bias term in calculate_biconjugate_direction to blow up.
        #     CFW has neither concern and uses α_max = 1.0 (uncapped).
        #
        # BFW Chicago-50 rgap: 1.14e-3 (was 1.54e-3 at HEAD baseline).
        if self.algorithm in ("bfw", "cfw") and self.line_search == "trapezoidal":
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

        # Exact line search: root-find the directional derivative of the Beckmann objective over [0, 1]. Used by
        # Frank-Wolfe always, and by CFW/BFW when line_search == "exact". No step cap is applied here.
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

        ``self.rgap`` uses the AequilibraE convention,
        ``|Σ flow·cost − Σ AON·cost| / Σ flow·cost``.
        """
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
