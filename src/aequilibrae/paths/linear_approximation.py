from __future__ import annotations

import logging
import os
import time
from functools import partial
from tempfile import gettempdir
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import root_scalar

from aequilibrae.paths.all_or_nothing import allOrNothing
from aequilibrae.paths.cython.parallel_numpy import (
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

logger = logging.getLogger(__name__)


class LinearApproximation(WorkerThread):
    equilibration = SIGNAL(object)
    assignment = SIGNAL(object)
    signal = SIGNAL(object)

    def __init__(self, assig_spec: TrafficAssignment, algorithm, project=None) -> None:
        WorkerThread.__init__(self, None)
        self.signal.emit(["set_text", "Linear Approximation"])

        self.project_path = project.project_base_path if project else gettempdir()

        self.algorithm = algorithm
        self.bfw_conjugacy = getattr(assig_spec, "bfw_conjugacy", "approximate")  # BFW only
        # Conjugacy diagnostics for the iteration in progress; see _record_conjugacy_diagnostics.
        self.conjugacy_prev = np.nan
        self.conjugacy_prev2 = np.nan
        self.hessian_drift = np.nan
        self.bfw_clamped = np.nan
        self.rgap_target = assig_spec.rgap_target
        self.max_iter = assig_spec.max_iter
        self.cores = assig_spec.cores
        self.elementwise_cores = assig_spec.elementwise_cores
        self.threading_threshold = assig_spec.threading_threshold
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

        if assig_spec.classes is None:
            raise ValueError(
                "Traffic classes parameter missing. Setting the algorithm is the last thing to do when assigning."
            )
        elif assig_spec.vdf is None:
            raise ValueError(
                "vdf has not been specified. Setting the algorithm is the last thing to do when assigning."
            )
        elif assig_spec.vdf_parameters is None:
            raise ValueError("vdf_parameters missing. Setting the algorithm is the last thing to do when assigning.")
        elif assig_spec.capacity_field is None:
            raise ValueError("capacity field is not set in TrafficAssignment.")
        elif assig_spec.time_field is None:
            raise ValueError("time field is not set in TrafficAssignment.")

        self.traffic_classes: list[TrafficClass] = assig_spec.classes
        self.num_classes = len(assig_spec.classes)

        self.time_field = assig_spec.time_field
        self.vdf = assig_spec.vdf
        self.vdf_parameters = assig_spec.vdf_parameters
        self.capacity = assig_spec.capacity
        self.procedure_id: str = assig_spec.procedure_id

        self.iter = 0
        self.rgap: int | float = np.inf
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

        # Creates preload vector from preloads
        self.preload = None
        if assig_spec.preloads is not None:
            cols = assig_spec.preloads.columns.difference(["link_id", "direction"])
            self.preload = assig_spec.preloads[cols].sum(axis=1).to_numpy()

        self.free_flow_tt = assig_spec.free_flow_tt
        self.total_flow = assig_spec.total_flow
        self.congested_time = assig_spec.congested_time
        self.vdf_der = np.array(assig_spec.congested_time, copy=True)
        self.congested_value = np.array(assig_spec.congested_time, copy=True)

        # Turn penalty cost tracking for convergence calculation
        self.fw_total_turn_cost = 0.0
        self.aon_total_turn_cost = 0.0
        self.step_direction_turn_cost = {}
        self.previous_step_direction_turn_cost = {}

        self.step_direction: dict[str, AssignmentResults] = {}
        self.previous_step_direction: dict[str, AssignmentResults] = {}
        self.temp_step_direction_for_copy: dict[str, AssignmentResults] = {}

        self.aons = {}

        for c in self.traffic_classes:
            r = AssignmentResults()
            r.prepare(c.graph, c.matrix)
            self.step_direction[c._id] = r
            self.step_direction_turn_cost[c._id] = 0.0
            self.previous_step_direction_turn_cost[c._id] = 0.0

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
            delta=self.vdf_der,
            link_flows=self.total_flow,
            fftime=self.free_flow_tt,
            capacity=self.capacity,
            cores=self.elementwise_cores,
            **self.vdf_parameters,
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
            delta=self.vdf_der,
            link_flows=self.total_flow,
            fftime=self.free_flow_tt,
            capacity=self.capacity,
            cores=self.elementwise_cores,
            **self.vdf_parameters,
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
        logger.debug(
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
        logger.debug(message)
        self.iteration_issue.append(message)

    def _apply_assigned_flow(self, link_flows):
        """Records the flows just assigned, stacking any preload on top of them."""
        total = np.array(link_flows, dtype=np.float64, copy=True)
        if self.preload is not None:
            total += self.preload
        self.total_flow = total

    def _refresh_congested_costs(self):
        self.vdf.apply_vdf(
            congested_time=self.congested_time,
            link_flows=self.total_flow,
            fftime=self.free_flow_tt,
            capacity=self.capacity,
            cores=self.elementwise_cores,
            **self.vdf_parameters,
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
        logger.info(f"{self.iter},{self.rgap},{'nan' if terminal else self.stepsize}")

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
                copy_two_dimensions(
                    stp_dir_res.link_loads, aon_res.link_loads, self.elementwise_cores, self.threading_threshold
                )
                stp_dir_res.total_flows()
                if c.results.num_skims > 0:
                    copy_three_dimensions(
                        stp_dir_res.skims.matrix_view,
                        aon_res.skims.matrix_view,
                        self.elementwise_cores,
                        self.threading_threshold,
                    )
                sd_flows.append(aon_res.total_link_loads)
                # Step direction for FW/MSA is AoN
                self.step_direction_turn_cost[c._id] = aon_res.total_turn_penalty

                if c._selected_links:
                    aux_res = self.aons[c._id].aux_res
                    for name, idx in c._aon_results._selected_links.items():
                        copy_two_dimensions(
                            self.sl_step_dir_ll[c._id][name]["sdr"],
                            np.sum(aux_res.temp_sl_link_loading, axis=0)[idx, :, :],
                            self.elementwise_cores,
                            self.threading_threshold,
                        )
                        copy_three_dimensions(
                            self.sl_step_dir_od[c._id][name]["sdr"],
                            np.sum(aux_res.temp_sl_od_matrix, axis=0)[idx, :, :, :],
                            self.elementwise_cores,
                            self.threading_threshold,
                        )

        # 3rd iteration is cfw. also, if we had to reset direction search we need a cfw step before bfw
        elif (self.iter == 3) or (direction == "cfw") or (self.algorithm == "cfw"):
            self.current_direction = "cfw"
            if not self.calculate_conjugate_stepsize():
                self.next_direction = "fw"
                self.__calculate_step_direction()
                return
            # The conjugate direction is computed into a spare buffer and the
            # references rotated (current -> previous, spare -> current) instead
            # of copying the current direction out of the way.
            for c in self.traffic_classes:
                sdr = self.step_direction[c._id]
                spare = self.temp_step_direction_for_copy[c._id]

                linear_combination(
                    spare.link_loads,
                    sdr.link_loads,
                    c._aon_results.link_loads,
                    self.conjugate_stepsize,
                    self.elementwise_cores,
                    self.threading_threshold,
                )

                if c.results.num_skims > 0:
                    linear_combination_skims(
                        spare.skims.matrix_view,
                        sdr.skims.matrix_view,
                        c._aon_results.skims.matrix_view,
                        self.conjugate_stepsize,
                        self.elementwise_cores,
                        self.threading_threshold,
                    )

                # Update turn cost with the same conjugate stepsize
                self.previous_step_direction_turn_cost[c._id] = self.step_direction_turn_cost[c._id]
                self.step_direction_turn_cost[c._id] = (1.0 - self.conjugate_stepsize) * self.step_direction_turn_cost[
                    c._id
                ] + self.conjugate_stepsize * c._aon_results.total_turn_penalty

                if c._selected_links:
                    aux_res = self.aons[c._id].aux_res
                    for name, idx in c._aon_results._selected_links.items():
                        sl_step_dir_ll = self.sl_step_dir_ll[c._id][name]
                        sl_step_dir_od = self.sl_step_dir_od[c._id][name]

                        linear_combination(
                            sl_step_dir_ll["temp_prev_sdr"],
                            sl_step_dir_ll["sdr"],
                            np.sum(aux_res.temp_sl_link_loading, axis=0)[idx, :, :],
                            self.conjugate_stepsize,
                            self.elementwise_cores,
                            self.threading_threshold,
                        )

                        linear_combination_skims(
                            sl_step_dir_od["temp_prev_sdr"],
                            sl_step_dir_od["sdr"],
                            np.sum(aux_res.temp_sl_od_matrix, axis=0)[idx, :, :, :],
                            self.conjugate_stepsize,
                            self.elementwise_cores,
                            self.threading_threshold,
                        )

                        self.__rotate_select_link_buffers(sl_step_dir_ll, sl_step_dir_od)

                self.__rotate_direction_buffers(c._id)

                spare.total_flows()
                sd_flows.append(spare.total_link_loads)
        # biconjugate
        else:
            self.current_direction = "bfw"
            if not self.calculate_biconjugate_direction():
                self.next_direction = "fw"
                self.__calculate_step_direction()
                return
            # The biconjugate direction is computed into a spare buffer and the
            # references rotated (current -> previous, spare -> current, previous
            # -> spare) instead of shuffling the arrays through copies.
            for c in self.traffic_classes:
                spare: AssignmentResults = self.temp_step_direction_for_copy[c._id]
                prev_stp_dir: AssignmentResults = self.previous_step_direction[c._id]
                stp_dir: AssignmentResults = self.step_direction[c._id]

                triple_linear_combination(
                    spare.link_loads,
                    c._aon_results.link_loads,
                    stp_dir.link_loads,
                    prev_stp_dir.link_loads,
                    self.betas,
                    self.elementwise_cores,
                    self.threading_threshold,
                )

                if c.results.num_skims > 0:
                    triple_linear_combination_skims(
                        spare.skims.matrix_view,
                        c._aon_results.skims.matrix_view,
                        stp_dir.skims.matrix_view,
                        prev_stp_dir.skims.matrix_view,
                        self.betas,
                        self.elementwise_cores,
                        self.threading_threshold,
                    )

                # Update turn cost with the same beta weights
                prev_turn_cost = self.step_direction_turn_cost[c._id]
                self.step_direction_turn_cost[c._id] = (
                    self.betas[0] * c._aon_results.total_turn_penalty
                    + self.betas[1] * self.step_direction_turn_cost[c._id]
                    + self.betas[2] * self.previous_step_direction_turn_cost[c._id]
                )
                self.previous_step_direction_turn_cost[c._id] = prev_turn_cost

                if c._selected_links:
                    aux_res = self.aons[c._id].aux_res
                    for name, idx in c._aon_results._selected_links.items():
                        sl_step_dir_ll = self.sl_step_dir_ll[c._id][name]
                        sl_step_dir_od = self.sl_step_dir_od[c._id][name]

                        triple_linear_combination(
                            sl_step_dir_ll["temp_prev_sdr"],
                            np.sum(aux_res.temp_sl_link_loading, axis=0)[idx, :, :],
                            sl_step_dir_ll["sdr"],
                            sl_step_dir_ll["prev_sdr"],
                            self.betas,
                            self.elementwise_cores,
                            self.threading_threshold,
                        )

                        triple_linear_combination_skims(
                            sl_step_dir_od["temp_prev_sdr"],
                            np.sum(aux_res.temp_sl_od_matrix, axis=0)[idx, :, :, :],
                            sl_step_dir_od["sdr"],
                            sl_step_dir_od["prev_sdr"],
                            self.betas,
                            self.elementwise_cores,
                            self.threading_threshold,
                        )

                        self.__rotate_select_link_buffers(sl_step_dir_ll, sl_step_dir_od)

                self.__rotate_direction_buffers(c._id)

                spare.total_flows()
                sd_flows.append(spare.total_link_loads)

        self.step_direction_flow = np.sum(sd_flows, axis=0)
        if self.preload is not None:
            self.step_direction_flow += self.preload

    def __rotate_direction_buffers(self, c_id: str):
        """Promotes the direction just computed into the spare buffer
        (``temp_step_direction_for_copy``): it becomes the current step
        direction, the old current becomes the previous and the old previous is
        recycled as the next spare."""
        self.step_direction[c_id], self.previous_step_direction[c_id], self.temp_step_direction_for_copy[c_id] = (
            self.temp_step_direction_for_copy[c_id],
            self.step_direction[c_id],
            self.previous_step_direction[c_id],
        )

    @staticmethod
    def __rotate_select_link_buffers(sl_step_dir_ll: dict, sl_step_dir_od: dict):
        """Promotes a select-link direction just computed into ``temp_prev_sdr``:
        it becomes ``sdr``, the old ``sdr`` becomes ``prev_sdr`` and the old
        ``prev_sdr`` is recycled as the next scratch buffer."""
        for buffers in (sl_step_dir_ll, sl_step_dir_od):
            buffers["sdr"], buffers["prev_sdr"], buffers["temp_prev_sdr"] = (
                buffers["temp_prev_sdr"],
                buffers["sdr"],
                buffers["prev_sdr"],
            )

    def __retry_with_fw_direction(self, msg: str):
        self._reset_conjugate_direction(msg)
        self.next_direction = "fw"
        self.__calculate_step_direction()
        self.calculate_stepsize()

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

        self._apply_assigned_flow(np.zeros_like(self.congested_time))
        self._refresh_congested_costs()

        logger.info(f"{self.algorithm} Assignment stats")
        logger.info("Iteration, RelativeGap (AoN), stepsize")

        msg = "Equilibrium Assignment"
        for self.iter in simple_progress(range(1, self.max_iter + 1), self.signal, msg):  # noqa: B020
            self.iteration_issue = []
            # Stale diagnostics must not be reported on an iteration that ends up taking an FW or CFW step.
            self.conjugacy_prev = np.nan
            self.conjugacy_prev2 = np.nan
            self.hessian_drift = np.nan
            self.bfw_clamped = np.nan

            aon_flows = []

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

                if aon.results.save_path_file:
                    aon.aux_res.save_path_files(
                        os.path.join(self.project_path, "path_files.h5"),
                        aon.graph,
                        self.iter,
                    )

                c._aon_results.link_loads *= c.pce
                c._aon_results.total_flows()
                aon_flows.append(c._aon_results.total_link_loads)

            self.aon_total_flow = np.sum(aon_flows, axis=0)

            # Accumulate AoN turn penalty costs from all traffic classes.
            self.aon_total_turn_cost = sum(c._aon_results.total_turn_penalty for c in self.traffic_classes)

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
                    copy_two_dimensions(
                        c.results.link_loads,
                        c._aon_results.link_loads,
                        self.elementwise_cores,
                        self.threading_threshold,
                    )
                    c.results.total_flows()
                    if c.results.num_skims > 0:
                        copy_three_dimensions(
                            c.results.skims.matrix_view,
                            c._aon_results.skims.matrix_view,
                            self.elementwise_cores,
                            self.threading_threshold,
                        )

                    if c._selected_links:
                        for name, idx in c._aon_results._selected_links.items():
                            # Copy the temporary results into the final od matrix, referenced by link_set name
                            # The temp has an index associated with the link_set name
                            copy_three_dimensions(
                                c.results.select_link_od.matrix[name],  # matrix being written into
                                np.sum(self.aons[c._id].aux_res.temp_sl_od_matrix, axis=0)[
                                    idx, :, :, :
                                ],  # results after the iteration
                                self.elementwise_cores,  # core count
                                self.threading_threshold,
                            )
                            copy_two_dimensions(
                                c.results.select_link_loading[name],  # output matrix
                                np.sum(self.aons[c._id].aux_res.temp_sl_link_loading, axis=0)[idx, :, :],  # matrix 1
                                self.elementwise_cores,  # core count
                                self.threading_threshold,
                            )
                    flows.append(c.results.total_link_loads)

                # For iteration 1, turn cost equals AoN turn cost
                self.fw_total_turn_cost = self.aon_total_turn_cost

            else:
                self.__calculate_step_direction()
                self.calculate_stepsize()
                for c in self.traffic_classes:
                    stp_dir = self.step_direction[c._id]

                    cls_res = c.results

                    linear_combination(
                        cls_res.link_loads,
                        stp_dir.link_loads,
                        cls_res.link_loads,
                        self.stepsize,
                        self.elementwise_cores,
                        self.threading_threshold,
                    )

                    if cls_res.num_skims > 0:
                        linear_combination_skims(
                            cls_res.skims.matrix_view,
                            stp_dir.skims.matrix_view,
                            cls_res.skims.matrix_view,
                            self.stepsize,
                            self.elementwise_cores,
                            self.threading_threshold,
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
                                self.elementwise_cores,  # core count
                                self.threading_threshold,
                            )

                            linear_combination(
                                cls_res.select_link_loading[name],  # output matrix
                                self.sl_step_dir_ll[c._id][name]["sdr"],
                                cls_res.select_link_loading[name],  # matrix 2 (previous iteration)
                                self.stepsize,  # stepsize
                                self.elementwise_cores,  # core count
                                self.threading_threshold,
                            )

                    cls_res.total_flows()
                    flows.append(cls_res.total_link_loads)

                # Update aggregate turn cost with the same stepsize used for flows.
                # Turn penalties are fixed costs (not flow-dependent VDF outputs), so this
                # convex combination tracks the weighted-average turn cost of the current
                # flow solution - analogous to how link flows are combined.
                direction_turn_cost = sum(self.step_direction_turn_cost.values())  # TODO: optimize aggregation
                self.fw_total_turn_cost = (
                    self.stepsize * direction_turn_cost + (1.0 - self.stepsize) * self.fw_total_turn_cost
                )

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
            logger.error(f"Desired RGap of {self.rgap_target} was NOT reached")
        logger.info(f"{self.algorithm} Assignment finished. {self.iter} iterations, final AoN rgap = {self.rgap}")

        self.signal.emit(["finished"])

    def __derivative_of_objective_stepsize_dependent(self, stepsize, const_term):
        """The stepsize-dependent part of the derivative of the objective function. If fixed costs are defined,
        the corresponding contribution needs to be passed in"""
        x = np.zeros_like(self.total_flow)
        linear_combination_1d(
            x, self.step_direction_flow, self.total_flow, stepsize, self.elementwise_cores, self.threading_threshold
        )
        # x = self.total_flow + stepsize * (self.step_direction_flow - self.total_flow)
        self.vdf.apply_vdf(
            congested_time=self.congested_value,
            link_flows=x,
            fftime=self.free_flow_tt,
            capacity=self.capacity,
            cores=self.elementwise_cores,
            **self.vdf_parameters,
        )
        link_cost_term = sum_a_times_b_minus_c(
            self.congested_value,
            self.step_direction_flow,
            self.total_flow,
            self.elementwise_cores,
            self.threading_threshold,
        )
        return link_cost_term + const_term

    def __derivative_of_objective_stepsize_independent(self):
        """The part of the derivative of the objective function that does not dependent on stepsize. Non-zero
        only for fixed cost contributions."""
        class_specific_term = 0.0
        for c in self.traffic_classes:
            # fixed cost is scaled by vot
            class_link_costs = sum_a_times_b_minus_c(
                c.fixed_cost,
                self.step_direction[c._id].total_link_loads,
                c.results.total_link_loads,
                self.elementwise_cores,
                self.threading_threshold,
            )
            class_specific_term += class_link_costs
        return class_specific_term

    def __clip_stepsize(self, stepsize: float) -> float:
        if not np.isfinite(stepsize):
            raise ValueError(f"Non-finite stepsize {stepsize} encountered")

        clipped = min(max(float(stepsize), 0.0), 1.0)
        if clipped != stepsize:
            msg = f"Stepsize {stepsize} outside [0, 1]; clipping to {clipped}."
            logger.debug(msg)
            self.iteration_issue.append(msg)
        return clipped

    def calculate_stepsize(self):
        """Calculate optimal stepsize in descent direction"""
        if self.algorithm == "msa":
            self.stepsize = self.__clip_stepsize(1.0 / self.iter)
            return

        # Exact line search: root-find the directional derivative of the Beckmann objective over [0, 1].
        # This is the line search the conjugate-direction theory of Mitradjieva & Lindberg assumes, and it
        # is used by every descent algorithm here. No step cap is applied.
        class_specific_term = self.__derivative_of_objective_stepsize_independent()
        # TODO: optimize aggregation
        turn_derivative = sum(self.step_direction_turn_cost.values()) - self.fw_total_turn_cost
        # Turn penalties are constant w.r.t. stepsize (they don't depend on flows or VDF),
        # so they shift the derivative by a fixed amount. Including them here ensures the
        # line search accounts for turn costs when finding the optimal stepsize.
        derivative_of_objective = partial(
            self.__derivative_of_objective_stepsize_dependent, const_term=class_specific_term + turn_derivative
        )

        x_tol = max(min(1e-6, self.rgap * 1e-5), 1e-12)

        try:
            min_res = root_scalar(derivative_of_objective, bracket=[0, 1], xtol=x_tol)
            self.stepsize = self.__clip_stepsize(min_res.root)
            if not min_res.converged:
                logger.warning("Descent direction stepsize finder has not converged")

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
                    # For the Frank-Wolfe direction this derivative is exactly the negated gap, so
                    # a non-negative value means the same impossibility the convergence check
                    # guards against. Explain it before papering over it with a nominal step.
                    # congested_value holds C(x) from the derivative evaluation just above.
                    direction = self.step_direction_flow - self.total_flow
                    derivative_scale = float(np.sum(np.abs(self.congested_value * direction)))
                    for c in self.traffic_classes:
                        derivative_scale += float(
                            np.sum(
                                np.abs(
                                    c.fixed_cost
                                    * (self.step_direction[c._id].total_link_loads - c.results.total_link_loads)
                                )
                            )
                        )
                    self._diagnose_negative_gap("line search", -float(d0_for_branch), derivative_scale)

                    tiny_step = 1e-2 / self.iter  # use a fraction of the MSA stepsize. We observe that using 1e-4
                    # works well in practice, however for a large number of iterations this might be too much so
                    # use this heuristic instead.
                    logger.warning(f"# Alert fw ex: Adding {tiny_step} as step size to make it non-zero. {e.args}")
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
                logger.info("Line-search optimum at the boundary (alpha = 1.0); descent throughout [0, 1]")

        assert 0 <= self.stepsize <= 1.0

    # Accumulated tolerance for the cost sums. numpy's pairwise summation gives an error of order
    # log2(links) * eps * sum|terms|; 64 eps is comfortably above that for any realistic network
    # while still being ~1e-14 relative, far below any error worth reporting.
    _GAP_NOISE_FACTOR = 64.0

    def _diagnose_negative_gap(self, where: str, signed: float, scale: float) -> bool:
        """Explain an all-or-nothing solution that costs more than the current one.

        The all-or-nothing assignment minimises ``C(x)^T z`` over the feasible set, so ``aon_cost``
        can never exceed ``current_cost`` and the Frank-Wolfe direction can never fail to be a
        descent direction. If it does, either the gap has reached the precision floor of the cost
        sums, or the cost handed to the shortest path is not the gradient of the objective being
        minimised. The second is a modelling or configuration error and the user needs to know.

        ``signed`` is the quantity that must be non-negative, and ``scale`` is the sum of the
        magnitudes of the terms it was accumulated from -- the cancellation scale. The scale has to
        come from the caller: the size of a difference says nothing about the rounding error of the
        sums that produced it.

        Returns ``True`` when the violation is real rather than rounding. Costs nothing unless a
        violation has already been detected by the caller.
        """
        tolerance = self._GAP_NOISE_FACTOR * np.finfo(np.float64).eps * scale
        if -signed <= tolerance:
            # Indistinguishable from zero: the iterate is optimal to the precision available.
            logger.info(
                f"Gap is negative by {-signed:.3e} at iteration {self.iter}, within the "
                f"{tolerance:.3e} noise floor of the cost sums. Treating as converged."
            )
            return False

        # self._negative_gap_iterations += 1
        # if self._negative_gap_iterations > 1:
        #    # Already explained once; do not repeat the full report every iteration.
        self.iteration_issue.append("All-or-nothing solution costs more than the current one.")
        #    return True

        relative = -signed / scale if scale else float("nan")
        detail = [
            f"the all-or-nothing solution costs MORE than the current one, by "
            f"{-signed:.6e} ({relative:.3e} relative), which is "
            f"{-signed / tolerance if tolerance else float('inf'):.3g} times the numerical "
            f"noise floor.",
            "This is not possible when the cost minimised by the shortest path is the gradient of "
            "the objective, so the two have diverged. The assignment will continue but the "
            "relative gap is no longer meaningful.",
        ]

        congested = np.asarray(self.congested_time, dtype=np.float64)
        nonfinite = int((~np.isfinite(congested)).sum())
        if nonfinite:
            detail.append(
                f"CAUSE: {nonfinite} link(s) have a non-finite congested time (max finite value "
                f"{np.nanmax(congested[np.isfinite(congested)]) if nonfinite < congested.size else float('nan'):.6g}). "
                f"Input fields are validated at setup but the volume-delay function output is not, "
                f"so this is an overflow during the run. A large BPR exponent combined with a "
                f"volume well above capacity will do it; check the exponent and the capacity units."
            )
        else:
            worst_id, worst_amount = None, 0.0
            for c in self.traffic_classes:
                unit = congested + np.asarray(c.fixed_cost, dtype=np.float64)
                violation = float(
                    np.sum(unit * c._aon_results.total_link_loads) - np.sum(unit * c.results.total_link_loads)
                )
                if violation > worst_amount:
                    worst_id, worst_amount = c._id, violation
                negative = int((unit < 0).sum())
                if negative:
                    detail.append(
                        f"CAUSE: class '{c._id}' has {negative} link(s) with negative total cost "
                        f"(minimum {unit.min():.6g}). Shortest-path search assumes non-negative "
                        f"arc costs, so its solution is not a lower bound."
                    )
            if worst_id is not None and len(self.traffic_classes) > 1:
                detail.append(f"The violation is largest for class '{worst_id}'.")
            if not any(d.startswith("CAUSE") for d in detail):
                detail.append(
                    "Costs are finite and non-negative, so check for a term present on one side "
                    "only: a fixed cost scaled by a value of time, a toll or distance term added "
                    "to the graph cost but not to the objective, or a volume-delay function whose "
                    "derivative is inconsistent with the integral the objective uses."
                )

        message = f"Iteration {self.iter} ({where}): " + " ".join(detail)
        logger.warning(message)
        self.iteration_issue.append("All-or-nothing solution costs more than the current one.")
        return True

    def check_convergence(self):
        """Calculate relative gap and return ``True`` if it is smaller than desired precision.

        ``self.rgap`` uses the AequilibraE convention,
        ``|Σ flow·cost − Σ AON·cost| / Σ flow·cost``.
        """
        # Include turn penalty costs in the objective function.
        # Turn penalties are fixed (not flow-dependent), so they act like additive constants
        # in the Beckmann objective. They don't affect the VDF derivative, but they must be
        # included in the gap calculation to correctly measure convergence when turn costs
        # are a significant fraction of total travel cost.
        aon_cost = self.aon_total_turn_cost
        current_cost = self.fw_total_turn_cost
        for c in self.traffic_classes:
            aon_class_flow = c._aon_results.total_link_loads
            current_class_flow = c.results.total_link_loads

            aon_cost += np.sum((self.congested_time + c.fixed_cost) * aon_class_flow)
            current_cost += np.sum((self.congested_time + c.fixed_cost) * current_class_flow)

        if current_cost != 0.0:
            # abs() below keeps the historical reporting convention, but a negative difference is
            # not a small gap -- it is impossible, so check the sign before discarding it.
            if current_cost < aon_cost:
                self._diagnose_negative_gap(
                    "convergence check", current_cost - aon_cost, abs(current_cost) + abs(aon_cost)
                )
            self.rgap = abs(current_cost - aon_cost) / current_cost
            # ``step_direction`` is populated by ``__calculate_step_direction``
            # which only runs when ``self.iter > 1`` (and not for the
            # all-or-nothing algorithm, which short-circuits before this method
            # is called). Both conditions are already satisfied by the gate in
            # ``execute()``: ``converged = self.check_convergence() if self.iter > 1 else False``.
        else:
            # Nothing loaded yet, so we are converged only when the AoN solution carries no cost either
            trivially_converged = aon_cost == 0.0
            self.rgap = 0.0 if trivially_converged else np.inf
            return trivially_converged

        if self.rgap_target >= self.rgap:
            return True
        return False
