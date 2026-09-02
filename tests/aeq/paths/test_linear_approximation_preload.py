# ---------------------------------------------------------------------------------------------------------------------
# Portions of this file were contributed by Lim Junmin and are
# retained under the license below: the MIT License (with added clause) under which it was
# contributed to AequilibraE. See LICENSE.TXT.
#
# Copyright (c) 2026 Lim Junmin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute and/or sublicense
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Additional clause:
#
# Reference to the software has to be made in all documentation for
# work developed with the software.
# ---------------------------------------------------------------------------------------------------------------------

import numpy as np
import pytest
from types import SimpleNamespace

import aequilibrae.paths.linear_approximation as linear_approximation
from aequilibrae.paths.linear_approximation import LinearApproximation


class DummyVDF:
    def __init__(self):
        self.last_link_flows = None

    def apply_vdf(self, congested_time, link_flows, capacity, fftime, scale, offset, cores):
        del capacity, cores
        self.last_link_flows = link_flows.copy()
        congested_time[:] = fftime + scale * link_flows + offset


class DummyDerivativeVDF:
    def __init__(self, derivative):
        self.derivative = derivative

    def apply_derivative(self, output, *_args):
        output[:] = self.derivative


def test_stepsize_derivative_uses_fw_total_flow_state():
    assignment = LinearApproximation.__new__(LinearApproximation)
    assignment.cores = 1
    assignment.preload = np.array([10.0, 20.0])
    assignment.current_assigned_flow = np.array([3.0, 4.0])
    assignment.fw_total_flow = assignment.current_assigned_flow + assignment.preload
    assigned_direction = np.array([7.0, 8.0])
    assignment.step_direction_flow = assigned_direction + assignment.preload
    assignment.congested_value = np.zeros(2)
    assignment.capacity = np.ones(2)
    assignment.free_flow_tt = np.zeros(2)
    assignment.vdf_parameters = [1.0, 0.0]
    assignment.vdf = DummyVDF()

    stepsize = 0.25
    derivative = assignment._LinearApproximation__derivative_of_objective_stepsize_dependent(stepsize, 0.0)

    candidate_total_flow = (
        assignment.preload
        + assignment.current_assigned_flow
        + stepsize * (assigned_direction - assignment.current_assigned_flow)
    )
    expected = np.sum(candidate_total_flow * (assignment.step_direction_flow - assignment.fw_total_flow))

    assert np.isclose(derivative, expected)
    np.testing.assert_array_equal(assignment.vdf.last_link_flows, candidate_total_flow)


@pytest.mark.parametrize("stepsize", [0.0, 0.25, 1.0])
def test_trapezoidal_stepsize_keeps_constant_preload(stepsize):
    assignment = LinearApproximation.__new__(LinearApproximation)
    assignment.cores = 1
    assignment.preload = np.array([10.0, 20.0])
    current_assigned_flow = np.array([3.0, 4.0])
    assigned_direction = np.array([7.0, 8.0])
    assignment.fw_total_flow = current_assigned_flow + assignment.preload
    assignment.step_direction_flow = assigned_direction + assignment.preload
    assignment.congested_time = np.zeros(2)
    assignment._trap_new_flow = np.zeros(2)
    assignment._trap_new_cost = np.zeros(2)
    assignment._trap_avg_cost = np.zeros(2)
    assignment.capacity = np.ones(2)
    assignment.free_flow_tt = np.zeros(2)
    assignment.vdf_parameters = [1.0, 0.0]
    assignment.vdf = DummyVDF()

    assignment._LinearApproximation__objective_change_at_stepsize(0.0, stepsize)

    expected = assignment.preload + current_assigned_flow + stepsize * (assigned_direction - current_assigned_flow)
    np.testing.assert_array_equal(assignment.vdf.last_link_flows, expected)


def test_relative_gap_ignores_constant_preload():
    assignment = LinearApproximation.__new__(LinearApproximation)
    assignment.congested_time = np.array([2.0, 3.0])

    cls = SimpleNamespace(
        _id="car",
        fixed_cost=np.array([0.5, 1.5]),
        _aon_results=SimpleNamespace(total_link_loads=np.array([10.0, 1.0])),
        results=SimpleNamespace(total_link_loads=np.array([8.0, 2.0])),
    )
    assignment.traffic_classes = [cls]
    assignment.step_direction = {"car": SimpleNamespace(total_link_loads=np.array([9.0, 1.5]))}

    # Preload contributes to VDF calculations via fw_total_flow but should not affect rgap.
    assignment.preload = np.array([100.0, 100.0])
    assignment.fw_total_flow = cls.results.total_link_loads + assignment.preload
    assignment.rgap_target = 0.1
    assignment.stepsize = 0.1  # not 1.0

    assert assignment.check_convergence()

    expected_current_cost = np.sum((assignment.congested_time + cls.fixed_cost) * cls.results.total_link_loads)
    expected_aon_cost = np.sum((assignment.congested_time + cls.fixed_cost) * cls._aon_results.total_link_loads)
    expected_rgap = abs(expected_current_cost - expected_aon_cost) / expected_current_cost

    assert np.isclose(assignment.rgap, expected_rgap)


def test_relative_gap_is_not_converged_for_zero_current_cost_and_nonzero_aon_cost():
    assignment = LinearApproximation.__new__(LinearApproximation)
    assignment.congested_time = np.array([2.0, 3.0])

    cls = SimpleNamespace(
        _id="car",
        fixed_cost=np.zeros(2),
        _aon_results=SimpleNamespace(total_link_loads=np.array([10.0, 1.0])),
        results=SimpleNamespace(total_link_loads=np.zeros(2)),
    )
    assignment.traffic_classes = [cls]
    assignment.step_direction = {"car": SimpleNamespace(total_link_loads=np.zeros(2))}

    assignment.rgap_target = 0.1
    assignment.stepsize = 0.1  # not 1.0

    assert not assignment.check_convergence()
    assert np.isinf(assignment.rgap)


def test_failed_bfw_direction_retries_with_fw_in_same_iteration(monkeypatch):
    assignment = LinearApproximation.__new__(LinearApproximation)
    assignment.algorithm = "bfw"
    assignment.line_search = "trapezoidal"
    assignment.iter = 4
    assignment.rgap = np.inf
    assignment.current_direction = "bfw"
    assignment.next_direction = None
    assignment.iteration_issue = []
    assignment.logger = SimpleNamespace(warning=lambda *_args, **_kwargs: None, debug=lambda *_args, **_kwargs: None)
    assignment.betas = np.array([1.0, 0.0, 0.0])

    monkeypatch.setattr(
        assignment,
        "_LinearApproximation__derivative_of_objective_stepsize_independent",
        lambda: 0.0,
    )

    monkeypatch.setattr(
        assignment,
        "_LinearApproximation__objective_change_at_stepsize",
        lambda _const, _alpha: 1.0 if assignment.current_direction == "bfw" else -0.5,
    )

    def fake_minimize_scalar(*_args, **_kwargs):
        if assignment.current_direction == "bfw":
            return SimpleNamespace(x=0.3, fun=1.0)
        return SimpleNamespace(x=0.25, fun=-0.5)

    monkeypatch.setattr(linear_approximation, "minimize_scalar", fake_minimize_scalar)

    def fake_calculate_step_direction():
        assert assignment.next_direction == "fw"
        assignment.current_direction = "fw"
        assignment.next_direction = "cfw"

    monkeypatch.setattr(assignment, "_LinearApproximation__calculate_step_direction", fake_calculate_step_direction)

    assignment.calculate_stepsize()

    assert assignment.current_direction == "fw"
    assert assignment.next_direction == "cfw"
    assert assignment.stepsize == 0.25
    assert assignment.iteration_issue == ["BFW/CFW direction yielded no improvement; falling back to FW."]
    np.testing.assert_array_equal(assignment.betas, np.array([1.0, 0.0, 0.0]))


def test_failed_fw_direction_uses_tiny_step_instead_of_recursing(monkeypatch):
    assignment = LinearApproximation.__new__(LinearApproximation)
    assignment.algorithm = "bfw"
    assignment.line_search = "trapezoidal"
    assignment.iter = 5
    assignment.rgap = np.inf
    assignment.current_direction = "fw"
    assignment.next_direction = "cfw"
    assignment.iteration_issue = []
    assignment.logger = SimpleNamespace(warning=lambda *_args, **_kwargs: None, debug=lambda *_args, **_kwargs: None)

    monkeypatch.setattr(
        assignment,
        "_LinearApproximation__derivative_of_objective_stepsize_independent",
        lambda: 0.0,
    )
    monkeypatch.setattr(
        assignment,
        "_LinearApproximation__objective_change_at_stepsize",
        lambda _const, _alpha: 1.0,
    )
    monkeypatch.setattr(
        linear_approximation,
        "minimize_scalar",
        lambda *_args, **_kwargs: SimpleNamespace(x=0.3, fun=1.0),
    )

    assignment.calculate_stepsize()

    assert assignment.stepsize == 1e-2 / assignment.iter
    assert assignment.next_direction == "cfw"
    assert assignment.iteration_issue == []


def test_failed_bfw_direction_clips_retry_stepsize_to_alpha_max(monkeypatch):
    assignment = LinearApproximation.__new__(LinearApproximation)
    assignment.algorithm = "bfw"
    assignment.line_search = "trapezoidal"
    assignment.iter = 4
    assignment.rgap = np.inf
    assignment.current_direction = "bfw"
    assignment.next_direction = None
    assignment.iteration_issue = []
    assignment.logger = SimpleNamespace(warning=lambda *_args, **_kwargs: None, debug=lambda *_args, **_kwargs: None)
    assignment.betas = np.array([1.0, 0.0, 0.0])

    monkeypatch.setattr(
        assignment,
        "_LinearApproximation__derivative_of_objective_stepsize_independent",
        lambda: 0.0,
    )

    monkeypatch.setattr(
        assignment,
        "_LinearApproximation__objective_change_at_stepsize",
        lambda _const, _alpha: 1.0 if assignment.current_direction == "bfw" else -0.5,
    )

    def fake_minimize_scalar(*_args, **_kwargs):
        if assignment.current_direction == "bfw":
            return SimpleNamespace(x=0.3, fun=1.0)
        return SimpleNamespace(x=1.25, fun=-0.5)

    monkeypatch.setattr(linear_approximation, "minimize_scalar", fake_minimize_scalar)

    def fake_calculate_step_direction():
        assignment.current_direction = "fw"
        assignment.next_direction = "cfw"

    monkeypatch.setattr(assignment, "_LinearApproximation__calculate_step_direction", fake_calculate_step_direction)

    assignment.calculate_stepsize()

    assert assignment.current_direction == "fw"
    assert assignment.next_direction == "cfw"
    assert assignment.stepsize == 0.5
    assert any("clipping to 0.5" in msg for msg in assignment.iteration_issue)


def test_nonfinite_fw_retry_stepsize_uses_tiny_step_instead_of_zero(monkeypatch):
    assignment = LinearApproximation.__new__(LinearApproximation)
    assignment.algorithm = "bfw"
    assignment.line_search = "trapezoidal"
    assignment.iter = 4
    assignment.rgap = np.inf
    assignment.current_direction = "bfw"
    assignment.next_direction = None
    assignment.iteration_issue = []
    assignment.logger = SimpleNamespace(warning=lambda *_args, **_kwargs: None, debug=lambda *_args, **_kwargs: None)
    assignment.betas = np.array([1.0, 0.0, 0.0])

    monkeypatch.setattr(
        assignment,
        "_LinearApproximation__derivative_of_objective_stepsize_independent",
        lambda: 0.0,
    )
    monkeypatch.setattr(
        assignment,
        "_LinearApproximation__objective_change_at_stepsize",
        lambda _const, _alpha: 1.0 if assignment.current_direction == "bfw" else -0.5,
    )

    def fake_minimize_scalar(*_args, **_kwargs):
        if assignment.current_direction == "bfw":
            return SimpleNamespace(x=0.3, fun=1.0)
        return SimpleNamespace(x=np.nan, fun=-0.5)

    monkeypatch.setattr(linear_approximation, "minimize_scalar", fake_minimize_scalar)

    def fake_calculate_step_direction():
        assignment.current_direction = "fw"
        assignment.next_direction = "cfw"

    monkeypatch.setattr(assignment, "_LinearApproximation__calculate_step_direction", fake_calculate_step_direction)

    assignment.calculate_stepsize()

    assert assignment.current_direction == "fw"
    assert assignment.next_direction == "cfw"
    assert assignment.stepsize == 1e-2 / assignment.iter
    assert assignment.stepsize > 0.0
    assert any("invalid stepsize" in msg for msg in assignment.iteration_issue)


def test_cfw_zero_denominator_falls_back_to_fw():
    assignment = LinearApproximation.__new__(LinearApproximation)
    assignment.cores = 1
    assignment.vdf = DummyDerivativeVDF(np.ones(2))
    assignment.vdf_der = np.zeros(2)
    assignment.fw_total_flow = np.ones(2)
    assignment.capacity = np.ones(2)
    assignment.free_flow_tt = np.ones(2)
    assignment.vdf_parameters = []
    assignment.conjugate_direction_max = 0.99999
    assignment.conjugate_stepsize = 0.5
    assignment.betas = np.array([0.5, 0.5, 0.0])
    assignment.algorithm = "cfw"
    assignment.current_direction = "cfw"
    assignment.next_direction = None
    assignment.iteration_issue = []
    assignment.logger = SimpleNamespace(debug=lambda *_args, **_kwargs: None)

    cls = SimpleNamespace(
        _id="car",
        results=SimpleNamespace(link_loads=np.array([[1.0], [2.0]])),
        _aon_results=SimpleNamespace(link_loads=np.array([[2.0], [3.0]])),
    )
    assignment.traffic_classes = [cls]
    assignment.step_direction = {
        "car": SimpleNamespace(link_loads=np.array([[1.0], [2.0]])),
    }

    assignment.calculate_conjugate_stepsize()

    assert assignment.current_direction == "fw"
    assert assignment.conjugate_stepsize == 0.0
    np.testing.assert_array_equal(assignment.betas, np.array([1.0, 0.0, 0.0]))
    assert assignment.iteration_issue == ["Invalid CFW coefficient; using the Frank-Wolfe direction."]


def test_bfw_nonfinite_coefficient_falls_back_to_fw():
    assignment = LinearApproximation.__new__(LinearApproximation)
    assignment.cores = 1
    assignment.vdf = DummyDerivativeVDF(np.array([np.nan, 1.0]))
    assignment.vdf_der = np.zeros(2)
    assignment.fw_total_flow = np.ones(2)
    assignment.capacity = np.ones(2)
    assignment.free_flow_tt = np.ones(2)
    assignment.vdf_parameters = []
    assignment.stepsize = 0.5
    assignment.conjugate_stepsize = 0.5
    assignment.betas = np.array([0.2, 0.3, 0.5])
    assignment.algorithm = "bfw"
    assignment.current_direction = "bfw"
    assignment.next_direction = None
    assignment.iteration_issue = []
    assignment.logger = SimpleNamespace(debug=lambda *_args, **_kwargs: None)

    cls = SimpleNamespace(
        _id="car",
        results=SimpleNamespace(link_loads=np.array([[1.0], [2.0]])),
        _aon_results=SimpleNamespace(link_loads=np.array([[2.0], [3.0]])),
    )
    assignment.traffic_classes = [cls]
    assignment.step_direction = {
        "car": SimpleNamespace(link_loads=np.array([[3.0], [5.0]])),
    }
    assignment.previous_step_direction = {
        "car": SimpleNamespace(link_loads=np.array([[4.0], [7.0]])),
    }

    assignment.calculate_biconjugate_direction()

    assert assignment.current_direction == "fw"
    assert assignment.next_direction == "cfw"
    np.testing.assert_array_equal(assignment.betas, np.array([1.0, 0.0, 0.0]))
    assert assignment.iteration_issue == ["Invalid BFW mu coefficient; using the Frank-Wolfe direction."]


def test_append_terminal_convergence_report_uses_nan_direction_coefficients():
    assignment = LinearApproximation.__new__(LinearApproximation)
    assignment._LinearApproximation__start_time = 0.0
    assignment.iter = 4
    assignment.rgap = 0.001
    assignment.stepsize = 0.25
    assignment.betas = np.array([0.2, 0.3, 0.5])
    assignment.algorithm = "bfw"
    assignment.iteration_issue = []
    assignment.convergence_report = {
        "time": [],
        "iteration": [],
        "rgap": [],
        "warnings": [],
        "alpha": [],
        "beta0": [],
        "beta1": [],
        "beta2": [],
    }
    assignment.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)

    assignment._append_convergence_report(terminal=True)

    assert assignment.convergence_report["iteration"] == [4]
    assert assignment.convergence_report["rgap"] == [0.001]
    assert np.isnan(assignment.convergence_report["alpha"][0])
    assert all(np.isnan(assignment.convergence_report[key][0]) for key in ("beta0", "beta1", "beta2"))


# Seeds chosen so the resulting coefficients land strictly inside their clamps; a clamped coefficient would
# make the comparison against the explicit Hessian vacuous. Keyed by class count.
_INTERIOR_SEEDS = {1: 4, 2: 8, 3: 24, 5: 1}


def _multiclass_fixture(num_links=4, num_classes=3, num_cores=2, seed=None):
    """Random multi-class state plus the explicitly assembled block Hessian it implies.

    Each link's Hessian block is ``H_a = t'_a * ones(M, M)``, so the full Hessian is block diagonal over links.
    Vectors are flattened link-major/class-minor to match that layout.
    """
    seed = _INTERIOR_SEEDS[num_classes] if seed is None else seed
    rng = np.random.default_rng(seed)
    vdf_der = rng.uniform(0.5, 2.0, size=num_links)
    loads = {
        name: rng.uniform(1.0, 10.0, size=(num_classes, num_links, num_cores))
        for name in ("results", "aon", "step_dir", "prev_step_dir")
    }

    hessian = np.zeros((num_links * num_classes, num_links * num_classes))
    for a in range(num_links):
        block = slice(a * num_classes, (a + 1) * num_classes)
        hessian[block, block] = vdf_der[a] * np.ones((num_classes, num_classes))

    def flatten(per_class_per_link):
        """(M, L) class/link array -> length L*M vector ordered link-major, matching ``hessian``."""
        return per_class_per_link.T.ravel()

    # Sum over matrix cores, as the implementation does before contracting.
    aggregated = {name: value.sum(axis=2) for name, value in loads.items()}

    classes = []
    step_direction = {}
    previous_step_direction = {}
    for m in range(num_classes):
        cid = f"class_{m}"
        classes.append(
            SimpleNamespace(
                _id=cid,
                results=SimpleNamespace(link_loads=loads["results"][m]),
                _aon_results=SimpleNamespace(link_loads=loads["aon"][m]),
            )
        )
        step_direction[cid] = SimpleNamespace(link_loads=loads["step_dir"][m])
        previous_step_direction[cid] = SimpleNamespace(link_loads=loads["prev_step_dir"][m])

    assignment = LinearApproximation.__new__(LinearApproximation)
    assignment.cores = 1
    assignment.vdf = DummyDerivativeVDF(vdf_der)
    assignment.vdf_der = np.zeros(num_links)
    assignment.fw_total_flow = np.ones(num_links)
    assignment.capacity = np.ones(num_links)
    assignment.free_flow_tt = np.ones(num_links)
    assignment.vdf_parameters = []
    assignment.conjugate_direction_max = 0.99999
    assignment.conjugate_stepsize = 0.0
    assignment.betas = np.array([1.0, 0.0, 0.0])
    assignment.iteration_issue = []
    assignment.logger = SimpleNamespace(debug=lambda *_args, **_kwargs: None)
    assignment.traffic_classes = classes
    assignment.step_direction = step_direction
    assignment.previous_step_direction = previous_step_direction

    return assignment, hessian, aggregated, flatten


def test_cfw_coefficient_matches_explicit_block_hessian():
    assignment, hessian, agg, flatten = _multiclass_fixture()
    assignment.algorithm = "cfw"
    assignment.current_direction = "cfw"
    assignment.next_direction = None

    # u = s_{k-1} - x_k, v = y_k - x_k, w = y_k - s_{k-1}
    u = flatten(agg["step_dir"] - agg["results"])
    v = flatten(agg["aon"] - agg["results"])
    w = flatten(agg["aon"] - agg["step_dir"])
    expected_alpha = (u @ hessian @ v) / (u @ hessian @ w)

    assert assignment.calculate_conjugate_stepsize()

    # Guard against the assertion passing only because the coefficient was clamped at a bound.
    assert 0.0 < expected_alpha < assignment.conjugate_direction_max
    assert np.isclose(assignment.conjugate_stepsize, expected_alpha, rtol=1e-12, atol=0.0)
    assert np.isclose(assignment.betas[1], expected_alpha, rtol=1e-12, atol=0.0)
    assert assignment.iteration_issue == []


def test_bfw_coefficients_match_explicit_block_hessian():
    assignment, hessian, agg, flatten = _multiclass_fixture()
    assignment.algorithm = "bfw"
    assignment.current_direction = "bfw"
    assignment.next_direction = None
    assignment.stepsize = 0.4

    tau = assignment.stepsize
    # Appendix A notation: x_ is the residual direction d_{k-2}, z_ is d_{k-1}, y_ is the FW direction.
    x_ = flatten(agg["step_dir"] * tau + agg["prev_step_dir"] * (1.0 - tau) - agg["results"])
    y_ = flatten(agg["aon"] - agg["results"])
    z_ = flatten(agg["step_dir"] - agg["results"])
    w_ = flatten(agg["prev_step_dir"] - agg["step_dir"])

    expected_mu = max(0.0, -(x_ @ hessian @ y_) / (x_ @ hessian @ w_))
    expected_nu = max(0.0, -(z_ @ hessian @ y_) / (z_ @ hessian @ z_) + expected_mu * tau / (1.0 - tau))
    expected_beta0 = 1.0 / (1.0 + expected_nu + expected_mu)
    expected = np.array([expected_beta0, expected_nu * expected_beta0, expected_mu * expected_beta0])

    assert assignment.calculate_biconjugate_direction()

    # Guard against the assertion passing only because both coefficients were clamped to zero.
    assert expected_mu > 0.0 and expected_nu > 0.0
    np.testing.assert_allclose(assignment.betas, expected, rtol=1e-12, atol=0.0)
    assert np.isclose(assignment.betas.sum(), 1.0)
    assert assignment.iteration_issue == []


@pytest.mark.parametrize("num_classes", [1, 2, 5])
def test_contractions_match_block_hessian_for_any_class_count(num_classes):
    """The class-pair factorization must hold for any number of classes, not just the default fixture."""
    assignment, hessian, agg, flatten = _multiclass_fixture(num_classes=num_classes)
    assignment.algorithm = "cfw"
    assignment.current_direction = "cfw"
    assignment.next_direction = None

    u = flatten(agg["step_dir"] - agg["results"])
    v = flatten(agg["aon"] - agg["results"])
    w = flatten(agg["aon"] - agg["step_dir"])
    expected_alpha = (u @ hessian @ v) / (u @ hessian @ w)

    assert assignment.calculate_conjugate_stepsize()
    assert 0.0 < expected_alpha < assignment.conjugate_direction_max
    assert np.isclose(assignment.conjugate_stepsize, expected_alpha, rtol=1e-12, atol=0.0)
