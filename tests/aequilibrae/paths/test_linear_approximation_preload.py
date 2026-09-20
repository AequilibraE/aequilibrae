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

from types import SimpleNamespace

import numpy as np
import pytest

import aequilibrae.paths.linear_approximation as linear_approximation
from aequilibrae.paths.linear_approximation import LinearApproximation


class DummyVDF:
    def apply_vdf(self, *, congested_time, link_flows, fftime, capacity, cores, offset, scale):
        del capacity, cores
        self.last_link_flows = link_flows.copy()
        congested_time[:] = fftime + scale * link_flows + offset


class DummyDerivativeVDF:
    def __init__(self, derivative):
        self.derivative = derivative

    def apply_derivative(self, *, delta, link_flows, fftime, capacity, cores, **kwargs):
        del link_flows, fftime, capacity, cores, kwargs
        delta[:] = self.derivative


def test_stepsize_derivative_uses_total_flow_state():
    assignment = LinearApproximation.__new__(LinearApproximation)
    assignment.cores = 1
    assignment.elementwise_cores = 1
    assignment.threading_threshold = 10000
    assignment.preload = np.array([10.0, 20.0])
    assignment.current_assigned_flow = np.array([3.0, 4.0])
    assignment.total_flow = assignment.current_assigned_flow + assignment.preload
    assignment.step_direction_flow = np.array([7.0, 8.0])
    assignment.congested_value = np.zeros(2)
    assignment.capacity = np.ones(2)
    assignment.free_flow_tt = np.zeros(2)
    assignment.vdf_parameters = {"scale": 1.0, "offset": 0.0}
    assignment.vdf = DummyVDF()
    assignment.aon_total_turn_cost = 0.0
    assignment.fw_total_turn_cost = 0.0

    stepsize = 0.25
    derivative = assignment._LinearApproximation__derivative_of_objective_stepsize_dependent(stepsize, 0.0)

    candidate_total_flow = assignment.total_flow + stepsize * (assignment.step_direction_flow - assignment.total_flow)
    expected = np.sum(candidate_total_flow * (assignment.step_direction_flow - assignment.total_flow))

    assert np.isclose(derivative, expected)
    np.testing.assert_array_equal(assignment.vdf.last_link_flows, candidate_total_flow)


def test_relative_gap_ignores_constant_preload():
    assignment = LinearApproximation.__new__(LinearApproximation)
    assignment.iteration_issue = []
    assignment.congested_time = np.array([2.0, 3.0])
    assignment.aon_total_turn_cost = 0.0
    assignment.fw_total_turn_cost = 0.0

    cls = SimpleNamespace(
        _id="car",
        fixed_cost=np.array([0.5, 1.5]),
        _aon_results=SimpleNamespace(total_link_loads=np.array([10.0, 1.0])),
        results=SimpleNamespace(total_link_loads=np.array([8.0, 2.0])),
    )
    assignment.traffic_classes = [cls]
    assignment.step_direction = {"car": SimpleNamespace(total_link_loads=np.array([9.0, 1.5]))}

    # Preload contributes to VDF calculations via total_flow but should not affect rgap.
    assignment.preload = np.array([100.0, 100.0])
    assignment.total_flow = cls.results.total_link_loads + assignment.preload
    assignment.rgap_target = 0.1
    assignment.stepsize = 0.1  # not 1.0
    assignment.iter = 1

    assert assignment.check_convergence()

    expected_current_cost = np.sum((assignment.congested_time + cls.fixed_cost) * cls.results.total_link_loads)
    expected_aon_cost = np.sum((assignment.congested_time + cls.fixed_cost) * cls._aon_results.total_link_loads)
    expected_rgap = abs(expected_current_cost - expected_aon_cost) / expected_current_cost

    assert np.isclose(assignment.rgap, expected_rgap)


def test_relative_gap_is_not_converged_for_zero_current_cost_and_nonzero_aon_cost():
    assignment = LinearApproximation.__new__(LinearApproximation)
    assignment.congested_time = np.array([2.0, 3.0])
    assignment.aon_total_turn_cost = 0.0
    assignment.fw_total_turn_cost = 0.0

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


@pytest.mark.parametrize("bad_derivative", [0.0, 1.0])
def test_failed_bfw_direction_retries_with_fw_in_same_iteration(monkeypatch, bad_derivative):
    """Test that a non-descent BFW direction is dropped and the Frank-Wolfe step is searched in the same iteration."""
    assignment = LinearApproximation.__new__(LinearApproximation)
    assignment.algorithm = "bfw"
    assignment.iter = 4
    assignment.rgap = np.inf
    assignment.current_direction = "bfw"
    assignment.next_direction = None
    assignment.iteration_issue = []
    assignment.fw_total_turn_cost = 0.0
    assignment.step_direction_turn_cost = {}
    assignment.betas = np.array([1.0, 0.0, 0.0])

    monkeypatch.setattr(
        assignment,
        "_LinearApproximation__derivative_of_objective_stepsize_independent",
        lambda: 0.0,
    )
    # A non-negative derivative at alpha = 0 is what marks the direction as non-descent.
    monkeypatch.setattr(
        assignment,
        "_LinearApproximation__derivative_of_objective_stepsize_dependent",
        lambda stepsize, const_term=0.0: (bad_derivative if assignment.current_direction == "bfw" else stepsize - 0.25),
    )

    def fake_root_scalar(*_args, **_kwargs):
        assert assignment.current_direction == "fw"
        return SimpleNamespace(root=0.25, converged=True)

    monkeypatch.setattr(linear_approximation, "root_scalar", fake_root_scalar)

    def fake_calculate_step_direction():
        assert assignment.next_direction == "fw"
        assignment.current_direction = "fw"
        assignment.next_direction = "cfw"

    monkeypatch.setattr(assignment, "_LinearApproximation__calculate_step_direction", fake_calculate_step_direction)

    assignment.calculate_stepsize()

    assert assignment.current_direction == "fw"
    assert assignment.next_direction == "cfw"
    assert assignment.stepsize == 0.25
    assert len(assignment.iteration_issue) == 1
    assert assignment.iteration_issue[0].startswith("Found bad conjugate direction step. Performing FW search.")
    np.testing.assert_array_equal(assignment.betas, np.array([1.0, 0.0, 0.0]))


def test_failed_fw_direction_uses_tiny_step_instead_of_recursing(monkeypatch):
    """Test that a failed line search on the Frank-Wolfe direction takes a tiny step rather than recursing."""
    assignment = LinearApproximation.__new__(LinearApproximation)
    assignment.algorithm = "bfw"
    assignment.iter = 5
    assignment.rgap = np.inf
    assignment.current_direction = "fw"
    assignment.next_direction = "cfw"
    assignment.iteration_issue = []
    assignment.fw_total_turn_cost = 0.0
    assignment.step_direction_turn_cost = {}
    assignment.congested_value = np.ones(2)
    assignment.step_direction_flow = np.array([2.0, 3.0])
    assignment.total_flow = np.array([1.0, 1.0])
    assignment.traffic_classes = []

    monkeypatch.setattr(
        assignment,
        "_LinearApproximation__derivative_of_objective_stepsize_independent",
        lambda: 0.0,
    )
    monkeypatch.setattr(
        assignment,
        "_LinearApproximation__derivative_of_objective_stepsize_dependent",
        lambda _stepsize, const_term=0.0: 1.0,
    )
    monkeypatch.setattr(assignment, "_diagnose_negative_gap", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        linear_approximation,
        "root_scalar",
        lambda *_args, **_kwargs: pytest.fail("root_scalar should not be called for a non-descent direction"),
    )

    assignment.calculate_stepsize()

    assert assignment.stepsize == 1e-2 / assignment.iter
    assert assignment.next_direction == "cfw"
    assert assignment.iteration_issue == []


def test_nonconverged_fw_retry_uses_tiny_step(monkeypatch):
    """Test that an unconverged Frank-Wolfe root search is not accepted after retrying a BFW direction."""
    assignment = LinearApproximation.__new__(LinearApproximation)
    assignment.algorithm = "bfw"
    assignment.iter = 4
    assignment.rgap = np.inf
    assignment.current_direction = "bfw"
    assignment.next_direction = None
    assignment.iteration_issue = []
    assignment.fw_total_turn_cost = 0.0
    assignment.step_direction_turn_cost = {}
    assignment.betas = np.array([1.0, 0.0, 0.0])

    monkeypatch.setattr(
        assignment,
        "_LinearApproximation__derivative_of_objective_stepsize_independent",
        lambda: 0.0,
    )
    monkeypatch.setattr(
        assignment,
        "_LinearApproximation__derivative_of_objective_stepsize_dependent",
        lambda stepsize, const_term=0.0: 1.0 if assignment.current_direction == "bfw" else stepsize - 0.25,
    )

    def fake_root_scalar(*_args, **_kwargs):
        assert assignment.current_direction == "fw"
        return SimpleNamespace(root=0.25, converged=False)

    monkeypatch.setattr(linear_approximation, "root_scalar", fake_root_scalar)

    def fake_calculate_step_direction():
        assignment.current_direction = "fw"
        assignment.next_direction = "cfw"

    monkeypatch.setattr(assignment, "_LinearApproximation__calculate_step_direction", fake_calculate_step_direction)

    assignment.calculate_stepsize()

    assert assignment.current_direction == "fw"
    assert assignment.next_direction == "cfw"
    assert assignment.stepsize == 1e-2 / assignment.iter
    assert any("Found bad conjugate direction step" in msg for msg in assignment.iteration_issue)


def test_nonfinite_fw_retry_derivative_uses_tiny_step_instead_of_zero(monkeypatch):
    """Test that a non-finite derivative on the Frank-Wolfe retry still yields a positive step size."""
    assignment = LinearApproximation.__new__(LinearApproximation)
    assignment.algorithm = "bfw"
    assignment.iter = 4
    assignment.rgap = np.inf
    assignment.current_direction = "bfw"
    assignment.next_direction = None
    assignment.iteration_issue = []
    assignment.fw_total_turn_cost = 0.0
    assignment.step_direction_turn_cost = {}
    assignment.betas = np.array([1.0, 0.0, 0.0])
    assignment.congested_value = np.ones(2)
    assignment.step_direction_flow = np.array([2.0, 3.0])
    assignment.total_flow = np.array([1.0, 1.0])
    assignment.traffic_classes = []

    monkeypatch.setattr(
        assignment,
        "_LinearApproximation__derivative_of_objective_stepsize_independent",
        lambda: 0.0,
    )
    monkeypatch.setattr(
        assignment,
        "_LinearApproximation__derivative_of_objective_stepsize_dependent",
        lambda _stepsize, const_term=0.0: 1.0 if assignment.current_direction == "bfw" else np.nan,
    )
    monkeypatch.setattr(
        linear_approximation,
        "root_scalar",
        lambda *_args, **_kwargs: pytest.fail("root_scalar should not be called with a non-finite endpoint"),
    )

    def fake_calculate_step_direction():
        assignment.current_direction = "fw"
        assignment.next_direction = "cfw"

    monkeypatch.setattr(assignment, "_LinearApproximation__calculate_step_direction", fake_calculate_step_direction)

    assignment.calculate_stepsize()

    assert assignment.current_direction == "fw"
    assert assignment.next_direction == "cfw"
    assert assignment.stepsize == 1e-2 / assignment.iter
    assert assignment.stepsize > 0.0
    assert any("Found bad conjugate direction step" in msg for msg in assignment.iteration_issue)


def test_cfw_zero_denominator_falls_back_to_fw():
    assignment = LinearApproximation.__new__(LinearApproximation)
    assignment.cores = 1
    assignment.elementwise_cores = 1
    assignment.threading_threshold = 10_000
    assignment.vdf = DummyDerivativeVDF(np.ones(2))
    assignment.vdf_der = np.zeros(2)
    assignment.total_flow = np.ones(2)
    assignment.capacity = np.ones(2)
    assignment.free_flow_tt = np.ones(2)
    assignment.vdf_parameters = {}
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
    assignment.elementwise_cores = 1
    assignment.threading_threshold = 10_000
    assignment.vdf = DummyDerivativeVDF(np.array([np.nan, 1.0]))
    assignment.vdf_der = np.zeros(2)
    assignment.total_flow = np.ones(2)
    assignment.capacity = np.ones(2)
    assignment.free_flow_tt = np.ones(2)
    assignment.vdf_parameters = {}
    assignment.stepsize = 0.5
    assignment.conjugate_stepsize = 0.5
    assignment.betas = np.array([0.2, 0.3, 0.5])
    assignment.algorithm = "bfw"
    assignment.bfw_conjugacy = "approximate"
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
    assignment.conjugacy_prev = 0.1
    assignment.conjugacy_prev2 = 0.2
    assignment.hessian_drift = 0.3
    assignment.bfw_clamped = True
    assignment.convergence_report = {
        "time": [],
        "iteration": [],
        "rgap": [],
        "warnings": [],
        "alpha": [],
        "beta0": [],
        "beta1": [],
        "beta2": [],
        "conjugacy_prev": [],
        "conjugacy_prev2": [],
        "hessian_drift": [],
        "bfw_clamped": [],
    }
    assignment.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)

    assignment._append_convergence_report(terminal=True)

    assert assignment.convergence_report["iteration"] == [4]
    assert assignment.convergence_report["rgap"] == [0.001]
    assert np.isnan(assignment.convergence_report["alpha"][0])
    assert all(np.isnan(assignment.convergence_report[key][0]) for key in ("beta0", "beta1", "beta2"))
    diagnostics = ("conjugacy_prev", "conjugacy_prev2", "hessian_drift", "bfw_clamped")
    assert all(np.isnan(assignment.convergence_report[key][0]) for key in diagnostics)


# Seeds chosen so the resulting coefficients land strictly inside their clamps; a clamped coefficient would
# make the comparison against the explicit Hessian vacuous. Keyed by class count.
_INTERIOR_SEEDS = {1: 4, 2: 8, 3: 43, 5: 1}


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
    assignment.elementwise_cores = 1
    assignment.threading_threshold = 10_000
    assignment.vdf = DummyDerivativeVDF(vdf_der)
    assignment.vdf_der = np.zeros(num_links)
    assignment.total_flow = np.ones(num_links)
    assignment.capacity = np.ones(num_links)
    assignment.free_flow_tt = np.ones(num_links)
    assignment.vdf_parameters = {}
    assignment.conjugate_direction_max = 0.99999
    assignment.conjugate_stepsize = 0.0
    assignment.betas = np.array([1.0, 0.0, 0.0])
    assignment.iteration_issue = []
    assignment.iter = 4
    assignment.bfw_conjugacy = "approximate"
    assignment.conjugacy_prev = np.nan
    assignment.conjugacy_prev2 = np.nan
    assignment.hessian_drift = np.nan
    assignment.bfw_clamped = np.nan
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


def _bfw_setup(bfw_conjugacy, tau=0.4, seed=None):
    assignment, hessian, agg, flatten = _multiclass_fixture(seed=seed)
    assignment.algorithm = "bfw"
    assignment.bfw_conjugacy = bfw_conjugacy
    assignment.current_direction = "bfw"
    assignment.next_direction = None
    assignment.stepsize = tau
    # A = d_{k-1}, B = d_{k-2}, W = s_{k-2} - s_{k-1}, g = the Frank-Wolfe direction.
    vectors = {
        "A": flatten(agg["step_dir"] - agg["results"]),
        "B": flatten(agg["step_dir"] * tau + agg["prev_step_dir"] * (1.0 - tau) - agg["results"]),
        "W": flatten(agg["prev_step_dir"] - agg["step_dir"]),
        "g": flatten(agg["aon"] - agg["results"]),
    }
    return assignment, hessian, vectors


def _direction(vectors, mu, nu):
    """d_k / beta_0 = g + (nu + mu) A + mu W."""
    return vectors["g"] + (nu + mu) * vectors["A"] + mu * vectors["W"]


def test_exact_bfw_direction_is_conjugate_to_both_previous_directions():
    assignment, hessian, vectors = _bfw_setup("exact")

    assert assignment.calculate_biconjugate_direction()

    mu = assignment.betas[2] / assignment.betas[0]
    nu = assignment.betas[1] / assignment.betas[0]
    # Both coefficients must be interior, otherwise the clamp - not the solve - produced this direction.
    assert mu > 0.0 and nu > 0.0

    d = _direction(vectors, mu, nu)
    norm = np.sqrt((d @ hessian @ d))
    for name in ("A", "B"):
        v = vectors[name]
        residual = abs(v @ hessian @ d) / (np.sqrt(v @ hessian @ v) * norm)
        assert residual < 1e-12, f"exact BFW direction is not conjugate to {name}: {residual:.3e}"


def test_approximate_bfw_direction_is_conjugate_to_neither_previous_direction():
    """Appendix A drops the cross term from *both* conditions, so both residuals stay non-zero."""
    assignment, hessian, vectors = _bfw_setup("approximate")

    assert assignment.calculate_biconjugate_direction()

    mu = assignment.betas[2] / assignment.betas[0]
    nu = assignment.betas[1] / assignment.betas[0]
    assert mu > 0.0 and nu > 0.0  # the clamp did not fire, so this is the approximation talking
    assert assignment.bfw_clamped is False

    d = _direction(vectors, mu, nu)
    norm = np.sqrt(d @ hessian @ d)
    for name in ("A", "B"):
        v = vectors[name]
        residual = abs(v @ hessian @ d) / (np.sqrt(v @ hessian @ v) * norm)
        assert residual > 1e-6, f"approximation should leave a measurable residual against {name}"


def test_exact_and_approximate_bfw_disagree_when_the_dropped_term_is_non_zero():
    """The paths differ only by the dropped cross term, so a non-zero drift must change the weights."""
    approximate, hessian, vectors = _bfw_setup("approximate")
    a, b = vectors["A"], vectors["B"]
    drift = abs(b @ hessian @ a) / np.sqrt((a @ hessian @ a) * (b @ hessian @ b))
    assert drift > 1e-3, "fixture must have a genuinely non-conjugate history for this to mean anything"

    exact, _, _ = _bfw_setup("exact")
    assert exact.calculate_biconjugate_direction()
    assert approximate.calculate_biconjugate_direction()

    assert not np.allclose(exact.betas, approximate.betas)
    assert np.isclose(exact.betas.sum(), 1.0) and np.isclose(approximate.betas.sum(), 1.0)


def test_exact_bfw_records_conjugacy_diagnostics():
    assignment, _, _ = _bfw_setup("exact")

    assert assignment.calculate_biconjugate_direction()

    assert assignment.bfw_clamped is False
    assert abs(assignment.conjugacy_prev) < 1e-12
    assert abs(assignment.conjugacy_prev2) < 1e-12
    # hessian_drift is the term the approximation drops; it must be non-trivial for this fixture.
    assert np.isfinite(assignment.hessian_drift)
    assert abs(assignment.hessian_drift) > 1e-6


def test_approximate_bfw_records_nonzero_conjugacy_residuals():
    assignment, _, _ = _bfw_setup("approximate")

    assert assignment.calculate_biconjugate_direction()

    assert abs(assignment.conjugacy_prev) > 1e-6
    assert abs(assignment.conjugacy_prev2) > 1e-6


def test_exact_bfw_falls_back_to_the_simplex_face_when_a_coefficient_goes_negative():
    """Seed 24 drives mu negative; the face fallback must stay conjugate to the direction it keeps."""
    assignment, hessian, vectors = _bfw_setup("exact", seed=24)

    assert assignment.calculate_biconjugate_direction()

    assert assignment.bfw_clamped is True
    assert assignment.betas[2] == 0.0  # weight on d_{k-2} dropped entirely
    mu = assignment.betas[2] / assignment.betas[0]
    nu = assignment.betas[1] / assignment.betas[0]
    d = _direction(vectors, mu, nu)
    a = vectors["A"]
    residual = abs(a @ hessian @ d) / (np.sqrt(a @ hessian @ a) * np.sqrt(d @ hessian @ d))
    assert residual < 1e-12, "the retained direction must still be conjugate after dropping the other"


def test_singular_exact_bfw_system_falls_back_to_fw():
    """When d_{k-1} and d_{k-2} coincide there is no direction conjugate to both."""
    assignment, _, _ = _bfw_setup("exact")
    # stepsize == 0 makes x_ (d_{k-2}) equal psd - ll while z_ stays sd - ll; force them equal instead by
    # pointing the previous step direction at the current one.
    for cid, sdr in assignment.step_direction.items():
        assignment.previous_step_direction[cid] = SimpleNamespace(link_loads=sdr.link_loads)

    assert not assignment.calculate_biconjugate_direction()
    assert assignment.current_direction == "fw"
    np.testing.assert_array_equal(assignment.betas, np.array([1.0, 0.0, 0.0]))
    assert assignment.iteration_issue == ["Singular exact BFW system; using the Frank-Wolfe direction."]
