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

import aequilibrae.paths.linear_approximation as linear_approximation
from aequilibrae.paths.linear_approximation import LinearApproximation


class DummyVDF:
    def apply_vdf(self, congested_time, link_flows, fftime, cores, offset, scale, capacity):
        del capacity, cores
        congested_time[:] = fftime + scale * link_flows + offset


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
    capacity = np.ones(2)
    assignment.free_flow_tt = np.zeros(2)
    assignment.vdf_parameters = {"scale": 1.0, "offset": 0.0, "capacity": capacity}
    assignment.vdf = DummyVDF()

    stepsize = 0.25
    derivative = assignment._LinearApproximation__derivative_of_objective_stepsize_dependent(stepsize, 0.0)

    candidate_total_flow = assignment.total_flow + stepsize * (assignment.step_direction_flow - assignment.total_flow)
    expected = np.sum(candidate_total_flow * (assignment.step_direction_flow - assignment.total_flow))

    assert np.isclose(derivative, expected)


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

    # Preload contributes to VDF calculations via total_flow but should not affect rgap.
    assignment.preload = np.array([100.0, 100.0])
    assignment.total_flow = cls.results.total_link_loads + assignment.preload
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
    np.testing.assert_array_equal(assignment.betas, np.array([-1.0, -1.0, -1.0]))


def test_failed_fw_direction_uses_tiny_step_instead_of_recursing(monkeypatch):
    assignment = LinearApproximation.__new__(LinearApproximation)
    assignment.algorithm = "bfw"
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
