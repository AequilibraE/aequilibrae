import numpy as np
from types import SimpleNamespace

from aequilibrae.paths.linear_approximation import LinearApproximation


class DummyVDF:
    def apply_vdf(self, congested_time, link_flows, capacity, fftime, scale, offset, cores):
        del capacity, cores
        congested_time[:] = fftime + scale * link_flows + offset


def test_stepsize_derivative_uses_total_flow_state():
    assignment = LinearApproximation.__new__(LinearApproximation)
    assignment.cores = 1
    assignment.preload = np.array([10.0, 20.0])
    assignment.current_assigned_flow = np.array([3.0, 4.0])
    assignment.total_flow = assignment.current_assigned_flow + assignment.preload
    assignment.step_direction_flow = np.array([7.0, 8.0])
    assignment.congested_value = np.zeros(2)
    assignment.capacity = np.ones(2)
    assignment.free_flow_tt = np.zeros(2)
    assignment.vdf_parameters = [1.0, 0.0]
    assignment.vdf = DummyVDF()

    stepsize = 0.25
    derivative = assignment._LinearApproximation__derivative_of_objective_stepsize_dependent(stepsize, 0.0)

    candidate_total_flow = assignment.total_flow + stepsize * (assignment.step_direction_flow - assignment.total_flow)
    expected = np.sum(candidate_total_flow * (assignment.step_direction_flow - assignment.total_flow))

    assert np.isclose(derivative, expected)


def test_relative_gap_ignores_constant_preload():
    assignment = LinearApproximation.__new__(LinearApproximation)
    assignment.stepsize_has_been_reset = False
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

    assert assignment.check_convergence()

    expected_current_cost = np.sum((assignment.congested_time + cls.fixed_cost) * cls.results.total_link_loads)
    expected_aon_cost = np.sum((assignment.congested_time + cls.fixed_cost) * cls._aon_results.total_link_loads)
    expected_rgap = abs(expected_current_cost - expected_aon_cost) / expected_current_cost

    assert np.isclose(assignment.rgap, expected_rgap)


def test_relative_gap_is_not_converged_for_zero_current_cost_and_nonzero_aon_cost():
    assignment = LinearApproximation.__new__(LinearApproximation)
    assignment.stepsize_has_been_reset = False
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

    assert not assignment.check_convergence()
    assert np.isinf(assignment.rgap)
