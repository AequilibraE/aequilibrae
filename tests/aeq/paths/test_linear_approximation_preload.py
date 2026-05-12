import numpy as np

from aequilibrae.paths.linear_approximation import LinearApproximation


class DummyVDF:
    def apply_vdf(self, congested_time, link_flows, capacity, fftime, scale, offset, cores):
        del capacity, cores
        congested_time[:] = fftime + scale * link_flows + offset


def test_stepsize_derivative_treats_preload_as_constant_background_flow():
    assignment = LinearApproximation.__new__(LinearApproximation)
    assignment.cores = 1
    assignment.preload = np.array([10.0, 20.0])
    assignment.current_assigned_flow = np.array([3.0, 4.0])
    assignment.fw_total_flow = assignment.current_assigned_flow + assignment.preload
    assignment.step_direction_flow = np.array([7.0, 8.0])
    assignment.congested_value = np.zeros(2)
    assignment.capacity = np.ones(2)
    assignment.free_flow_tt = np.zeros(2)
    assignment.vdf_parameters = [1.0, 0.0]
    assignment.vdf = DummyVDF()

    stepsize = 0.25
    derivative = assignment._LinearApproximation__derivative_of_objective_stepsize_dependent(stepsize, 0.0)

    candidate_assigned_flow = (
        assignment.step_direction_flow * stepsize + assignment.current_assigned_flow * (1.0 - stepsize)
    )
    candidate_total_flow = candidate_assigned_flow + assignment.preload
    expected = np.sum(candidate_total_flow * (assignment.step_direction_flow - assignment.current_assigned_flow))

    assert np.isclose(derivative, expected)


def test_relative_gap_ignores_constant_preload():
    assignment = LinearApproximation.__new__(LinearApproximation)
    assignment.stepsize_has_been_reset = False
    assignment.congested_time = np.array([2.0, 3.0])
    assignment.aon_total_flow = np.array([10.0, 1.0])
    assignment.current_assigned_flow = np.array([8.0, 2.0])
    assignment.fw_total_flow = assignment.current_assigned_flow + np.array([100.0, 100.0])
    assignment.rgap_target = 0.1

    assert assignment.check_convergence()

    expected_current_cost = np.sum(assignment.congested_time * assignment.current_assigned_flow)
    expected_aon_cost = np.sum(assignment.congested_time * assignment.aon_total_flow)
    expected_rgap = abs(expected_current_cost - expected_aon_cost) / expected_current_cost

    assert np.isclose(assignment.rgap, expected_rgap)


def test_relative_gap_is_not_converged_for_zero_current_cost_and_nonzero_aon_cost():
    assignment = LinearApproximation.__new__(LinearApproximation)
    assignment.stepsize_has_been_reset = False
    assignment.congested_time = np.array([2.0, 3.0])
    assignment.aon_total_flow = np.array([10.0, 1.0])
    assignment.current_assigned_flow = np.zeros(2)
    assignment.fw_total_flow = np.array([100.0, 100.0])
    assignment.rgap_target = 0.1

    assert not assignment.check_convergence()
    assert np.isinf(assignment.rgap)
