import numpy as np
import pandas as pd
import pytest

from aequilibrae.distribution import GravityApplication, GravityCalibration, SyntheticGravityModel
from aequilibrae.matrix import AequilibraeMatrix

ZONES = 80


@pytest.fixture(scope="module")
def zones():
    rng = np.random.default_rng(7)
    coords = rng.random((ZONES, 2)) * 50
    cost = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)) + 1
    np.fill_diagonal(cost, 2)  # positive, as the power and gamma functions require
    productions, attractions = rng.random(ZONES) * 1000, rng.random(ZONES) * 1000
    return cost, productions, attractions * productions.sum() / attractions.sum()


def matrix(values, name):
    mat = AequilibraeMatrix(index=np.arange(1, ZONES + 1))
    mat[name] = values
    return mat


def gravity_model(function, alpha=None, beta=None):
    model = SyntheticGravityModel()
    model.function, model.alpha, model.beta = function, alpha, beta
    return model


def model_trips(zones, model):
    cost, productions, attractions = zones
    parameters = {"max trip length": -1, "max iterations": 5000, "convergence level": 1e-10, "balancing tolerance": 1}
    gravity = GravityApplication(
        impedance=matrix(cost, "cost"),
        vectors=pd.DataFrame({"rows": productions, "columns": attractions}, index=np.arange(1, ZONES + 1)),
        row_field="rows",
        column_field="columns",
        model=model,
        parameters=parameters,
    )
    gravity.apply()
    return gravity.output["gravity"]


@pytest.mark.parametrize("function, alpha, beta", [("EXPO", None, 0.08), ("POWER", 1.3, None), ("GAMMA", -0.6, 0.05)])
def test_calibration_reproduces_observed_statistics(zones, function, alpha, beta):
    """Test that calibrating on noisy trips of a gravity model reproduces their statistics and nears its parameters."""
    cost = zones[0]
    observed = np.random.default_rng(11).poisson(model_trips(zones, gravity_model(function, alpha, beta)))
    calibration = GravityCalibration(matrix=matrix(observed, "trips"), impedance=matrix(cost, "c"), function=function)
    calibration.calibrate()
    modelled = calibration.result_matrix["gravity"]

    assert calibration.gap < calibration.parameters["max error"]
    assert calibration.model.alpha == (None if alpha is None else pytest.approx(alpha, rel=0.05))
    assert calibration.model.beta == (None if beta is None else pytest.approx(beta, rel=0.05))
    np.testing.assert_allclose(modelled.sum(axis=1), observed.sum(axis=1), rtol=1e-6)
    np.testing.assert_allclose(modelled.sum(axis=0), observed.sum(axis=0), rtol=1e-6)
    for statistic in {"EXPO": [cost], "POWER": [np.log(cost)], "GAMMA": [cost, np.log(cost)]}[function]:
        assert (modelled * statistic).sum() == pytest.approx((observed * statistic).sum(), rel=1e-6)


def test_calibration_leaves_out_unusable_cells(zones):
    """Test that trips with zero cost, or above the max trip length, are left out of a gamma calibration."""
    cost = zones[0].copy()
    np.fill_diagonal(cost, 0)
    max_cost = np.quantile(cost, 0.9)
    trips = model_trips(zones, gravity_model("GAMMA", -0.6, 0.05))

    parameters = {"max trip length": max_cost, "max iterations": 100, "max error": 1e-4}
    calibration = GravityCalibration(
        matrix=matrix(trips, "trips"), impedance=matrix(cost, "cost"), function="GAMMA", parameters=parameters
    )
    calibration.calibrate()

    assert calibration.gap < 1e-4
    assert not calibration.result_matrix["gravity"][(cost == 0) | (cost >= max_cost)].any()
    assert "were not used" in "\n".join(calibration.report)


def test_save_and_load(tmp_path):
    model_filename = tmp_path / "aequilibrae_model_test.mod"

    model = SyntheticGravityModel()
    model.function = "EXPO"
    model.beta = 0.1
    assert model.function == "EXPO"  # Did we save the value?

    model.function = "POWER"
    # Check if we zeroed the parameters when changing the function
    assert model.beta is None
    model.alpha = 0.1

    model.save(model_filename)

    model2 = SyntheticGravityModel()
    model2.load(model_filename)

    assert model2.alpha == 0.1, "Gravity model: Alpha not saved properly"
    assert model2.function == "POWER", "Gravity model: Function not saved properly"
    assert model2.beta is None, "Gravity model: Beta not saved properly"
