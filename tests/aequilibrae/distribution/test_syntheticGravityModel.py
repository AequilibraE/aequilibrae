import pytest

from aequilibrae.distribution import SyntheticGravityModel


@pytest.fixture
def model_filename(tmp_path):
    return tmp_path / "aequilibrae_model_test.mod"


@pytest.fixture
def saved_model(model_filename):
    model = SyntheticGravityModel()
    model.function = "POWER"
    model.alpha = 0.1
    model.save(model_filename)
    return model_filename


def test_save(model_filename):
    model = SyntheticGravityModel()
    model.function = "EXPO"
    model.beta = 0.1
    assert model.function == "EXPO"  # Did we save the value?

    model.function = "POWER"
    # Check if we zeroed the parameters when changing the function
    assert model.beta is None
    model.alpha = 0.1

    model.save(model_filename)
    return model_filename


def test_load(saved_model):
    model = SyntheticGravityModel()
    model.load(saved_model)

    assert model.alpha == 0.1, "Gravity model: Alpha not saved properly"
    assert model.function == "POWER", "Gravity model: Function not saved properly"
    assert model.beta is None, "Gravity model: Beta not saved properly"
