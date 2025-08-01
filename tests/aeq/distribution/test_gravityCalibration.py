from aequilibrae.distribution import SyntheticGravityModel


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
