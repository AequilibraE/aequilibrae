from unittest import TestCase
from aequilibrae.distribution import SyntheticGravityModel
import tempfile
import os

filename = os.path.join(tempfile.gettempdir(), "aequilibrae_model_test.mod")


class TestSyntheticGravityModel(TestCase):
    def test_load(self):
        self.test_save()
        model = SyntheticGravityModel()
        model.load(filename)

        if model.alpha != 0.1:
            self.fail("Gravity model: Alpha not saved properly")

        if model.function != "POWER":
            self.fail("Gravity model: Function not saved properly")

        if model.beta is not None:
            self.fail("Gravity model: Beta not saved properly")

    def test_save(self):
        model = SyntheticGravityModel()
        model.function = "EXPO"
        model.beta = 0.1
        self.assertEqual(model.function, "EXPO")  # Did we save the value?

        model.function = "POWER"
        # Check if we zeroed the parameters when changing the function
        self.assertEqual(model.beta, None)
        model.alpha = 0.1

        model.save(filename)
