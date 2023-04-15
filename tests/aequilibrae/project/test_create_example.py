from aequilibrae.utils.create_example import create_example
from aequilibrae import Parameters
import pytest


class TestCreateExample:
    # We use fixture parametrization here to create the projects so that we can do proper teardown
    # of the project, even when the test fails
    # see also: https://docs.pytest.org/en/6.2.x/fixture.html#fixture-parametrize
    @pytest.fixture(params=["nauru", "sioux_falls", "coquimbo"])
    def model_project(self, tmp_path, request):
        proj = create_example(str(tmp_path / request.param), from_model=request.param)
        yield proj
        proj.close()

    def test_create_example(self, model_project):
        par = Parameters._default
        parproj = model_project.parameters
        assert par.keys() == parproj.keys(), "Wrong parameter keys for example project"
