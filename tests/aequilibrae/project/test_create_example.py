import pytest
from aequilibrae.utils.create_example import create_example
from aequilibrae import Parameters


@pytest.fixture(params=["nauru", "sioux_falls", "coquimbo"])
def model_project(test_folder, request):
    proj = create_example(str(test_folder / request.param), from_model=request.param)
    yield proj
    proj.close()


def test_create_example(model_project):
    par = Parameters._default
    parproj = model_project.parameters
    assert par.keys() == parproj.keys(), "Wrong parameter keys for example project"
