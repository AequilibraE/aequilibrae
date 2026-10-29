import pytest


@pytest.fixture(scope="session")
def pbf_path():
    return pytest.importorskip("pyrosm").get_data("test_pbf")
