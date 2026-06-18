import inspect
import pytest

from aequilibrae.project.network.network import Network

_BANNED_KWARGS_EVERYWHERE = ("progress", "projected_crs", "clean", "commit")
_SUPPORTED_METHODS = ("import_network", "import_from_osm", "import_from_overture")


@pytest.mark.parametrize("method_name", _SUPPORTED_METHODS)
@pytest.mark.parametrize("banned", _BANNED_KWARGS_EVERYWHERE)
def test_no_banned_kwargs(method_name, banned):
    sig = inspect.signature(getattr(Network, method_name))
    assert banned not in sig.parameters


def test_import_from_osm_rejects_xml_path():
    sig = inspect.signature(Network.import_from_osm)
    assert "xml_path" not in sig.parameters


@pytest.mark.parametrize("banned", ["backend", "parquet_path", "keep_rule_arrays", "release"])
def test_import_from_overture_rejects_backend_knobs(banned):
    sig = inspect.signature(Network.import_from_overture)
    assert banned not in sig.parameters
