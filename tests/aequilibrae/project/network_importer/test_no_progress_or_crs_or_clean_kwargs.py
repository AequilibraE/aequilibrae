"""Guards that the public import API does not expose removed/banned kwargs."""

import inspect
import pytest

from aequilibrae.project.network.network import Network

_SUPPORTED_METHODS = ("import_network", "import_from_osm", "import_from_overture")
_BANNED_KWARGS = (
    "progress",
    "projected_crs",
    "clean",
    "commit",
    "link_types",
    "allowed_highways",
    "denylist",
    "link_type_filter",
    "highway_filter",
)


@pytest.mark.parametrize("method_name", _SUPPORTED_METHODS)
@pytest.mark.parametrize("banned", _BANNED_KWARGS)
def test_no_banned_kwargs(method_name, banned):
    sig = inspect.signature(getattr(Network, method_name))
    assert banned not in sig.parameters


@pytest.mark.parametrize("method_name", _SUPPORTED_METHODS)
def test_modes_kwarg_exists(method_name):
    sig = inspect.signature(getattr(Network, method_name))
    assert "modes" in sig.parameters


def test_import_from_osm_rejects_xml_path():
    assert "xml_path" not in inspect.signature(Network.import_from_osm).parameters


@pytest.mark.parametrize("banned", ["backend", "parquet_path", "keep_rule_arrays", "release"])
def test_import_from_overture_rejects_backend_knobs(banned):
    assert banned not in inspect.signature(Network.import_from_overture).parameters
