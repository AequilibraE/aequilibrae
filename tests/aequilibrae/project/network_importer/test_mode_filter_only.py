import inspect
import sqlite3

import pytest

from aequilibrae.project.network.network import Network

_BANNED_FILTER_KWARGS = ("link_types", "allowed_highways", "denylist", "link_type_filter", "highway_filter")
_SUPPORTED_METHODS = ("import_network", "import_from_osm", "import_from_overture")


@pytest.mark.parametrize("method_name", _SUPPORTED_METHODS)
@pytest.mark.parametrize("kwarg", _BANNED_FILTER_KWARGS)
def test_no_link_type_filter_kwarg(method_name, kwarg):
    sig = inspect.signature(getattr(Network, method_name))
    assert kwarg not in sig.parameters


def test_modes_kwarg_exists():
    for method_name in _SUPPORTED_METHODS:
        sig = inspect.signature(getattr(Network, method_name))
        assert "modes" in sig.parameters


def test_osm_import_preserves_all_link_types_for_active_modes(empty_project):
    pytest.importorskip("pyrosm")
    from pyrosm import get_data

    empty_project.network.import_from_osm(
        pbf_path=get_data("test_pbf"),
        modes=("walk",),
        simplify=False,
    )

    with sqlite3.connect(empty_project.path_to_file) as conn:
        link_types = {r[0] for r in conn.execute("SELECT DISTINCT link_type FROM links").fetchall()}
    pedestrian = {"footway", "pedestrian", "path", "steps", "cycleway"}
    assert link_types & pedestrian
