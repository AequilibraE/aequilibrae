"""Plan §11.4: mode filtering is the only filter; no link-type filter exists."""

import inspect
import json
import sqlite3

import pytest

from aequilibrae.project.network.network import Network


_BANNED_FILTER_KWARGS = (
    "link_types",
    "allowed_highways",
    "denylist",
    "link_type_filter",
    "highway_filter",
)


@pytest.mark.parametrize(
    "method_name",
    [
        "import_network",
        "import_from_osm",
        "import_from_overture",
        "import_from_geodataframes",
        "import_from_file",
    ],
)
@pytest.mark.parametrize("kwarg", _BANNED_FILTER_KWARGS)
def test_no_link_type_filter_kwarg(method_name, kwarg):
    method = getattr(Network, method_name)
    sig = inspect.signature(method)
    assert kwarg not in sig.parameters, (
        f"{method_name}() must not accept '{kwarg}' (plan §1.3 rule 2)"
    )


def test_modes_kwarg_exists():
    """The only filter we allow."""
    for method_name in (
        "import_network",
        "import_from_osm",
        "import_from_overture",
    ):
        method = getattr(Network, method_name)
        sig = inspect.signature(method)
        assert "modes" in sig.parameters, f"{method_name} should accept modes"


def test_osm_import_preserves_all_link_types_for_active_modes(empty_project):
    """Walking-only mode keeps footways/cycleways alongside roads where foot is permitted."""
    pyrosm = pytest.importorskip("pyrosm")
    from pyrosm import get_data

    empty_project.network.import_from_osm(
        pbf_path=get_data("test_pbf"),
        modes=("walk",),
        simplify=False,
    )

    with sqlite3.connect(empty_project.path_to_file) as conn:
        link_types = {
            r[0]
            for r in conn.execute("SELECT DISTINCT link_type FROM links").fetchall()
        }
    # Walking-only must include at least one of the typically pedestrian highway tags
    pedestrian = {"footway", "pedestrian", "path", "steps", "cycleway"}
    assert link_types & pedestrian, (
        f"Expected pedestrian link_types to survive a walk-only import; got {link_types}"
    )
