"""Plan §1.3 rule 4: no ``progress`` / ``projected_crs`` / ``clean`` / ``commit`` kwargs.

Also: ``import_from_osm`` rejects ``xml_path``; ``import_from_overture`` rejects
``backend`` / ``parquet_path`` / ``keep_rule_arrays``.
"""

import inspect

import pytest

from aequilibrae.project.network.network import Network


_BANNED_KWARGS_EVERYWHERE = ("progress", "projected_crs", "clean", "commit")


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
@pytest.mark.parametrize("banned", _BANNED_KWARGS_EVERYWHERE)
def test_no_banned_kwargs(method_name, banned):
    method = getattr(Network, method_name)
    sig = inspect.signature(method)
    assert banned not in sig.parameters, (
        f"{method_name}() must not accept '{banned}' (plan §1.3 rules 4 & 5)"
    )


def test_import_from_osm_rejects_xml_path():
    sig = inspect.signature(Network.import_from_osm)
    assert "xml_path" not in sig.parameters, "OSM XML source is not supported"


@pytest.mark.parametrize("banned", ["backend", "parquet_path", "keep_rule_arrays"])
def test_import_from_overture_rejects_backend_knobs(banned):
    sig = inspect.signature(Network.import_from_overture)
    assert banned not in sig.parameters, (
        f"import_from_overture() must not accept '{banned}' "
        "(cloud is the only backend; rule arrays are always preserved)"
    )
