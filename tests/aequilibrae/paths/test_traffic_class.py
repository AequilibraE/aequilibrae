from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from aequilibrae.paths import TrafficClass


@pytest.fixture
def tc(sioux_falls_example):
    matrix = sioux_falls_example.matrices.get_matrix("demand_omx", subset=["matrix"])
    return TrafficClass(name="car", mode="c", cost="free_flow_time", demand=matrix)


def test_class_is_configuration_only(tc):
    assert tc.pce == tc.vot == 1.0
    assert not hasattr(tc, "graph")
    assert not hasattr(tc, "results")
    with pytest.raises(FrozenInstanceError):
        tc.pce = 2.0
    assert replace(tc, pce=3.9).pce == 3.9


@pytest.mark.parametrize("field", ["pce", "vot"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, -np.inf, True, False, "not a number"])
def test_rejects_invalid_pce_and_vot(tc, field, value):
    with pytest.raises(ValueError, match="positive finite"):
        replace(tc, **{field: value})


def test_copies_skims_and_selections(tc):
    skims = ["distance"]
    links = {"screenline": [1, (2, -1)]}
    configured = replace(tc, skimming=skims, select_link_sets=links)
    skims.append("capacity")
    links["screenline"].append(3)
    assert configured.skimming == ("distance",)
    assert configured.select_link_sets["screenline"] == ((1, 0), (2, -1))
    with pytest.raises(TypeError):
        configured.select_link_sets["new"] = ()


def test_old_class_constructor_is_removed(tc):
    with pytest.raises(TypeError):
        TrafficClass("car", None, tc.demand)
