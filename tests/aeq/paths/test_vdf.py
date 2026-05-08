import pytest

from aequilibrae.paths.vdf import VDF


def test_functions_available():
    v = VDF()
    assert v.functions_available() == ["bpr", "bpr2", "conical", "inrets"], "VDF class returning wrong availability"
    assert v.apply_vdf is None, "VDF is missing term"
    assert v.apply_derivative is None, "VDF is missing term"
    assert v.apply_integral is None, "VDF is missing term"

    with pytest.raises(ValueError):
        v.function = "Cubic"

    with pytest.raises(AttributeError):
        v.apply_vdf = isinstance


@pytest.mark.parametrize("name", ["BPR", "BPR2", "CONICAL", "INRETS"])
def test_each_vdf_wires_apply_integral(name):
    v = VDF()
    v.function = name
    assert v.apply_vdf is not None, f"{name} did not wire apply_vdf"
    assert v.apply_derivative is not None, f"{name} did not wire apply_derivative"
    assert v.apply_integral is not None, f"{name} did not wire apply_integral"
