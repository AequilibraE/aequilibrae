import pytest

from aequilibrae.paths.vdf import VDF


def test_functions_available():
    v = VDF()
    assert v.functions_available() == ["bpr", "bpr2", "conical", "inrets", "akcelik"], (
        "VDF class returning wrong availability"
    )
    assert v.apply_vdf is None, "VDF is missing term"
    assert v.apply_derivative is None, "VDF is missing term"

    with pytest.raises(ValueError):
        v.function = "Cubic"

    with pytest.raises(AttributeError):
        v.apply_vdf = isinstance
