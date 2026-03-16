import pandas as pd
import pytest

from aequilibrae.transit.transit_elements.basic_element import BasicPTElement


class DummyElement(BasicPTElement):
    def __init__(self):
        self.foo = None
        self.bar = None


def test_basic_element_from_row_sets_attributes():
    element = DummyElement()
    element.from_row(pd.Series({"foo": 1, "bar": "baz"}))
    assert element.foo == 1
    assert element.bar == "baz"


def test_basic_element_from_row_rejects_unknown_key():
    element = DummyElement()
    with pytest.raises(KeyError):
        element.from_row(pd.Series({"foo": 1, "missing": "nope"}))
