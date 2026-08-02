import pytest

from .pathological_components import make_finite_turn_component, make_mixed_direction_component
from .pathological_network import PathologicalNetwork


@pytest.fixture
def finite_turn_network() -> PathologicalNetwork:
    """A finite movement cost makes the geometrically short branch suboptimal."""
    return PathologicalNetwork.compose(make_finite_turn_component())


@pytest.fixture
def mixed_direction_network() -> PathologicalNetwork:
    """Every physical-link direction convention appears in one four-node component."""
    return PathologicalNetwork.compose(make_mixed_direction_component())
