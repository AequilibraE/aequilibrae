"""``NeatnetSimplifier``."""

from typing import ClassVar

from ..staged_network import StagedNetwork
from .base import register_simplifier


@register_simplifier
class NeatnetSimplifier:
    name: ClassVar[str] = "neatnet"
    required_extras: ClassVar[tuple] = ("neatnet",)

    def simplify(self, net: StagedNetwork, **kwargs) -> StagedNetwork:
        from .impl_neatnet import run_neatnet_simplify

        return run_neatnet_simplify(net, **kwargs)
