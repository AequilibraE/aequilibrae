"""``OSMnxSimplifier``."""

from typing import ClassVar

from ..staged_network import StagedNetwork
from .base import register_simplifier


@register_simplifier
class OSMnxSimplifier:
    name: ClassVar[str] = "osmnx"
    required_extras: ClassVar[tuple] = ("osmnx",)

    def simplify(self, net: StagedNetwork, **kwargs) -> StagedNetwork:
        from .impl_osmnx import run_osmnx_simplify

        return run_osmnx_simplify(net, **kwargs)
