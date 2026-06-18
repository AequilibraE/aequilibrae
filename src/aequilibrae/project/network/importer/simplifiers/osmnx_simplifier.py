"""``OSMnxSimplifier``."""

from typing import ClassVar

from aequilibrae.project.network.importer.simplifiers.base import register_simplifier
from aequilibrae.project.network.importer.staged_network import StagedNetwork


@register_simplifier
class OSMnxSimplifier:
    name: ClassVar[str] = "osmnx"
    required_extras: ClassVar[tuple] = ("osmnx",)

    def simplify(self, net: StagedNetwork, **kwargs) -> StagedNetwork:
        from aequilibrae.project.network.importer.simplifiers.impl_osmnx import run_osmnx_simplify

        return run_osmnx_simplify(net, **kwargs)
