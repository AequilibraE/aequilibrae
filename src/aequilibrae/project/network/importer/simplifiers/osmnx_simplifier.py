"""``OSMnxSimplifier`` (filled in PR 4)."""

from __future__ import annotations

from typing import ClassVar

from ..ir import RoutableNetwork
from .base import register_simplifier


@register_simplifier
class OSMnxSimplifier:
    name: ClassVar[str] = "osmnx"
    required_extras: ClassVar[tuple[str, ...]] = ("osmnx",)

    def simplify(self, net: RoutableNetwork, **kwargs) -> RoutableNetwork:
        from .impl_osmnx import run_osmnx_simplify

        return run_osmnx_simplify(net, **kwargs)
