"""``NeatnetSimplifier`` (filled in PR 4)."""

from __future__ import annotations

from typing import ClassVar

from ..ir import RoutableNetwork
from .base import register_simplifier


@register_simplifier
class NeatnetSimplifier:
    name: ClassVar[str] = "neatnet"
    required_extras: ClassVar[tuple[str, ...]] = ("neatnet",)

    def simplify(self, net: RoutableNetwork, **kwargs) -> RoutableNetwork:
        from .impl_neatnet import run_neatnet_simplify

        return run_neatnet_simplify(net, **kwargs)
