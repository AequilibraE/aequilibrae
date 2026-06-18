"""``NeatnetSimplifier``."""

from typing import ClassVar

from aequilibrae.project.network.importer.simplifiers.base import register_simplifier
from aequilibrae.project.network.importer.staged_network import StagedNetwork


@register_simplifier
class NeatnetSimplifier:
    name: ClassVar[str] = "neatnet"
    required_extras: ClassVar[tuple] = ("neatnet",)

    def simplify(self, net: StagedNetwork, **kwargs) -> StagedNetwork:
        from aequilibrae.project.network.importer.simplifiers.impl_neatnet import (
            run_neatnet_simplify,
        )

        return run_neatnet_simplify(net, **kwargs)
