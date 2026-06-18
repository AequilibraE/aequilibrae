from typing import ClassVar

from aequilibrae.project.network.importer.staged_network import StagedNetwork


class NeatnetSimplifier:
    name: ClassVar[str] = "neatnet"

    def simplify(self, net: StagedNetwork, **kwargs) -> StagedNetwork:
        from aequilibrae.project.network.importer.simplifiers.impl_neatnet import run_neatnet_simplify

        return run_neatnet_simplify(net, **kwargs)
