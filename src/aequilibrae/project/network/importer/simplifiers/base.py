from typing import ClassVar, Protocol, runtime_checkable

from aequilibrae.project.network.importer.exceptions import SourceResolutionError
from aequilibrae.project.network.importer.staged_network import StagedNetwork


@runtime_checkable
class Simplifier(Protocol):
    name: ClassVar[str]

    def simplify(self, net: StagedNetwork, **kwargs) -> StagedNetwork: ...


def _simplifiers() -> dict:
    from aequilibrae.project.network.importer.simplifiers.neatnet_simplifier import NeatnetSimplifier
    from aequilibrae.project.network.importer.simplifiers.osmnx_simplifier import OSMnxSimplifier

    return {OSMnxSimplifier.name: OSMnxSimplifier, NeatnetSimplifier.name: NeatnetSimplifier}


SIMPLIFIERS = _simplifiers()


def resolve_simplifier(simplifier, **kwargs):
    if simplifier is False or simplifier is None:
        return None
    if simplifier is True:
        simplifier = "osmnx"
    if isinstance(simplifier, str):
        simplifiers = _simplifiers()
        if simplifier not in simplifiers:
            available = sorted(simplifiers.keys())
            raise SourceResolutionError(f"Unknown simplifier name: {simplifier!r}. Available simplifiers: {available}")
        return simplifiers[simplifier](**kwargs)
    return simplifier
