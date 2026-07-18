from typing import ClassVar, Protocol, runtime_checkable

from aequilibrae.project.network.importer.exceptions import SourceResolutionError
from aequilibrae.project.network.importer.simplifiers.neatnet_simplifier import NeatnetSimplifier
from aequilibrae.project.network.importer.simplifiers.osmnx_simplifier import OSMnxSimplifier
from aequilibrae.project.network.importer.staged_network import StagedNetwork


@runtime_checkable
class Simplifier(Protocol):
    name: ClassVar[str]

    def simplify(self, net: StagedNetwork, **kwargs) -> StagedNetwork: ...


SIMPLIFIERS = {OSMnxSimplifier.name: OSMnxSimplifier, NeatnetSimplifier.name: NeatnetSimplifier}


def resolve_simplifier(simplifier, **kwargs):
    if simplifier is False or simplifier is None:
        return None
    if simplifier is True:
        simplifier = "osmnx"
    if isinstance(simplifier, str):
        if simplifier not in SIMPLIFIERS:
            available = sorted(SIMPLIFIERS)
            raise SourceResolutionError(f"Unknown simplifier name: {simplifier!r}. Available simplifiers: {available}")
        return SIMPLIFIERS[simplifier](**kwargs)
    if kwargs:
        raise SourceResolutionError(
            f"Keyword arguments {sorted(kwargs)} only apply when the simplifier is given by name; "
            "pass them to the simplifier object's constructor instead"
        )
    return simplifier
