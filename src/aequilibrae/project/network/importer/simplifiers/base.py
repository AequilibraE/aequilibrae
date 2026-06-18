"""``Simplifier`` protocol + string-name registry."""

from typing import ClassVar, Protocol, runtime_checkable

from aequilibrae.project.network.importer.exceptions import SourceResolutionError
from aequilibrae.project.network.importer.staged_network import StagedNetwork


@runtime_checkable
class Simplifier(Protocol):
    name: ClassVar[str]
    required_extras: ClassVar[tuple]

    def simplify(self, net: StagedNetwork, **kwargs) -> StagedNetwork: ...


SIMPLIFIERS: dict = {}


def register_simplifier(cls: type) -> type:
    SIMPLIFIERS[cls.name] = cls
    return cls


def resolve_simplifier(simplifier, **kwargs):
    """Resolve a Simplifier from an instance, string name, or boolean.

    ``True`` resolves to the default (``"osmnx"``); ``False`` returns ``None``
    meaning "skip simplification".
    """
    if simplifier is False:
        return None
    if simplifier is True:
        simplifier = "osmnx"
    if isinstance(simplifier, str):
        if simplifier not in SIMPLIFIERS:
            available = sorted(SIMPLIFIERS.keys())
            raise SourceResolutionError(f"Unknown simplifier name: {simplifier!r}. Available simplifiers: {available}")
        return SIMPLIFIERS[simplifier](**kwargs)
    return simplifier
