"""``Simplifier`` protocol + string-name registry."""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

from ..exceptions import SourceResolutionError
from ..ir import RoutableNetwork


@runtime_checkable
class Simplifier(Protocol):
    name: ClassVar[str]
    required_extras: ClassVar[tuple[str, ...]]

    def simplify(self, net: RoutableNetwork, **kwargs) -> RoutableNetwork:  # pragma: no cover
        ...


SIMPLIFIERS: dict[str, type] = {}


def register_simplifier(cls: type) -> type:
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} must define a class attribute `name`")
    SIMPLIFIERS[name] = cls
    return cls


def resolve_simplifier(simplifier: "Simplifier | str | bool", **kwargs) -> Simplifier | None:
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
            raise SourceResolutionError(
                f"Unknown simplifier name: {simplifier!r}. Available simplifiers: {available}"
            )
        return SIMPLIFIERS[simplifier](**kwargs)
    return simplifier
