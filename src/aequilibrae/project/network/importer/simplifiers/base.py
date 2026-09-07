from aequilibrae.project.network.importer.exceptions import SourceResolutionError
from aequilibrae.project.network.importer.simplifiers.impl_neatnet import run_neatnet_simplify
from aequilibrae.project.network.importer.simplifiers.impl_osmnx import run_osmnx_simplify

SIMPLIFIERS = {
    "osmnx": run_osmnx_simplify,
    "neatnet": run_neatnet_simplify,
}


def resolve_simplifier(simplifier) -> tuple:
    """Return ``(name, simplify)``, or ``(None, None)`` when simplification is disabled."""
    if simplifier is False or simplifier is None:
        return None, None
    if simplifier is True:
        simplifier = "osmnx"
    if isinstance(simplifier, str):
        if simplifier not in SIMPLIFIERS:
            raise SourceResolutionError(
                f"Unknown simplifier name: {simplifier!r}. Available simplifiers: {sorted(SIMPLIFIERS)}"
            )
        return simplifier, SIMPLIFIERS[simplifier]
    return simplifier.name, simplifier.simplify
