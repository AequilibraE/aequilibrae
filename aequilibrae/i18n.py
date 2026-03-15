import os
import sys
import inspect
import weakref
from gettext import gettext as _default_gettext
from gettext import translation
from pathlib import Path
from types import ModuleType
from typing import Optional

DOMAIN = "aequilibrae"
LOCALE_DIR = Path(__file__).resolve().parent / "locale"
__all__ = ["_", "set_language", "available_languages", "translate_docstrings"]
_translate = _default_gettext
_original_docstrings: dict[int, tuple[object, Optional[str]]] = {}


def _(message: str) -> str:
    return _translate(message)


def _safe_ref(obj: object):
    try:
        return weakref.ref(obj)
    except TypeError:
        return lambda: obj


def _set_docstring(obj: object, doc: Optional[str]) -> None:
    try:
        obj.__doc__ = doc
    except Exception:
        return


def _translate_doc_for_object(obj: object, translator) -> None:
    try:
        doc = getattr(obj, "__doc__", None)
    except Exception:
        return

    key = id(obj)
    if key not in _original_docstrings:
        _original_docstrings[key] = (_safe_ref(obj), doc)

    original_doc = _original_docstrings[key][1]

    if translator is None:
        _set_docstring(obj, original_doc)
        return

    if not original_doc:
        return

    translated = translator(original_doc)
    if translated != doc:
        _set_docstring(obj, translated)


def _translate_docstrings_for_module(module: ModuleType, translator) -> None:
    _translate_doc_for_object(module, translator)
    for name, attr in list(vars(module).items()):
        if name.startswith("__") and name.endswith("__"):
            continue
        if inspect.isclass(attr):
            _translate_doc_for_object(attr, translator)
            for member_name, member in list(vars(attr).items()):
                if member_name.startswith("__") and member_name.endswith("__"):
                    continue
                if inspect.isfunction(member) or inspect.ismethod(member):
                    _translate_doc_for_object(member, translator)
            continue
        if inspect.isfunction(attr) or inspect.ismethod(attr):
            _translate_doc_for_object(attr, translator)


def translate_docstrings(language: Optional[str] = None) -> None:
    """Translates docstrings for loaded aequilibrae modules and restores originals on fallback."""
    translator = None
    if language:
        try:
            translator = translation(DOMAIN, localedir=LOCALE_DIR, languages=[language]).gettext
        except OSError:
            translator = None

    for module_name, module in list(sys.modules.items()):
        if not module_name or not isinstance(module, ModuleType):
            continue
        if module_name != "aequilibrae" and not module_name.startswith("aequilibrae."):
            continue
        _translate_docstrings_for_module(module, translator)


def set_language(language: Optional[str] = None) -> None:
    """Configures package translations from compiled gettext catalogs if available.

    This function updates process-global translator state and should be configured at startup.
    """
    global _translate
    language = language or os.getenv("AEQUILIBRAE_LANGUAGE")
    if not language:
        _translate = _default_gettext
        translate_docstrings(None)
        return
    try:
        translator = translation(DOMAIN, localedir=LOCALE_DIR, languages=[language])
        _translate = translator.gettext
        translate_docstrings(language)
    except OSError:
        _translate = _default_gettext
        translate_docstrings(None)
        return


def available_languages() -> list[str]:
    """Lists languages for which text catalogs are available in the source tree."""
    return sorted(p.parent.parent.name for p in LOCALE_DIR.glob(f"*/LC_MESSAGES/{DOMAIN}.mo"))
