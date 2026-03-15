import os
from gettext import gettext as _default_gettext
from gettext import translation
from pathlib import Path

DOMAIN = "aequilibrae"
LOCALE_DIR = Path(__file__).resolve().parent / "locale"
__all__ = ["_", "set_language", "available_languages"]
_translate = _default_gettext


def _(message: str) -> str:
    return _translate(message)


def set_language(language: str | None = None) -> None:
    """Configures package translations from compiled gettext catalogs if available.

    This function updates process-global translator state and should be configured at startup.
    """
    global _translate
    language = language or os.getenv("AEQUILIBRAE_LANGUAGE")
    if not language:
        _translate = _default_gettext
        return
    try:
        translator = translation(DOMAIN, localedir=LOCALE_DIR, languages=[language])
        _translate = translator.gettext
    except OSError:
        _translate = _default_gettext
        return


def available_languages() -> list[str]:
    """Lists languages for which text catalogs are available in the source tree."""
    return sorted(p.parent.parent.name for p in LOCALE_DIR.glob(f"*/LC_MESSAGES/{DOMAIN}.mo"))
