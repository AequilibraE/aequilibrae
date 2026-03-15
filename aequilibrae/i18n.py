import os
from gettext import gettext as _
from gettext import translation
from pathlib import Path

DOMAIN = "aequilibrae"
LOCALE_DIR = Path(__file__).resolve().parent / "locale"
__all__ = ["_", "set_language", "available_languages"]


def set_language(language: str | None = None) -> None:
    """Configures package translations from compiled gettext catalogs if available."""
    language = language or os.getenv("AEQUILIBRAE_LANGUAGE")
    if not language:
        return
    try:
        translator = translation(DOMAIN, localedir=LOCALE_DIR, languages=[language])
        translator.install()
    except OSError:
        return


def available_languages() -> list[str]:
    """Lists languages for which text catalogs are available in the source tree."""
    return sorted(p.parent.parent.name for p in LOCALE_DIR.glob(f"*/LC_MESSAGES/{DOMAIN}.po"))
