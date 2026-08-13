from pathlib import Path

from babel.messages import pofile

from aequilibrae.i18n import _, available_languages, set_language


def test_available_languages_include_initial_catalogs():
    assert {"pt_BR", "es", "fr"}.issubset(set(available_languages()))


def test_set_language_and_fallback_behavior():
    msgid = "Lists languages for which text catalogs are available in the source tree."
    set_language(None)
    set_language("es")
    assert _(msgid) != msgid

    set_language("zz")
    assert _(msgid) == msgid


def test_translated_runtime_message():
    msgid = (
        "multiprocessing start method already set. On MacOS, AequilibraE requires the 'fork' start method. "
        "AequilibraE may crash when using procedures that utilise multiprocessing or progress bars."
    )
    set_language(None)
    set_language("pt_BR")
    assert _(msgid) != msgid


def test_translated_docstring_changes_with_language():
    from aequilibrae import i18n as i18n_module

    set_language(None)
    original_doc = i18n_module.available_languages.__doc__
    set_language("es")
    translated_doc = i18n_module.available_languages.__doc__
    assert translated_doc != original_doc

    set_language(None)
    restored_doc = i18n_module.available_languages.__doc__
    assert restored_doc == original_doc


def test_catalogs_are_fully_translated():
    base = Path(__file__).resolve().parents[3] / "aequilibrae" / "locale"
    for locale in ["es", "fr", "pt_BR"]:
        po_path = base / locale / "LC_MESSAGES" / "aequilibrae.po"
        with po_path.open("rb") as fobj:
            catalog = pofile.read_po(fobj, locale=locale, domain="aequilibrae")
        untranslated = [entry.id for entry in catalog if entry.id and not str(entry.string).strip()]
        assert not untranslated, f"{locale} has untranslated entries: {len(untranslated)}"

