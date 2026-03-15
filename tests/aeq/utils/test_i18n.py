from aequilibrae.i18n import available_languages, set_language


def test_available_languages_include_initial_catalogs():
    assert {"pt_BR", "es", "fr"}.issubset(set(available_languages()))


def test_set_language_is_safe_for_missing_catalog():
    set_language("zz")
