from aequilibrae.i18n import _, available_languages, set_language


def test_available_languages_include_initial_catalogs():
    assert {"pt_BR", "es", "fr"}.issubset(set(available_languages()))


def test_set_language_and_fallback_behavior():
    msgid = "Lists languages for which text catalogs are available in the source tree."
    translated_msg = "Enumera los idiomas para los que hay catálogos de texto disponibles en el árbol de código fuente."
    set_language("es")
    assert _(msgid) == translated_msg

    set_language("zz")
    assert _(msgid) == msgid


def test_translated_runtime_message():
    msgid = "Configures package translations from compiled gettext catalogs if available."
    translated_msg = "Configura as traduções do pacote a partir de catálogos gettext compilados, quando disponíveis."
    set_language("pt_BR")
    assert _(msgid) == translated_msg
