# i18n synchronization workflow

This helper scans the Python source tree for runtime message strings and docstrings,
updates gettext catalogs, and compiles `.mo` files.

## Run

```powershell
python -u tools\i18n_sync.py
```

## What it does

- parses `aequilibrae/**/*.py`
- collects:
  - module/class/function docstrings
  - `_()` strings
  - logger/warnings message strings
  - exception message strings in `raise ...("...")`
- updates and compiles:
  - `aequilibrae/locale/es/LC_MESSAGES/aequilibrae.po|mo`
  - `aequilibrae/locale/fr/LC_MESSAGES/aequilibrae.po|mo`
  - `aequilibrae/locale/pt_BR/LC_MESSAGES/aequilibrae.po|mo`

