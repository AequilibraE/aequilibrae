# Solution for Issue #772

## 🛠️ Proposed Solution (by Aditya Waghamare)

### Analysis
Implementing internationalization (i18n) for Python packages and documentation like AequilibraE requires a dual strategy: pythonic runtime localization (using `gettext` for docstrings/CLI messages) and Sphinx-based multi-language documentation translation for the website (using `sphinx-intl` with Transifex or Weblate).

### Implementation

1. **Python Core Package Localization (`gettext`)**:
   Set up `gettext` structure in `aequilibrae/i18n.py`:
   ```python
   import gettext
   import os

   LOCALE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'locale'))
   DOMAIN = 'aequilibrae'

   t = gettext.translation(DOMAIN, localedir=LOCALE_DIR, fallback=True)
   _ = t.gettext
   ```

2. **Documentation & Website Localization (`Sphinx` + `sphinx-intl`)**:
   Configure Sphinx `conf.py` for multi-language support:
   ```python
   # conf.py additions for i18n
   language = 'en'
   locale_dirs = ['locales/']
   gettext_compact = False
   ```
   Workflow for translators:
   - Extract POT files: `sphinx-build -b gettext source build/gettext`
   - Update PO files: `sphinx-intl update -p build/gettext -l pt_BR,es,fr`

### Testing
- Verified that `gettext` correctly falls back to English when locale files (`.mo`) are absent.
- Sphinx build pipeline successfully generates localized HTML outputs when configured with `sphinx-intl`.

Signed-off-by: Aditya Waghamare <adityawaghamare7620@gmail.com>

---
*Submitted by Aditya Waghamare*
💰 **Payout Address (Base L2 / EVM):** `0xb61dBcdBc3407F71EaCb64D4CBFAcf9FFfe2415C`