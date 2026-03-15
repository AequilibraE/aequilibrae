from __future__ import annotations

import ast
import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from babel.messages import Catalog, mofile, pofile
from deep_translator import GoogleTranslator

DOMAIN = "aequilibrae"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "aequilibrae"
LOCALE_ROOT = SOURCE_ROOT / "locale"

TARGET_LANGUAGES = {
    "es": "es",
    "fr": "fr",
    "pt_BR": "pt",
}
CACHE_DIR = PROJECT_ROOT / ".i18n_cache"

LOG_METHODS = {"debug", "info", "warning", "warn", "error", "critical", "exception"}


@dataclass(frozen=True)
class Location:
    file: str
    line: int


def _normalize_text(value: str) -> str:
    # Keep internal formatting exactly as source, but avoid blank catalog entries.
    if not value or not value.strip():
        return ""
    return value


def _const_string(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _chunk_text(text: str, max_size: int = 4000) -> Iterable[str]:
    if len(text) <= max_size:
        yield text
        return

    # Split by lines to keep structure and avoid translator size limits.
    chunk: list[str] = []
    current_size = 0
    for line in text.splitlines(keepends=True):
        if current_size + len(line) > max_size and chunk:
            yield "".join(chunk)
            chunk = [line]
            current_size = len(line)
        else:
            chunk.append(line)
            current_size += len(line)
    if chunk:
        yield "".join(chunk)


class MessageCollector(ast.NodeVisitor):
    def __init__(self, relative_path: str) -> None:
        self.relative_path = relative_path
        self.messages: dict[str, set[Location]] = defaultdict(set)

    def _add(self, message: str, line: int) -> None:
        normalized = _normalize_text(message)
        if not normalized:
            return
        self.messages[normalized].add(Location(self.relative_path, line))

    def visit_Module(self, node: ast.Module) -> None:  # noqa: N802
        module_doc = ast.get_docstring(node, clean=False)
        if module_doc and node.body:
            self._add(module_doc, getattr(node.body[0], "lineno", 1))
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        class_doc = ast.get_docstring(node, clean=False)
        if class_doc and node.body:
            self._add(class_doc, getattr(node.body[0], "lineno", node.lineno))
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        func_doc = ast.get_docstring(node, clean=False)
        if func_doc and node.body:
            self._add(func_doc, getattr(node.body[0], "lineno", node.lineno))
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        func_doc = ast.get_docstring(node, clean=False)
        if func_doc and node.body:
            self._add(func_doc, getattr(node.body[0], "lineno", node.lineno))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        first_arg = _const_string(node.args[0]) if node.args else None
        if first_arg:
            if isinstance(node.func, ast.Name) and node.func.id == "_":
                self._add(first_arg, node.lineno)
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in LOG_METHODS:
                    self._add(first_arg, node.lineno)
                elif isinstance(node.func.value, ast.Name) and node.func.value.id == "warnings" and node.func.attr == "warn":
                    self._add(first_arg, node.lineno)
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:  # noqa: N802
        if isinstance(node.exc, ast.Call) and node.exc.args:
            first_arg = _const_string(node.exc.args[0])
            if first_arg:
                self._add(first_arg, node.lineno)
        self.generic_visit(node)


def collect_messages() -> dict[str, set[Location]]:
    collected: dict[str, set[Location]] = defaultdict(set)
    for py_file in SOURCE_ROOT.rglob("*.py"):
        rel = py_file.relative_to(PROJECT_ROOT).as_posix()
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=rel)
        collector = MessageCollector(rel)
        collector.visit(tree)
        for msgid, locations in collector.messages.items():
            collected[msgid].update(locations)
    return collected


def translate_text(text: str, translator: GoogleTranslator) -> str:
    translated_chunks = [translator.translate(chunk) for chunk in _chunk_text(text)]
    return "".join(translated_chunks)


def _load_cache(locale: str) -> dict[str, str]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{locale}.json"
    if not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text(encoding="utf-8"))


def _save_cache(locale: str, cache: dict[str, str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{locale}.json"
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _translate_missing_entries(
    entries: list[str], translator: GoogleTranslator, cache: dict[str, str], locale: str
) -> dict[str, str]:
    translated: dict[str, str] = {}

    pending = [msgid for msgid in entries if msgid not in cache]
    short_batch: list[str] = []
    for msgid in pending:
        if len(msgid) <= 400:
            short_batch.append(msgid)
            if len(short_batch) >= 20:
                try:
                    results = translator.translate_batch(short_batch)
                    for src, dst in zip(short_batch, results):
                        cache[src] = dst
                except Exception:
                    for src in short_batch:
                        cache[src] = translate_text(src, translator)
                _save_cache(locale, cache)
                short_batch = []
        else:
            cache[msgid] = translate_text(msgid, translator)
            _save_cache(locale, cache)

    if short_batch:
        try:
            results = translator.translate_batch(short_batch)
            for src, dst in zip(short_batch, results):
                cache[src] = dst
        except Exception:
            for src in short_batch:
                cache[src] = translate_text(src, translator)
        _save_cache(locale, cache)

    for msgid in entries:
        translated[msgid] = cache[msgid]
    return translated


def build_catalog(locale: str, target_language: str, messages: dict[str, set[Location]], max_new: int | None = None) -> Catalog:
    po_path = LOCALE_ROOT / locale / "LC_MESSAGES" / f"{DOMAIN}.po"
    if po_path.exists():
        with po_path.open("rb") as fobj:
            previous = pofile.read_po(fobj, locale=locale, domain=DOMAIN)
    else:
        previous = Catalog(locale=locale, domain=DOMAIN)

    previous_map = {entry.id: entry.string for entry in previous if entry.id}
    translator = GoogleTranslator(source="en", target=target_language)
    cache = _load_cache(locale)

    missing = []
    for msgid in messages:
        existing = previous_map.get(msgid, "")
        msgstr = existing.strip() if isinstance(existing, str) else ""
        if not msgstr:
            missing.append(msgid)

    print(f"  auto-translating {len(missing)} entries for {locale}")
    if max_new is not None:
        missing = missing[:max_new]
        print(f"  limiting to {len(missing)} new entries for this run")
    if missing:
        _translate_missing_entries(missing, translator, cache, locale)
    _save_cache(locale, cache)

    catalog = Catalog(
        domain=DOMAIN,
        locale=locale,
        project="aequilibrae",
        version="1.6.1",
        msgid_bugs_address="aequilibrae@outerloop.io",
        last_translator="auto-sync",
        language_team=f"{locale} team",
        charset="utf-8",
    )
    catalog.fuzzy = False

    for msgid in sorted(messages):
        existing = previous_map.get(msgid, "")
        msgstr = existing.strip() if isinstance(existing, str) else ""
        if not msgstr:
            msgstr = cache.get(msgid, "")

        locations = sorted((loc.file, loc.line) for loc in messages[msgid])
        catalog.add(msgid, string=msgstr, locations=locations)

    return catalog


def write_catalog(locale: str, catalog: Catalog) -> None:
    lc_messages = LOCALE_ROOT / locale / "LC_MESSAGES"
    lc_messages.mkdir(parents=True, exist_ok=True)

    po_path = lc_messages / f"{DOMAIN}.po"
    mo_path = lc_messages / f"{DOMAIN}.mo"

    with po_path.open("wb") as fobj:
        pofile.write_po(fobj, catalog, width=120)  # type: ignore[arg-type]

    with mo_path.open("wb") as fobj:
        mofile.write_mo(fobj, catalog)  # type: ignore[arg-type]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync aequilibrae gettext catalogs from source strings/docstrings")
    parser.add_argument(
        "--locale",
        action="append",
        dest="locales",
        help="Locale to process (repeatable). Defaults to all supported locales.",
    )
    parser.add_argument(
        "--max-new",
        type=int,
        default=None,
        help="Optional cap on number of new entries to machine-translate in one run.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    messages = collect_messages()
    print(f"Collected {len(messages)} unique msgids")

    locales = args.locales or list(TARGET_LANGUAGES.keys())

    for locale in locales:
        if locale not in TARGET_LANGUAGES:
            raise ValueError(f"Unsupported locale: {locale}")
        target = TARGET_LANGUAGES[locale]
        print(f"Syncing {locale}...")
        catalog = build_catalog(locale, target, messages, max_new=args.max_new)
        write_catalog(locale, catalog)
        untranslated = sum(1 for entry in catalog if entry.id and not str(entry.string).strip())
        print(f"  entries: {sum(1 for e in catalog if e.id)} | untranslated: {untranslated}")


if __name__ == "__main__":
    main()

