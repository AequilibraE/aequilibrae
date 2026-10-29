"""Project-local cache for raw network downloads and their manifests."""

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


_BASE_FOLDER_NAME = "downloaded data"


def _slugify(text: str) -> str:
    """Best-effort filesystem-safe slug. Keeps letters/digits/underscore/dash."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9_\-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-_")
    return text or "untagged"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class DownloadCache:
    """Write one import's raw payload under ``<project>/downloaded data/``."""

    def __init__(self, project_base_path, source_name: str, tag: str):
        self._project_base_path = Path(project_base_path)
        self._source_name = _slugify(source_name)
        self._timestamp = _utc_timestamp()
        self._tag = _slugify(tag)
        self._folder = (
            self._project_base_path / _BASE_FOLDER_NAME / self._source_name / f"{self._timestamp}__{self._tag}"
        )
        self._created = False
        self._sha256s: dict = {}

    @property
    def folder(self) -> Path:
        return self._folder

    @property
    def relative_path(self):
        """The path relative to the project base, or ``None`` if nothing written."""
        if not self._created:
            return None
        rel = self._folder.relative_to(self._project_base_path)
        return str(rel).replace("\\", "/")

    def _ensure_folder(self) -> None:
        if not self._created:
            self._folder.mkdir(parents=True, exist_ok=True)
            self._created = True
            logger.info(f"Download cache folder: {self._folder}")

    def write_geoparquet(self, name: str, gdf) -> Path:
        """Write a GeoDataFrame, adding the ``.parquet`` suffix if needed."""
        self._ensure_folder()
        if not name.endswith(".parquet"):
            name = name + ".parquet"
        target = self._folder / name
        gdf.to_parquet(target)
        self._sha256s[name] = _sha256_of_file(target)
        return target

    def write_json(self, name: str, payload) -> Path:
        """Write JSON, adding the ``.json`` suffix if needed."""
        self._ensure_folder()
        if not name.endswith(".json"):
            name = name + ".json"
        target = self._folder / name
        target.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        self._sha256s[name] = _sha256_of_file(target)
        return target

    def write_manifest(self, manifest: dict) -> Path:
        """Convenience wrapper that writes ``manifest.json`` with provenance defaults."""
        payload = dict(manifest)
        payload.setdefault("source", self._source_name)
        payload.setdefault("tag", self._tag)
        payload.setdefault("fetched_at", self._timestamp)
        payload["sha256"] = dict(self._sha256s)
        return self.write_json("manifest.json", payload)
