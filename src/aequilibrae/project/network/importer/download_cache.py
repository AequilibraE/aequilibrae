"""Raw-download cache under ``<project>/downloaded data/``.

Every source that retrieves data over the network writes the raw payload to a
project-local folder **before any parsing/transformation runs**. Local-file
sources do not write anything.

Layout:

    <project_path>/
      downloaded data/
        <source_name>/
          <ISO timestamp>__<short tag>/
            <payload files>            # .parquet (Parquet) or .json
            manifest.json

Only two on-disk payload formats are supported:
  - GeoParquet (``write_geoparquet`` for ``gpd.GeoDataFrame``)
  - JSON (``write_json`` for the manifest and small metadata documents)

There is no gzip, no raw-bytes path, no per-source raw format. Sources that
naturally produce JSON (e.g. Overpass) must consolidate their data into a
single ``GeoDataFrame`` first and persist it as GeoParquet.
"""

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
    """Per-import handle for writing raw payloads under ``<project>/downloaded data/``.

    The folder is created lazily on first write. Local-file sources may
    construct a cache and never write anything; in that case no folder is
    created and ``relative_path`` returns ``None``.
    """

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
        """Write a GeoDataFrame as GeoParquet.

        :Arguments:
            **name**: File name (the ``.parquet`` extension is added if missing).
            **gdf**: A ``geopandas.GeoDataFrame``.
        """
        self._ensure_folder()
        if not name.endswith(".parquet"):
            name = name + ".parquet"
        target = self._folder / name
        gdf.to_parquet(target)
        self._sha256s[name] = _sha256_of_file(target)
        return target

    def write_json(self, name: str, payload) -> Path:
        """Write a JSON document.

        :Arguments:
            **name**: File name (the ``.json`` extension is added if missing).
            **payload**: A dict or list serialisable via ``json.dumps(..., default=str)``.
        """
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
