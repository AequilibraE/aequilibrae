"""Raw-download cache under ``<project>/downloaded data/``.

Every source that retrieves data over the network writes the raw, untransformed
payload to a project-local folder **before any parsing/transformation runs**.
Local-file sources do not write anything.

Layout (see plan §4.6):

    <project_path>/
      downloaded data/
        <source_name>/
          <ISO timestamp>__<short tag>/
            <payload files>
            manifest.json
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

logger = logging.getLogger(__name__)


_GZIP_THRESHOLD_BYTES = 10 * 1024 * 1024  # 10 MB
_BASE_FOLDER_NAME = "downloaded data"


def _slugify(text: str) -> str:
    """Best-effort filesystem-safe slug. Keeps letters/digits/underscore/dash."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9_\-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-_")
    return text or "untagged"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class DownloadCache:
    """Per-import handle for writing raw payloads under ``<project>/downloaded data/``.

    The folder is created lazily on first write. Local-file sources may construct
    a cache and never write anything; in that case no folder is created and
    ``relative_path`` returns ``None``.
    """

    def __init__(self, project_base_path: str | Path, source_name: str, tag: str):
        """
        :Arguments:
            **project_base_path**: The path returned by ``project.project_base_path``.
            **source_name**: Source identifier, e.g. ``"osm-overpass"``,
            ``"overture-cloud"``. Becomes a subfolder name.
            **tag**: A short, human-readable tag — e.g. a slugified place name
            or bbox tuple. Used in the timestamped subfolder name.
        """
        self._project_base_path = Path(project_base_path)
        self._source_name = _slugify(source_name)
        self._timestamp = _utc_timestamp()
        self._tag = _slugify(tag)
        self._folder = (
            self._project_base_path
            / _BASE_FOLDER_NAME
            / self._source_name
            / f"{self._timestamp}__{self._tag}"
        )
        self._created = False
        self._sha256s: dict[str, str] = {}

    @property
    def folder(self) -> Path:
        return self._folder

    @property
    def relative_path(self) -> str | None:
        """The path relative to the project base, or ``None`` if nothing written."""
        if not self._created:
            return None
        try:
            rel = self._folder.relative_to(self._project_base_path)
        except ValueError:
            return str(self._folder)
        return str(rel).replace("\\", "/")

    def _ensure_folder(self) -> None:
        if not self._created:
            self._folder.mkdir(parents=True, exist_ok=True)
            self._created = True
            logger.info(f"Download cache folder: {self._folder}")

    def write_bytes(self, name: str, payload: bytes, *, allow_gzip: bool = True) -> Path:
        """Write a raw byte payload.

        If ``allow_gzip`` is True and the payload exceeds ``_GZIP_THRESHOLD_BYTES``,
        the file is written with a ``.gz`` suffix using gzip compression.
        """
        self._ensure_folder()
        self._sha256s[name] = _sha256(payload)
        target = self._folder / name
        if allow_gzip and len(payload) > _GZIP_THRESHOLD_BYTES:
            target = target.with_suffix(target.suffix + ".gz")
            with gzip.open(target, "wb") as handle:
                handle.write(payload)
        else:
            target.write_bytes(payload)
        return target

    def write_text(self, name: str, payload: str, *, allow_gzip: bool = True) -> Path:
        return self.write_bytes(name, payload.encode("utf-8"), allow_gzip=allow_gzip)

    def write_table(self, name: str, table: "pa.Table") -> Path:
        """Write a ``pyarrow.Table`` as parquet (no geopandas round-trip).

        Used by the Overture cloud source.
        """
        import pyarrow.parquet as pq

        self._ensure_folder()
        target = self._folder / name
        pq.write_table(table, target)
        # SHA-256 of the written file (so identical-content tables across imports match)
        try:
            self._sha256s[name] = _sha256(target.read_bytes())
        except OSError:
            self._sha256s[name] = ""
        return target

    def write_manifest(self, manifest: dict[str, Any]) -> Path:
        """Write a ``manifest.json`` describing the request and payloads."""
        self._ensure_folder()
        payload = dict(manifest)
        payload.setdefault("source", self._source_name)
        payload.setdefault("tag", self._tag)
        payload.setdefault("fetched_at", self._timestamp)
        payload["sha256"] = dict(self._sha256s)
        target = self._folder / "manifest.json"
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return target
