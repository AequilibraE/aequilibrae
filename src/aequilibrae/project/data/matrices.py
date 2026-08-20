import logging
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.project.project_table import NonSpatialProjectTable
from aequilibrae.utils.db_utils import NestedTransactions

logger = logging.getLogger(__name__)


class Matrices(NonSpatialProjectTable):
    """Matrix metadata gateway with explicit file-managing helpers."""

    name = "matrices"
    key = "name"
    record_name = "MatrixRecord"

    def __init__(self, transactions: NestedTransactions, matrices_path: str | Path) -> None:
        """Create the matrix metadata gateway.

        :Arguments:
            **transactions** (:obj:`NestedTransactions`): Manager for the
            project database containing matrix metadata.

            **matrices_path** (:obj:`str` or :obj:`Path`): Directory containing
            matrix payload files.
        """
        super().__init__(transactions)
        self.folder: Path = Path(matrices_path)

    def create(
        self, name: str, file_name: str, matrix: AequilibraeMatrix | None = None
    ) -> Any:
        """Create matrix metadata and optionally export a matrix payload.

        This filesystem operation cannot participate in a project transaction.
        A file supplied for registration remains caller-owned on failure.

        :Arguments:
            **name** (:obj:`str`): Unique matrix name stored in project metadata.

            **file_name** (:obj:`str`): Matrix payload file name relative to the
            project's matrix directory.

            **matrix** (:obj:`AequilibraeMatrix`, *Optional*): Matrix to export.
            If omitted, ``file_name`` must already exist.

        :Returns:
            **matrix record** (:obj:`Any`): Generated frozen metadata record.
        """
        self._require_resource_idle()
        path = self.folder / file_name
        if name in self:
            raise ValueError(f"There is already a matrix of name ({name}). It must be unique.")
        if self._transactions.execute("SELECT 1 FROM matrices WHERE file_name=?", (file_name,)).fetchone():
            raise ValueError(f"There is already a matrix record for file name ({file_name}). It must be unique.")

        created = False
        temporary = None
        if matrix is not None and matrix.cores > 0:
            if path.exists():
                raise FileExistsError(f"{file_name} already exists. Choose a different name or matrix format")
            suffix = path.suffix.lower()
            if suffix not in (".omx", ".aem"):
                raise ValueError("Matrix needs to be either OMX or native AequilibraE")
            temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{suffix}")
            try:
                matrix.export(str(temporary))
                os.replace(temporary, path)
                created = True
            except BaseException:
                if temporary.exists():
                    temporary.unlink()
                raise
            cores = matrix.cores
        else:
            if not path.is_file():
                raise FileNotFoundError(f"{file_name} does not exist. Cannot create this matrix record")
            cores = self.__cores_on_disk(file_name)

        try:
            self.insert(name=name, file_name=file_name, cores=cores)
        except BaseException as primary:
            if created:
                try:
                    path.unlink()
                except BaseException as cleanup:
                    _add_resource_note(primary, f"created matrix remains at {path}: {cleanup!r}")
            raise
        return self.get(name)

    def register_matrix(self, name: str, file_name: str) -> Any:
        """Register an existing caller-owned matrix file.

        :Arguments:
            **name** (:obj:`str`): Unique matrix name.

            **file_name** (:obj:`str`): Existing matrix payload file name.

        :Returns:
            **matrix record** (:obj:`Any`): Generated frozen metadata record.
        """
        return self.create(name, file_name, matrix=None)

    def delete_matrix(self, name: str) -> None:
        """Delete matrix metadata and its underlying file.

        :Arguments:
            **name** (:obj:`str`): Matrix name to delete.
        """
        self._require_resource_idle()
        record = self.get(name)
        path = self.folder / record.file_name
        tombstone = path.with_name(f".{path.name}.{uuid.uuid4().hex}.deleted")
        moved = False
        if path.exists():
            os.replace(path, tombstone)
            moved = True
        try:
            super().delete(name)
        except BaseException as primary:
            if moved:
                try:
                    os.replace(tombstone, path)
                except BaseException as cleanup:
                    _add_resource_note(primary, f"matrix is stranded at {tombstone}: {cleanup!r}")
            raise
        if moved:
            tombstone.unlink()

    def get_matrix(self, name: str) -> AequilibraeMatrix:
        """Load a matrix payload by metadata name.

        :Arguments:
            **name** (:obj:`str`): Registered matrix name.

        :Returns:
            **matrix** (:obj:`AequilibraeMatrix`): Loaded matrix payload.
        """
        record = self.get(name)
        matrix = AequilibraeMatrix()
        matrix.load(str(self.folder / record.file_name))
        return matrix

    def check_exists(self, name: str) -> bool:
        """Return whether matrix metadata exists for ``name``.

        :Arguments:
            **name** (:obj:`str`): Matrix name to check.
        """
        return name in self

    def clear_database(self) -> None:
        """Remove metadata records whose files are absent."""
        with self._transactions.transaction():
            records = self._transactions.execute("SELECT name, file_name FROM matrices").fetchall()
            missing = [(name,) for name, file_name in records if not (self.folder / file_name).is_file()]
            if missing:
                self._transactions.executemany("DELETE FROM matrices WHERE name=?", missing)
        self._invalidate()

    def update_database(self) -> None:
        """Register unrecorded matrix files found in the matrix directory."""
        existing = {record.file_name for record in self}
        for path in self.folder.iterdir():
            if path.name in existing or path.suffix.lower() not in (".omx", ".aem"):
                continue
            candidate = path.name.replace(".", "_").replace(" ", "_")
            name = candidate
            sequence = 0
            while name in self:
                name = f"{candidate}_{sequence}"
                sequence += 1
            self.register_matrix(name, path.name)

    def list(self) -> pd.DataFrame:
        frame = pd.read_sql_query("SELECT * FROM matrices", self._transactions)
        frame["status"] = frame.file_name.map(
            lambda file_name: "" if (self.folder / file_name).is_file() else "file missing"
        )
        return frame

    def _require_resource_idle(self) -> None:
        if self._transactions.in_transaction:
            raise RuntimeError("matrix file helpers cannot run inside a database transaction")

    def __cores_on_disk(self, file_name: str) -> int:
        matrix = AequilibraeMatrix()
        matrix.load(str(self.folder / file_name))
        try:
            return len(matrix.names)
        finally:
            matrix.close()


def _add_resource_note(error: BaseException, message: str):
    if hasattr(error, "add_note"):
        error.add_note(message)
    else:  # pragma: no cover
        logger.error(message)
