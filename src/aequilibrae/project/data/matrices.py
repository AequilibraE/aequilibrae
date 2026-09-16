import logging
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.project.project_table import NonSpatialProjectTable
from aequilibrae.utils.db_utils import NestedTransactionManager

logger = logging.getLogger(__name__)


class Matrices(NonSpatialProjectTable):
    """Matrix project database table manager."""

    name = "matrices"
    key = "name"
    record_name = "MatrixRecord"

    def __init__(self, connection: NestedTransactionManager, matrices_path: str | Path) -> None:
        """Create the matrix table manager.

        :Arguments:
            **connection** (:obj:`NestedTransactionManager`): Manager for the
            project database containing matrix metadata.

            **matrices_path** (:obj:`str` or :obj:`Path`): Directory containing
            matrix files.
        """
        super().__init__(connection)
        self.folder: Path = Path(matrices_path)

    def create(
        self,
        name: str,
        file_name: str,
        matrix: AequilibraeMatrix | None = None,
        *,
        procedure: str | None = None,
        procedure_id: str | None = None,
        timestamp: str | None = None,
        description: str | None = None,
    ) -> Any:
        """
        Create matrix metadata and optionally export a matrix.

        File system operations cannot be rolled back as part of the database transaction.

        :Arguments:
            **name** (:obj:`str`): Unique matrix name stored in project metadata.

            **file_name** (:obj:`str`): Matrix file name relative to the project's matrix directory.

            **matrix** (:obj:`AequilibraeMatrix`, *Optional*): Matrix to export.
            If omitted, ``file_name`` must already exist.

        :Returns:
            **matrix record** (:obj:`Any`): Generated frozen metadata record.
        """
        path = self.folder / file_name
        if name in self:
            raise ValueError(f"There is already a matrix of name ({name}). It must be unique.")
        if self._connection._connection.execute("SELECT 1 FROM matrices WHERE file_name=?", (file_name,)).fetchone():
            raise ValueError(f"There is already a matrix record for file name ({file_name}). It must be unique.")

        created = False
        temporary = None
        if matrix is not None and matrix.cores > 0:
            if path.exists():
                raise FileExistsError(f"{file_name} already exists. Choose a different name or matrix format")
            suffix = path.suffix.lower()
            if suffix != ".omx":
                raise ValueError("Only OMX (.omx) matrixes are supported")
            temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{suffix}")
            try:
                matrix.export(temporary)
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
            self.insert(
                name=name,
                file_name=file_name,
                cores=cores,
                procedure=procedure,
                procedure_id=procedure_id,
                timestamp=timestamp,
                description=description,
            )
        except BaseException as primary:
            if created:
                try:
                    path.unlink()
                except BaseException as cleanup:
                    primary.add_note(f"created matrix remains at {path}: {cleanup!r}")
            raise
        return self.get(name)

    def register_matrix(self, name: str, file_name: str) -> Any:
        """Register an existing matrix file.

        :Arguments:
            **name** (:obj:`str`): Unique matrix name.

            **file_name** (:obj:`str`): Existing matrix file name.

        :Returns:
            **matrix record** (:obj:`Any`): Generated frozen metadata record.
        """
        return self.create(name, file_name, matrix=None)

    def delete_matrix(self, name: str) -> None:
        """Delete matrix metadata and its underlying file.

        :Arguments:
            **name** (:obj:`str`): Matrix name to delete.
        """
        file_name = self.get(name, column="file_name")
        path = self.folder / file_name
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
                    primary.add_note(f"matrix is stranded at {tombstone}: {cleanup!r}")
            raise
        if moved:
            tombstone.unlink()

    def get_matrix(self, name: str) -> AequilibraeMatrix:
        """Load a matrix by metadata name.

        :Arguments:
            **name** (:obj:`str`): Registered matrix name.

        :Returns:
            **matrix** (:obj:`AequilibraeMatrix`): Loaded matrix.
        """
        file_name = self.get(name, column="file_name")
        matrix = AequilibraeMatrix()
        matrix.load(self.folder / file_name)
        return matrix

    def clear_database(self) -> None:
        """Remove metadata records whose files are absent."""
        with self._connection.transaction() as conn:
            records = conn.execute("SELECT name, file_name FROM matrices").fetchall()
            missing = [(name,) for name, file_name in records if not (self.folder / file_name).is_file()]
            if missing:
                conn.executemany("DELETE FROM matrices WHERE name=?", missing)
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

    def sync(self) -> None:
        """Remove metadata for absent files, and register unrecorded files."""
        self.clear_database()
        self.update_database()

    def list(self) -> pd.DataFrame:
        frame = pd.read_sql_query("SELECT * FROM matrices", self._connection._connection)
        frame["status"] = frame.file_name.map(
            lambda file_name: "" if (self.folder / file_name).is_file() else "file missing"
        )
        return frame

    def file_exists(self, name: str) -> bool:
        return (self.folder / self.get(name, column="file_name")).is_file()

    def __cores_on_disk(self, file_name: str) -> int:
        matrix = AequilibraeMatrix()
        matrix.load(self.folder / file_name)
        try:
            return len(matrix.names)
        finally:
            matrix.close()
