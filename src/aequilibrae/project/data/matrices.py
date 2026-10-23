import logging
import os
import uuid
from pathlib import Path
from typing import Any, Sequence
import pandas as pd

import openmatrix as omx

from aequilibrae.matrix import AequilibraeMatrix, MatrixStore, from_file

logger = logging.getLogger(__name__)


class Matrices:
    """Matrix project manager."""

    name = "matrices"
    key = "name"
    record_name = "MatrixRecord"

    def __init__(self, matrices_path: str | Path) -> None:
        """Create the matrix table manager.

        :Arguments:
            **matrices_path** (:obj:`str` or :obj:`Path`): Directory containing
            matrix files.
        """
        self.folder: Path = Path(matrices_path)

    def get_available_matrices_and_metadata(self):
        """Returns the a dictionary, the keys being the available matrix omx files, and the values being their
        corresponding metadata."""
        fnames = [f.name for f in self.folder.iterdir() if f.is_file() and f.suffix == ".omx"]
        name_to_metadata = {}
        for file_name in fnames:
            with omx.open_file(self.folder / file_name, mode="r") as matrix:
                name_to_metadata[file_name] = {
                    "procedure": matrix.attrs.get("procedure"),
                    "procedure_id": matrix.attrs.get("procedure_id"),
                    "timestamp": matrix.attrs.get("timestamp"),
                    "description": matrix.attrs.get("description"),
                }

        return name_to_metadata

    def list(self):
        # def list(self) -> pd.DataFrame:
        # frame = pd.read_sql_query("SELECT * FROM matrices", self._connection._connection)
        # frame["status"] = frame.file_name.map(
        #     lambda file_name: "" if (self.folder / file_name).is_file() else "file missing"
        # )
        # return frame
        names_and_metadata = self.get_available_matrices_and_metadata()
        data_by_column = {"file_name": [], "procedure": [], "procedure_id": [], "timestamp": [], "description": []}
        for name, metadata in names_and_metadata.items():
            data_by_column["file_name"].append(name)
            for data_name, data in metadata.items():
                data_by_column[data_name].append(data)
        return pd.DataFrame(data_by_column)

    def file_exists(self, file_name: str) -> bool:
        return file_name in self.get_available_matrices_and_metadata()

    def cores_on_disk(self, file_name: str) -> int:
        with omx.open_file(self.folder / file_name, mode="r") as matrix:
            return len(matrix.list_matrices())

    def get_matrix(
        self,
        file_name: str,
        index_name: str | None = None,
        subset: Sequence[str | dict[str, Any]] | None = None,
    ) -> AequilibraeMatrix | MatrixStore:
        """
        Load a registered matrix, optionally selecting an in-memory subset.

        Without ``subset``, the matrix is opened as a
        :class:`~aequilibrae.matrix.MatrixStore` for generic OMX access.  When
        ``subset`` is provided, the selected matrices are copied into an
        in-memory :class:`~aequilibrae.matrix.AequilibraeMatrix`, ready for
        computation. Subset entries can be matrix names or matrix metadata
        queries, as described by
        :meth:`~aequilibrae.matrix.AequilibraeMatrix.subset`.

        If ``index_name`` is omitted, the only index in the OMX file is used
        when there is exactly one. If there are multiple indices and one is
        named ``main_index``, that index is used. Otherwise, pass
        ``index_name`` explicitly.

        :Arguments:
            **file_name** (:obj:`str`): The filename of the matrix.

            **index_name** (:obj:`str`, *Optional*): Name of the OMX zone
                index.  Inferred when omitted, as described above.

            **subset** (:obj:`Sequence`, *Optional*): Matrix names or metadata
                queries to copy into an in-memory matrix. If omitted, returns a
                :class:`MatrixStore` instead.

        :Returns:
            **matrix** (:class:`~aequilibrae.matrix.AequilibraeMatrix` or
                :class:`~aequilibrae.matrix.MatrixStore`): An in-memory matrix
                when ``subset`` is supplied, otherwise an OMX matrix store.

        :raises ValueError: If the file is not OMX, no usable index exists, or
            multiple indices are ambiguous.
        """
        path = self.folder / file_name
        if file_name not in self.get_available_matrices_and_metadata():
            raise ValueError(
                f"matrices folder does not have a matrix file named '{file_name}', "
                f"available are {self.get_available_matrices_and_metadata().keys()}"
            )
        if path.suffix.lower() != ".omx":
            raise ValueError("Only OMX (.omx) matrices are supported")

        if subset is not None:
            return from_file(path, index_name=index_name, subset=subset, store=False)
        else:
            return from_file(path, index_name=index_name, subset=None, store=True)

    def save_matrix(
        self,
        file_name: str,
        matrix: AequilibraeMatrix | MatrixStore,
        *,
        procedure: str | None = None,
        procedure_id: str | None = None,
        timestamp: str | None = None,
        description: str | None = None,
    ):
        """
        Create matrix metadata and optionally export a matrix.

        File system operations cannot be rolled back as part of the database transaction.

        :Arguments:
            **name** (:obj:`str`): Unique matrix name stored in project metadata.

            **file_name** (:obj:`str`): OMX file name relative to the project's matrix directory.

            **matrix** (:obj:`AequilibraeMatrix` or :obj:`MatrixStore`, *Optional*): Matrix to export.
            If omitted, ``file_name`` must already exist.
        """
        path = self.folder / file_name
        if file_name in [f.name for f in self.folder.iterdir() if f.is_file() and f.suffix == ".omx"]:
            raise FileExistsError(f"There is already a matrix of name ({file_name}). It must be unique.")

        suffix = path.suffix.lower()
        if suffix != ".omx":
            raise ValueError("Only OMX (.omx) matrices are supported")

        # created = False
        temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{suffix}")
        try:
            matrix.save_as_omx(temporary)
            os.replace(temporary, path)
            # created = True
        except BaseException:
            if temporary.exists():
                temporary.unlink()
            raise

        matrix.metadata.update(
            {
                "procedure": procedure,
                "procedure_id": procedure_id,
                "timestamp": timestamp,
                "description": description,
            }
        )

        # return self.get(name)

    # def register_matrix(self, name: str, file_name: str) -> Any:
    #     """Register an existing matrix file.

    #     :Arguments:
    #         **name** (:obj:`str`): Unique matrix name.

    #         **file_name** (:obj:`str`): Existing matrix file name.

    #     :Returns:
    #         **matrix record** (:obj:`Any`): Generated frozen metadata record.
    #     """
    #     return self.create(name, file_name, matrix=None)

    # def delete_matrix(self, name: str) -> None:
    #     """Delete matrix metadata and its underlying file.

    #     :Arguments:
    #         **name** (:obj:`str`): Matrix name to delete.
    #     """
    #     file_name = self.get(name, column="file_name")
    #     path = self.folder / file_name
    #     tombstone = path.with_name(f".{path.name}.{uuid.uuid4().hex}.deleted")
    #     moved = False
    #     if path.exists():
    #         os.replace(path, tombstone)
    #         moved = True
    #     try:
    #         super().delete(name)
    #     except BaseException as primary:
    #         if moved:
    #             try:
    #                 os.replace(tombstone, path)
    #             except BaseException as cleanup:
    #                 primary.add_note(f"matrix is stranded at {tombstone}: {cleanup!r}")
    #         raise
    #     if moved:
    #         tombstone.unlink()

    # def clear_database(self) -> None:
    #     """Remove metadata records whose files are absent."""
    #     with self._connection.transaction() as conn:
    #         records = conn.execute("SELECT name, file_name FROM matrices").fetchall()
    #         missing = [(name,) for name, file_name in records if not (self.folder / file_name).is_file()]
    #         if missing:
    #             conn.executemany("DELETE FROM matrices WHERE name=?", missing)
    #     self._invalidate()

    # def update_database(self) -> None:
    #     """Register unrecorded OMX files found in the matrix directory."""
    #     existing = {record.file_name for record in self}
    #     for path in self.folder.iterdir():
    #         if path.name in existing or path.suffix.lower() != ".omx" or not path.is_file():
    #             continue
    #         candidate = path.name.replace(".", "_").replace(" ", "_")
    #         name = candidate
    #         sequence = 0
    #         while name in self:
    #             name = f"{candidate}_{sequence}"
    #             sequence += 1
    #         self.register_matrix(name, path.name)

    # def sync(self) -> None:
    #     """Remove metadata for absent files, and register unrecorded files."""
    #     self.clear_database()
    #     self.update_database()
