import logging
import os
import uuid
from pathlib import Path
from typing import Any, Sequence

import openmatrix as omx
import pandas as pd

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

    def list(self) -> pd.DataFrame:
        """
        Returns a DataFrame with a row per matrix with the corresponding metadata.
        """
        df = pd.DataFrame(self.get_available_matrices_and_metadata()).T
        df.index.name = "file_name"
        return df.sort_index()

    def get_available_matrices_and_metadata(self):
        """
        Returns a dictionary, with the keys being the available matrix omx
        files, and the values being their corresponding metadata.
        """
        fnames = [f.name for f in self.folder.iterdir() if f.is_file() and f.suffix == ".omx"]
        name_to_metadata = {}
        for file_name in fnames:
            path = self.folder / file_name
            try:
                with omx.open_file(path, mode="r") as matrix:
                    name_to_metadata[file_name] = {
                        "procedure": matrix.attrs.get("procedure"),
                        "procedure_id": matrix.attrs.get("procedure_id"),
                        "timestamp": matrix.attrs.get("timestamp"),
                        "description": matrix.attrs.get("description"),
                    }
            except OSError as error:
                message = f"Could not read metadata from OMX matrix file {str(path)!r}: {error}"
                logger.error(message)
                raise OSError(message) from error

        return name_to_metadata

    def file_exists(self, file_name: str) -> bool:
        return (self.folder / file_name).is_file()

    def cores_on_disk(self, file_name: str) -> int:
        path = self.folder / file_name
        with omx.open_file(path, mode="r") as matrix:
            return len(matrix.list_matrices())

    def get_matrix(
        self,
        file_name: str,
        index_name: str | None = None,
        subset: Sequence[str | dict[str, Any]] | None = None,
        **kwargs,
    ) -> AequilibraeMatrix | MatrixStore:
        """
        Load a matrix file, optionally selecting an in-memory subset.

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

            **subset** (:obj:`Sequence`, *Optional*): Matrix names or metadata
                queries to copy into an in-memory matrix. If omitted, returns a
                :class:`MatrixStore` instead.

            **kwargs** (:obj:`dict[str, Any]`): Other keyword arguments, passed
                to ``from_file``.

        :Returns:
            **matrix** (:class:`~aequilibrae.matrix.AequilibraeMatrix` or
                :class:`~aequilibrae.matrix.MatrixStore`): An in-memory matrix
                when ``subset`` is supplied, otherwise an OMX matrix store.

        :raises ValueError: If the file is missing, no usable index exists,
            or multiple indices are ambiguous.
        :raises OSError: If the requested OMX file cannot be opened or read.
        """
        path = self.folder / file_name
        if not path.is_file():
            raise ValueError(f"matrices folder does not have a matrix file named '{file_name}'")

        try:
            return from_file(path, index_name=index_name, subset=subset, store=subset is None, **kwargs)
        except OSError as error:
            message = f"Could not load OMX matrix file {str(path)!r}: {error}"
            logger.error(message)
            raise OSError(message) from error

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
        Export a matrix and its metadata to a new OMX file.

        File system operations cannot be rolled back as part of the database transaction.
        Supplied metadata overrides the exported attributes without changing the source matrix.

        :Arguments:
            **file_name** (:obj:`str`): OMX file name relative to the project's matrix directory.

            **matrix** (:obj:`AequilibraeMatrix` or :obj:`MatrixStore`): Matrix to export.

            **procedure** (:obj:`str`, *Optional*): Producing procedure name.

            **procedure_id** (:obj:`str`, *Optional*): Producing procedure ID.

            **timestamp** (:obj:`str`, *Optional*): Result timestamp.

            **description** (:obj:`str`, *Optional*): Human-readable description.

        Omitted metadata retains the source matrix's corresponding attributes.
        """
        path = self.folder / file_name
        suffix = path.suffix
        if suffix != ".omx":
            raise ValueError("Only OMX (.omx) matrices are supported")
        if path.exists():
            raise FileExistsError(f"There is already a matrix of name ({file_name}). It must be unique.")

        temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{suffix}")
        try:
            matrix.save_as_omx(temporary)
            metadata = {
                key: value
                for key, value in {
                    "procedure": procedure,
                    "procedure_id": procedure_id,
                    "timestamp": timestamp,
                    "description": description,
                }.items()
                if value is not None
            }
            if metadata:
                with omx.open_file(temporary, mode="a") as exported:
                    exported.attrs.update(metadata)
            os.replace(temporary, path)
        except BaseException:
            if temporary.exists():
                temporary.unlink()
            raise
