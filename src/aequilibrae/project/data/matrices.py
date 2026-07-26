import logging
import os
from os.path import isfile, join

import pandas as pd

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.project.project_table import ProjectTable

logger = logging.getLogger(__name__)


class Matrices(ProjectTable):
    """Gateway into the matrices available/recorded in the model

    .. code-block:: python

        >>> project = create_example(project_path)

        >>> matrices = project.matrices

        # We can list all matrices in the model
        >>> matrices.list()  # doctest: +SKIP

        # get the record for one of them
        >>> record = matrices.get('demand_omx')

        # or the actual matrix data
        >>> mat = matrices.get_matrix('demand_omx')

        >>> project.close()
    """

    name = "matrices"
    key = "name"
    record_name = "MatrixRecord"

    def __init__(self, project):
        super().__init__(project)
        self.fldr = os.path.join(project.project_base_path, "matrices")

    def create(self, name: str, file_name: str, matrix: AequilibraeMatrix = None):
        """Creates a record for a matrix, returning it

        If a matrix is provided, it is exported to ``file_name`` inside the project's
        matrices folder (which must not exist yet). Otherwise the file must already
        be on disk.

        :Arguments:
            **name** (:obj:`str`): Name of the matrix. Must be unique

            **file_name** (:obj:`str`): Name of the file on disk

            **matrix** (:obj:`AequilibraeMatrix`, *Optional*): Matrix to export to ``file_name``

        :Returns:
            **matrix record**: The record for the new matrix
        """
        if name in self:
            raise ValueError(f"There is already a matrix of name ({name}). It must be unique.")

        with self._read_ctx(None) as conn:
            sql = "SELECT count(*) FROM matrices WHERE file_name=?"
            if conn.execute(sql, [file_name]).fetchone()[0] > 0:
                raise ValueError(f"There is already a matrix record for file name ({file_name}). It must be unique.")

        if matrix is not None and matrix.cores > 0:
            if isfile(join(self.fldr, file_name)):
                raise FileExistsError(f"{file_name} already exists. Choose a different name or matrix format")

            mat_format = file_name.split(".")[-1].lower()
            if mat_format not in ["omx", "aem"]:
                raise ValueError("Matrix needs to be either OMX or native AequilibraE")

            matrix.export(join(self.fldr, file_name))
            cores = matrix.cores
        else:
            if not isfile(join(self.fldr, file_name)):
                raise FileExistsError(f"{file_name} does not exist. Cannot create this matrix record")
            cores = self.__cores_on_disk(file_name)

        self.insert(name=name, file_name=file_name, cores=cores)
        logger.warning("Matrix Record has been saved to the database")
        return self.get(name)

    def update(self, key, conn=None, **values):
        """Writes the given columns of one matrix record

        When ``file_name`` changes, the core count is refreshed from the file on disk.
        """
        if "file_name" in values and "cores" not in values:
            values["cores"] = self.__cores_on_disk(values["file_name"])
        super().update(key, conn=conn, **values)

    def delete(self, name: str, conn=None):
        """Deletes a matrix record and the underlying data from disk"""
        record = self.get(name, conn=conn)
        super().delete(record.name, conn=conn)

        if isfile(join(self.fldr, record.file_name)):
            try:
                os.unlink(join(self.fldr, record.file_name))
            except Exception as e:
                logger.error(f"Could not remove matrix from disk: {e.args}")

    def get_matrix(self, name: str) -> AequilibraeMatrix:
        """Returns an AequilibraE matrix available in the project

        Raises an error if the matrix does not exist

        :Arguments:
            **name** (:obj:`str`): Name of the matrix to be loaded

        :Returns:
            **matrix** (:obj:`AequilibraeMatrix`): Matrix object
        """
        record = self.get(name)
        mat = AequilibraeMatrix()
        mat.load(join(self.fldr, record.file_name))
        return mat

    def check_exists(self, name: str) -> bool:
        """Checks whether a matrix with a given name exists

        :Returns:
            **exists** (:obj:`bool`): Does the matrix exist?
        """
        return name in self

    def clear_database(self) -> None:
        """Removes records from the matrices database that do not exist in disk"""

        with self._write_ctx(None) as conn:
            mats = conn.execute("SELECT name, file_name FROM matrices;").fetchall()

            remove = [nm for nm, file in mats if not isfile(join(self.fldr, file))]

            if remove:
                logger.warning(f"Matrix records not found in disk cleaned from database: {','.join(remove)}")

                conn.executemany("DELETE FROM matrices WHERE name=?;", [[x] for x in remove])

    def update_database(self) -> None:
        """Adds records to the matrices database for matrix files found on disk"""
        existing_files = os.listdir(self.fldr)
        paths_for_existing = [rec.file_name for rec in self]

        new_files = [x for x in existing_files if x not in paths_for_existing]
        new_files = [x for x in new_files if os.path.splitext(x)[1] in [".omx", ".aem"]]

        if new_files:
            logger.warning(f"New matrix found on disk. Added to the database: {','.join(new_files)}")

        for fl in new_files:
            mat = AequilibraeMatrix()
            mat.load(join(self.fldr, fl))

            name = None
            if not mat.is_omx():
                name = str(mat.name)

            if not name:
                name = fl

            name = name.replace(".", "_").replace(" ", "_")

            if name in self:
                i = 0
                while f"{name}_{i}" in self:
                    i += 1
                name = f"{name}_{i}"
            mat.close()
            self.create(name, fl)

    def list(self) -> pd.DataFrame:
        """List of all matrices available

        :Returns:
            **df** (:obj:`pd.DataFrame`): Pandas DataFrame listing all matrices available in the model
        """

        def check_if_exists(file_name):
            if os.path.isfile(os.path.join(self.fldr, file_name)):
                return ""
            else:
                return "file missing"

        with self._read_ctx(None) as conn:
            df = pd.read_sql_query("SELECT * FROM matrices;", conn)
            df = df.assign(status="")
            df.status = df.file_name.apply(check_if_exists)

        return df

    def __cores_on_disk(self, file_name: str) -> int:
        mat = AequilibraeMatrix()
        mat.load(join(self.fldr, file_name))
        cores = len(mat.names)
        mat.close()
        del mat
        return cores
