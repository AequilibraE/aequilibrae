import os
from os.path import isfile, join
import pandas as pd
from aequilibrae.project.table_loader import TableLoader
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.project.data.matrix_record import MatrixRecord


class Matrices:
    """Gateway into the matrices available/recorded in the model"""

    def __init__(self, project):
        self.project = project
        self.logger = project.logger
        self.__items = {}
        self.__fields = []

        self.fldr = os.path.join(project.project_base_path, "matrices")

        self.conn = project.connect()
        self.curr = self.conn.cursor()
        tl = TableLoader()
        matrices_list = tl.load_table(self.curr, "matrices")
        self.__fields = [x for x in tl.fields]
        if matrices_list:
            self.__properties = list(matrices_list[0].keys())
        for lt in matrices_list:
            if lt["name"] in self.__items:
                if not self.__items[lt["name"]]._exists:
                    del self.__items[lt["name"]]
            if lt["name"] not in self.__items:
                if isfile(join(self.fldr, lt["file_name"])):
                    lt["fldr"] = self.fldr
                    self.__items[lt["name"].lower()] = MatrixRecord(lt, project)

    def reload(self):
        """Discards all memory matrices in memory and loads recreate them"""
        self.__items.clear()
        self.__init__(self.project)

    def clear_database(self) -> None:
        """Removes records from the matrices database that do not exist in disk"""

        self.curr.execute("Select name, file_name from matrices;")

        remove = [nm for nm, file in self.curr.fetchall() if not isfile(join(self.fldr, file))]

        if remove:
            self.logger.warning(f'Matrix records not found in disk cleaned from database: {",".join(remove)}')

            remove = [[x] for x in remove]
            self.curr.executemany("DELETE from matrices where name=?;", remove)
            self.conn.commit()

    def update_database(self) -> None:
        """Adds records to the matrices database for matrix files found on disk"""
        existing_files = os.listdir(self.fldr)
        paths_for_existing = [mat.file_name for mat in self.__items.values()]

        new_files = [x for x in existing_files if x not in paths_for_existing]
        new_files = [x for x in new_files if os.path.splitext(x.lower())[1] in [".omx", ".aem"]]

        if new_files:
            self.logger.warning(f'New matrix found on disk. Added to the database: {",".join(new_files)}')

        for fl in new_files:
            mat = AequilibraeMatrix()
            mat.load(join(self.fldr, fl))

            name = None
            if not mat.is_omx():
                name = str(mat.name).lower()

            if not name:
                name = fl.lower()

            name = name.replace(".", "_").replace(" ", "_")

            if name in self.__items:
                i = 0
                while f"{name}_{i}" in self.__items:
                    i += 1
                name = f"{name}_{i}"
            rec = self.new_record(name, fl)
            rec.save()

    def list(self) -> pd.DataFrame:
        """List of all matrices available
        Returns:
             df (:obj:`pd.DataFrame`:) Pandas DataFrame listing all matrices available in the model
        """

        def check_if_exists(file_name):
            if os.path.isfile(os.path.join(self.fldr, file_name)):
                return ""
            else:
                return "file missing"

        df = pd.read_sql_query("Select * from matrices;", self.conn)
        df = df.assign(status="")
        df.status = df.file_name.apply(check_if_exists)

        return df

    def get_matrix(self, matrix_name: str) -> AequilibraeMatrix:
        """Returns an AequilibraE matrix available in the project

        Raises an error if matrix does not exist

        Args:
            matrix_name (:obj:`str`:) Name of the matrix to be loaded

        Returns:
            matrix (:obj:`AequilibraeMatrix`:) Matrix object
        """

        return self.get_record(matrix_name).get_data()

    def get_record(self, matrix_name: str) -> MatrixRecord:
        """Returns a model Matrix Record for manipulation in memory"""

        if matrix_name.lower() not in self.__items:
            raise Exception("There is no matrix record with that name")

        if not self.__items[matrix_name.lower()]._exists:
            raise Exception("This matrix was deleted during this session")

        return self.__items[matrix_name.lower()]

    def check_exists(self, name: str) -> bool:
        """Checks whether a matrix with a given name exists

        Returns:
            exists (:obj:`bool`:) Does the matrix exist?
        """
        return name.lower() in self.__items

    def delete_record(self, matrix_name: str) -> None:
        """Deletes a Matrix Record from the model and attempts to remove from disk"""
        mr = self.get_record(matrix_name)
        mr.delete()

    def new_record(self, name: str, file_name: str, matrix=AequilibraeMatrix()) -> MatrixRecord:
        """Creates a new record for a matrix in disk, but does not save it

        If the matrix file is not already on disk, it will fail

        Args:
            *name* (:obj:`str`): Name of the matrix
            *file_name* (:obj:`str`): Name of the file on disk

        Return:
            *matrix_record* (:obj:`MatrixRecord`): A matrix record that can be manipulated in memory before saving
        """

        if name in self.__items:
            raise ValueError(f"There is already a matrix of name ({name}). It must be unique.")

        for mat in self.__items.values():
            if mat.file_name == file_name:
                raise ValueError(f"There is already a matrix record for file name ({file_name}). It must be unique.")

        if matrix.cores > 0:
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
            mat = AequilibraeMatrix()
            mat.load(join(self.fldr, file_name))
            cores = mat.cores
            mat.close()
            del mat

        tp = {key: None for key in self.__fields}
        tp["name"] = name
        tp["file_name"] = file_name
        tp["cores"] = cores
        mr = MatrixRecord(tp, self.project)
        mr.save()
        self.__items[name.lower()] = mr
        self.logger.warning("Matrix Record has been saved to the database")
        return mr

    def _clear(self):
        """Eliminates records from memory. For internal use only"""
        self.__items.clear()
