import os
from os.path import isfile
from sqlite3 import Connection
import pandas as pd
from aequilibrae.project.table_loader import TableLoader
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.project.data.matrix_record import MatrixRecord


class Matrices:
    __items = {}

    def __init__(self, project):
        self.__project = project
        self.conn = project.conn  # type: Connection
        self.curr = self.conn.cursor()
        self.fldr = os.path.join(project.project_base_path, 'matrices')

        tl = TableLoader()
        matrices_list = tl.load_table(self.curr, 'matrices')
        existing_list = [lt['name'] for lt in matrices_list]
        if matrices_list:
            self.__properties = list(matrices_list[0].keys())
        for lt in matrices_list:
            if lt['name'] not in self.__items:
                lt['folder'] = self.fldr
                self.__items[lt['name']] = MatrixRecord(lt)

        to_del = [key for key in self.__items.keys() if key not in existing_list]
        for key in to_del:
            del self.__items[key]

    def clear_database(self) -> None:
        """Removes records from the matrices database that do not exist in disk"""

        self.curr.execute('Select name, file_name from matrices;')

        remove = [nm for nm, file in self.curr.fetchall() if isfile(os.path.join(self.fldr, file))]

        if remove:
            self.__project.logger.warning(f'Matrix records not found in disk cleaned from database: {",".join(remove)}')

            remove = [[x] for x in remove]
            self.curr.executemany('DELETE from matrices where name=?;', remove)
            self.conn.commit()

    def update_database(self) -> None:
        """Adds records to the matrices database for matrix files found on disk"""
        pass

    def list(self) -> pd.DataFrame:
        """

        Returns:
             df (:obj:`pd.DataFrame`:) Pandas DataFrame listing all matrices available in the model
        """

        def check_if_exists(file_name):
            if os.path.isfile(os.path.join(self.fldr, file_name)):
                return ''
            else:
                return 'file missing'

        df = pd.read_sql_query('Select * from matrices;', self.conn)
        df = df.assign(status='')
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
        pass
