import os
import pandas as pd

from aequilibrae.matrix import AequilibraeMatrix


class Matrices:
    def __init__(self, project):
        self.__project = project
        self.conn = project.conn
        self.fldr = os.path.join(project.project_base_path, 'matrices')

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
