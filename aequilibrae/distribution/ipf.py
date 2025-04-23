import os
from datetime import datetime
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml
from aequilibrae.distribution.ipf_core import ipf_core

from aequilibrae.context import get_active_project
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.project.data.matrix_record import MatrixRecord


class Ipf:
    """Iterative proportional fitting procedure

    .. code-block:: python

        >>> from aequilibrae.distribution import Ipf

        >>> project = create_example(project_path)

        >>> matrix = project.matrices.get_matrix("demand_omx")
        >>> matrix.computational_view()

        >>> vectors = pd.DataFrame({"productions":np.zeros(matrix.zones), "attractions":np.zeros(matrix.zones)}, index=matrix.index)

        >>> vectors["productions"] = matrix.rows()
        >>> vectors["attractions"] = matrix.columns()

        >>> ipf_args = {"matrix": matrix,
        ...             "vectors": vectors,
        ...             "row_field": "productions",
        ...             "column_field": "attractions",
        ...             "nan_as_zero": False}

        >>> fratar = Ipf(**ipf_args)
        >>> fratar.fit()

        # We can get back to our OMX matrix in the end
        >>> fratar.output.export(os.path.join(my_folder_path, "to_omx_output.omx"))
    """

    def __init__(self, project=None, **kwargs):
        """
        Instantiates the IPF problem

        :Arguments:
            **matrix** (:obj:`AequilibraeMatrix`): Seed Matrix

            **vectors** (:obj:`pd.DataFrame`): Dataframe with the vectors to be used for the IPF

            **row_field** (:obj:`str`): Field name that contains the data for the row totals

            **column_field** (:obj:`str`): Field name that contains the data for the column totals

            **parameters** (:obj:`str`, *Optional*): Convergence parameters. Defaults to those in the parameter file

            **nan_as_zero** (:obj:`bool`, *Optional*): If Nan values should be treated as zero. Defaults to ``True``

        :Results:
            **output** (:obj:`AequilibraeMatrix`): Result Matrix

            **report** (:obj:`list`): Iteration and convergence report

            **error** (:obj:`str`): Error description
        """
        self.cpus = 0
        self.parameters = kwargs.get("parameters", self.__get_parameters("ipf"))

        # Seed matrix
        self.matrix = kwargs.get("matrix", None)  # type: AequilibraeMatrix

        # NaN as zero
        self.nan_as_zero = kwargs.get("nan_as_zero", True)

        # row vector
        self.__col_vector: np.array = np.zeros([])
        self.__row_vector: np.array = np.zeros([])
        self.vectors = kwargs.get("vectors", None)
        self.rows_ = kwargs.get("row_field", None)
        self.cols_ = kwargs.get("column_field", None)
        self.output_name = kwargs.get("output")

        self.output = AequilibraeMatrix()
        self.error = None
        self.__required_parameters = ["convergence level", "max iterations", "balancing tolerance"]
        self.error_free = True
        self.report = ["  #####    IPF computation    #####  ", ""]
        self.gap = None
        self.procedure_date = ""
        self.procedure_id = ""

    def __check_data(self):
        self.__check_parameters()

        # check data types
        if not isinstance(self.vectors, pd.DataFrame):
            raise TypeError("Row vector needs to be a Pandas DataFrame")

        if not isinstance(self.matrix, AequilibraeMatrix):
            raise TypeError("Seed matrix needs to be an instance of AequilibraeMatrix")

        # Check data type
        if not np.issubdtype(self.matrix.dtype, np.floating):
            raise ValueError("Seed matrix need to be a float type")

        row_data = self.vectors[self.rows_]
        col_data = self.vectors[self.cols_]

        if not np.issubdtype(row_data.dtype, np.floating):
            raise ValueError("production/rows vector must be a float type")

        if not np.issubdtype(col_data.dtype, np.floating):
            raise ValueError("Attraction/columns vector must be a float type")

        if not np.array_equal(self.matrix.index, self.vectors.index):
            raise ValueError("Indices from vectors do not match those from seed matrix")

        # Check if matrix was set for computation
        if self.matrix.matrix_view is None:
            raise ValueError("Matrix needs to be set for computation")
        else:
            if len(self.matrix.matrix_view.shape[:]) > 2:
                raise ValueError("Matrix' computational view needs to be set for a single matrix core")

        # check balancing:
        sum_rows = np.nansum(row_data)
        sum_cols = np.nansum(col_data)
        self.__col_vector = col_data.to_numpy() * (sum_rows / sum_cols)
        self.__row_vector = row_data.to_numpy()
        if abs(sum_rows - sum_cols) > self.parameters["balancing tolerance"]:
            self.error = "Vectors are not balanced"
        else:
            # guarantees that they are precisely balanced
            self.__col_vector = col_data.to_numpy() * (sum_rows / sum_cols)

    def __check_parameters(self):
        for i in self.__required_parameters:
            if i not in self.parameters:
                self.error = "Parameters error. It needs to be a dictionary with the following keys: "
                for t in self.__required_parameters:
                    self.error = self.error + t + ", "
        if self.error:
            raise ValueError(self.error)

    def fit(self):
        """Runs the IPF instance problem to adjust the matrix

        Resulting matrix is the *output* class member
        """
        self.procedure_id = uuid4().hex
        self.procedure_date = str(datetime.today())
        t = perf_counter()
        self.__check_data()
        if self.error_free:
            max_iter = self.parameters["max iterations"]
            conv_criteria = self.parameters["convergence level"]

            if self.matrix.is_omx():
                self.output = AequilibraeMatrix()
                self.output.create_from_omx(omx_path=self.matrix.file_path, cores=self.matrix.view_names)
                self.output.computational_view()
            else:
                self.output = self.matrix.copy(self.output_name)
            if self.nan_as_zero:
                self.output.matrix_view[:, :] = np.nan_to_num(self.output.matrix_view)[:, :]

            tot_matrix = np.nansum(self.output.matrix_view[:, :])

            # Reporting
            self.report.append("Target convergence criteria: " + str(conv_criteria))
            self.report.append("Maximum iterations: " + str(max_iter))
            self.report.append("")
            self.report.append(f"Rows/columns: {self.vectors.shape[0]}")

            self.report.append("Total of seed matrix: " + "{:28,.4f}".format(float(tot_matrix)))
            self.report.append("Total of target vectors: " + "{:25,.4f}".format(float(np.nansum(self.__row_vector))))
            self.report.append("")
            self.report.append("Iteration,   Convergence")
            self.gap = conv_criteria + 1

            seed = np.array(self.output.matrix_view[:, :], copy=True)
            iter, self.gap = ipf_core(
                seed,
                self.__row_vector,
                self.__col_vector,
                max_iterations=max_iter,
                tolerance=conv_criteria,
                cores=self.cpus,
            )
            self.output.matrix_view[:, :] = seed[:, :]

            self.report.append(str(iter) + "   ,   " + str("{:4,.10f}".format(float(np.nansum(self.gap)))))

            self.report.append("")
            self.report.append("Running time: " + str("{:4,.3f}".format(perf_counter() - t)) + "s")

    def save_to_project(self, name: str, file_name: str, project=None) -> MatrixRecord:
        """Saves the matrix output to the project file

        :Arguments:
            **name** (:obj:`str`): Name of the desired matrix record

            **file_name** (:obj:`str`): Name for the matrix file name. AEM and OMX supported

            **project** (:obj:`Project`, *Optional*): Project we want to save the results to.
            Defaults to the active project
        """

        project = project or get_active_project()
        mats = project.matrices
        record = mats.new_record(name, file_name, self.output)
        record.procedure_id = self.procedure_id
        record.timestamp = self.procedure_date
        record.procedure = "Iterative Proportional fitting"
        record.save()
        return record

    def __tot_rows(self, matrix):
        return np.nansum(matrix, axis=1)

    def __tot_columns(self, matrix):
        return np.nansum(matrix, axis=0)

    def __factor(self, marginals, targets):
        f = np.divide(targets, marginals)  # We compute the factors
        f[f == np.NINF] = 1  # And treat the errors
        return f

    def __get_parameters(self, model):
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        with open(path + "/parameters.yml", "r") as yml:
            path = yaml.safe_load(yml)

        self.cpus = int(path["system"]["cpus"])
        return path["distribution"][model]
