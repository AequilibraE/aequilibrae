import os
from datetime import datetime
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from aequilibrae.context import get_active_project
from aequilibrae.distribution.cython.ipf_core import ipf_core
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.utils.core_setter import resolve_cores


class Ipf:
    """Iterative proportional fitting procedure

    .. code-block:: python

        >>> from aequilibrae.distribution import Ipf

        >>> project = create_example(project_path)

        >>> matrix = project.matrices.get_matrix("demand_omx", subset=["matrix"])

        >>> vectors = pd.DataFrame(
        ...     {"productions": np.zeros(len(matrix.index)), "attractions": np.zeros(len(matrix.index))},
        ...     index=matrix.index,
        ... )

        >>> core = next(iter(matrix))
        >>> vectors["productions"] = matrix[core].sum(axis=1)
        >>> vectors["attractions"] = matrix[core].sum(axis=0)

        >>> ipf_args = {"matrix": matrix,
        ...             "vectors": vectors,
        ...             "row_field": "productions",
        ...             "column_field": "attractions",
        ...             "nan_as_zero": False}

        >>> fratar = Ipf(**ipf_args)
        >>> fratar.fit()

        # We can get back to our OMX matrix in the end
        >>> fratar.output.save_as_omx(Path(my_folder_path) / "to_omx_output.omx")

        >>> project.close()
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
        self.output = None
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
        if len(self.matrix) != 1:
            raise ValueError("Seed matrix must contain exactly one named matrix")
        core = next(iter(self.matrix))
        if not np.issubdtype(self.matrix[core].dtype, np.floating):
            raise ValueError("Seed matrix need to be a float type")

        row_data = self.vectors[self.rows_]
        col_data = self.vectors[self.cols_]

        if not np.issubdtype(row_data.dtype, np.floating):
            raise ValueError("production/rows vector must be a float type")

        if not np.issubdtype(col_data.dtype, np.floating):
            raise ValueError("Attraction/columns vector must be a float type")

        if not np.array_equal(self.matrix.index, self.vectors.index):
            raise ValueError("Indices from vectors do not match those from seed matrix")

        # check balancing:
        sum_rows = np.nansum(row_data)
        sum_cols = np.nansum(col_data)
        self.__col_vector = col_data.to_numpy()  # No need to be editable
        self.__row_vector = row_data.to_numpy()  # No need to be editable
        if abs(sum_rows - sum_cols) > self.parameters["balancing tolerance"]:
            self.error = "Vectors are not balanced"
        else:
            # guarantees that they are precisely balanced
            self.__col_vector = self.__col_vector * (sum_rows / sum_cols)

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

            core = next(iter(self.matrix))
            self.output = self.matrix.subset([core])
            if self.nan_as_zero:
                self.output[core] = np.nan_to_num(self.output[core])

            tot_matrix = np.nansum(self.output[core])

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

            seed = np.array(self.output[core], copy=True)
            iterations, self.gap = ipf_core(
                seed,
                self.__row_vector,
                self.__col_vector,
                max_iterations=max_iter,
                tolerance=conv_criteria,
                cores=self.cpus,
            )
            self.output[core] = seed

            self.report.append(str(iterations) + "   ,   " + str("{:4,.10f}".format(float(np.nansum(self.gap)))))

            self.report.append("")
            self.report.append("Running time: " + str("{:4,.3f}".format(perf_counter() - t)) + "s")

    def save_to_project(self, name: str, file_name: str, project=None):
        """Saves the matrix output to the project file

        :Arguments:
            **name** (:obj:`str`): Name of the desired matrix record

            **file_name** (:obj:`str`): Name for the OMX matrix file

            **project** (:obj:`Project`, *Optional*): Project we want to save the results to.
            Defaults to the active project
        """

        project = project or get_active_project()
        mats = project.matrices
        record = mats.create(
            name,
            file_name,
            self.output,
            procedure="Iterative Proportional fitting",
            procedure_id=self.procedure_id,
            timestamp=self.procedure_date,
        )
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

        self.cpus = resolve_cores(path["system"])
        return path["distribution"][model]
