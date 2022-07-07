import sys
import numpy as np
import os
import tempfile
from time import perf_counter
import glob
import logging
from uuid import uuid4
from datetime import datetime
import importlib.util as iutil
from aequilibrae.context import get_active_project
from aequilibrae.distribution.ipf import Ipf
from aequilibrae.distribution.synthetic_gravity_model import SyntheticGravityModel
from aequilibrae.matrix import AequilibraeMatrix, AequilibraeData
from aequilibrae import Parameters

sys.dont_write_bytecode = True

spec = iutil.find_spec("openmatrix")
has_omx = spec is not None


class GravityApplication:
    """Applies a synthetic gravity model

    Model is an instance of SyntheticGravityModel class
    Impedance is an instance of AequilibraEMatrix
    Row and Column vectors are instances of AequilibraeData

    ::

        import pandas as pd
        import sqlite3

        from aequilibrae.matrix import AequilibraeMatrix
        from aequilibrae.matrix import AequilibraeData

        from aequilibrae.distribution import SyntheticGravityModel
        from aequilibrae.distribution import GravityApplication

        # We define the model we will use
        model = SyntheticGravityModel()

        # Before adding a parameter to the model, you need to define the model functional form
        model.function = "GAMMA" # "EXPO" or "POWER"

        # Only the parameter(s) applicable to the chosen functional form will have any effect
        model.alpha = 0.1
        model.beta = 0.0001

        # Or you can load the model from a file
        model.load('path/to/model/file')

        # We load the impedance matrix
        matrix = AequilibraeMatrix()
        matrix.load('path/to/impedance_matrix.aem')
        matrix.computational_view(['distance'])

        # We create the vectors we will use
        conn = sqlite3.connect('path/to/demographics/database')
        query = "SELECT zone_id, population, employment FROM demographics;"
        df = pd.read_sql_query(query,conn)

        index = df.zone_id.values[:]
        zones = index.shape[0]

        # You create the vectors you would have
        df = df.assign(production=df.population * 3.0)
        df = df.assign(attraction=df.employment * 4.0)

        # We create the vector database
        args = {"entries": zones, "field_names": ["productions", "attractions"],
            "data_types": [np.float64, np.float64], "memory_mode": True}
        vectors = AequilibraeData()
        vectors.create_empty(**args)

        # Assign the data to the vector object
        vectors.productions[:] = df.production.values[:]
        vectors.attractions[:] = df.attraction.values[:]
        vectors.index[:] = zones[:]

        # Balance the vectors
        vectors.attractions[:] *= vectors.productions.sum() / vectors.attractions.sum()

        # Create the problem object
        args = {"impedance": matrix,
                "rows": vectors,
                "row_field": "productions",
                "model": model,
                "columns": vectors,
                "column_field": "attractions",
                "output": 'path/to/output/matrix.aem',
                "nan_as_zero":True
                }
        gravity = GravityApplication(**args)

        # Solve and save the outputs
        gravity.apply()
        gravity.output.export('path/to/omx_file.omx')
        with open('path.to/report.txt', 'w') as f:
            for line in gravity.report:
                f.write(f'{line}\n')
    """

    def __init__(self, project=None, **kwargs):
        """
        Instantiates the Ipf problem

        Args:
            model (:obj:`SyntheticGravityModel`): Synthetic gravity model to apply

            impedance (:obj:`AequilibraeMatrix`): Impedance matrix to be used

            rows (:obj:`AequilibraeData`): Vector object with data for row totals

            row_field (:obj:`str`): Field name that contains the data for the row totals

            columns (:obj:`AequilibraeData`): Vector object with data for column totals

            column_field (:obj:`str`): Field name that contains the data for the column totals

            project (:obj:`Project`, optional): The Project to connect to. By default, uses the currently active project

            core_name (:obj:`str`, optional): Name for the output matrix core. Defaults to "gravity"

            parameters (:obj:`str`, optional): Convergence parameters. Defaults to those in the parameter file

            nan_as_zero (:obj:`bool`, optional): If Nan values should be treated as zero. Defaults to True

        Results:
            output (:obj:`AequilibraeMatrix`): Result Matrix

            report (:obj:`list`): Iteration and convergence report

            error (:obj:`str`): Error description
        """

        self.project = project or get_active_project()
        self.__required_parameters = ["max trip length"]
        self.__required_model = ["function", "parameters"]

        self.parameters = kwargs.get("parameters", self.__get_parameters())

        self.rows = kwargs.get("rows")
        self.row_field = kwargs.get("row_field", None)

        self.columns = kwargs.get("columns")
        self.column_field = kwargs.get("column_field", None)

        self.impedance = kwargs.get("impedance")
        self.model = kwargs.get("model")  # type: SyntheticGravityModel
        self.core_name = kwargs.get("output_core", "gravity")
        self.output_name = AequilibraeMatrix().random_name()
        self.nan_as_zero = kwargs.get("nan_as_zero", False)
        self.output = None
        self.gap = np.inf
        self.logger = logging.getLogger("aequilibrae")
        self.procedure_date = ""
        self.procedure_id = ""
        self.__ipf = None  # type: Ipf

    def apply(self):
        """Runs the Gravity Application instance as instantiated

        Resulting matrix is the *output* class member
        """
        self.__check_data()
        self.procedure_id = uuid4().hex
        self.procedure_date = str(datetime.today())
        t = perf_counter()
        max_cost = self.parameters["max trip length"]
        # We create the output
        self.output = self.impedance.copy(self.output_name, cores=self.impedance.view_names, names=[self.core_name])
        self.output.computational_view([self.core_name])
        if self.nan_as_zero:
            self.output.matrix_view[:, :] = np.nan_to_num(self.output.matrix_view)[:, :]

        # We apply the function
        self.__apply_function()

        # We zero those cells that have a trip length above the limit
        if max_cost > 0:
            a = (self.output.matrix_view[:, :] < max_cost).astype(int)
            self.output.matrix_view[:, :] = a * self.output.matrix_view[:, :]

        # We adjust the total of the self.output
        total_factor = np.nansum(self.rows.data[self.row_field]) / np.nansum(self.output.matrix_view[:, :])
        self.output.matrix_view[:, :] = self.output.matrix_view[:, :] * total_factor

        # And adjust with a fratar
        self.__ipf = Ipf(
            project=self.project,
            matrix=self.output,
            rows=self.rows,
            columns=self.columns,
            column_field=self.column_field,
            row_field=self.row_field,
            nan_as_zero=self.nan_as_zero,
        )

        # We use the model application parameters in case they were provided
        # not the standard way of using this tool)
        for p in self.__ipf.parameters:
            if p in self.parameters:
                self.__ipf.parameters[p] = self.parameters[p]

        # apply Fratar
        self.__ipf.fit()
        self.output = self.__ipf.output
        self.gap = self.__ipf.gap

        self.report.extend(self.__ipf.report[1:] + ["", ""])
        self.report.append("Total of matrix: " + "{:15,.4f}".format(float(np.nansum(self.output.matrix_view))))
        intrazonals = float(np.nansum(np.diagonal(self.output.matrix_view)))
        self.report.append("Intrazonal flow: " + "{:15,.4f}".format(intrazonals))
        self.report.append(f"Running time: {round(perf_counter() - t, 3)}")

        for i in glob.glob(tempfile.gettempdir() + "*.aem"):
            try:
                os.unlink(i)
            except PermissionError as err:
                self.logger.warning(f"Could not remove {err.filename}")

    def save_to_project(self, name: str, file_name: str) -> None:
        """Saves the matrix output to the project file

        Args:
            name (:obj:`str`): Name of the desired matrix record
            file_name (:obj:`str`): Name for the matrix file name. AEM and OMX supported
        """

        mats = self.project.matrices
        record = mats.new_record(name, file_name, self.output)
        record.procedure_id = self.procedure_id
        record.timestamp = self.procedure_date
        record.procedure = "Synthetic gravity trip distribution"
        record.description = f"Synthetic gravity trip distribution. {self.model.function}"
        record.save()

    def __get_parameters(self):
        par = self.project.parameters
        para = par["distribution"]["ipf"].copy()
        para.update(par["distribution"]["gravity"])
        return para

    def __check_data(self):
        self.report = ["  #####    GRAVITY APPLICATION    #####  ", ""]

        if not isinstance(self.model, SyntheticGravityModel):
            self.error_free = False
            raise TypeError("Model is not an instance of SyntheticGravityModel")

        self.report.append("Model specification:")
        self.report.append("    Function: " + self.model.function)
        if self.model.alpha is not None:
            self.report.append("    alpha: " + str(self.model.alpha))

        if self.model.beta is not None:
            self.report.append("    beta: " + str(self.model.beta))

        self.report.append("")

        # check dimensions
        # check data types
        if not isinstance(self.rows, AequilibraeData):
            raise TypeError("Row vector needs to be an instance of AequilibraeData")

        if not isinstance(self.columns, AequilibraeData):
            raise TypeError("Column vector needs to be an instance of AequilibraeData")

        if not isinstance(self.impedance, AequilibraeMatrix):
            raise TypeError("Impedance matrix needs to be an instance of AequilibraeMatrix")

        # Check data dimensions
        if not np.array_equal(self.rows.index, self.columns.index):
            raise ValueError("Indices from row vector do not match those from column vector")

        if not np.array_equal(self.impedance.index, self.columns.index):
            raise ValueError("Indices from vectors do not match those from seed matrix")

        # Check if matrix was set for computation
        if self.impedance.matrix_view is None:
            raise ValueError("Matrix needs to be set for computation")
        else:
            if len(self.impedance.matrix_view.shape[:]) > 2:
                raise ValueError("Matrix' computational view needs to be set for a single matrix core")

        # check balancing:
        sum_rows = np.nansum(self.rows.data[self.row_field])
        sum_cols = np.nansum(self.columns.data[self.column_field])
        if abs(sum_rows - sum_cols) > self.parameters["balancing tolerance"]:
            raise ValueError("Vectors are not balanced")
        else:
            # guarantees that they are precisely balanced
            self.columns.data[self.column_field][:] = self.columns.data[self.column_field][:] * (sum_rows / sum_cols)

        self.__check_parameters()

    def __check_parameters(self):
        # Check if parameters are configured properly
        for p in self.__required_parameters:
            if p not in self.parameters:
                self.error = "Parameters error. It needs to be a dictionary with the following keys: "
                for t in self.__required_parameters:
                    self.error = self.error + t + ", "
                break

    def __apply_function(self):
        self.core_name = self.output.view_names[0]
        for i in range(self.rows.entries):
            p = self.rows.data[self.row_field][i]
            a = self.columns.data[self.column_field][:]

            if self.model.function == "EXPO":
                self.output.matrix_view[i, :] = np.exp(-self.model.beta * self.impedance.matrix_view[i, :]) * p * a

            elif self.model.function == "POWER":
                # self.output.matrices[self.core_name][i, :] = (np.power(self.impedance.matrix_view[i, :, 0], - self.model.alpha) * p * a)[:]
                self.output.matrix_view[i, :] = (np.power(self.impedance.matrix_view[i, :], -self.model.alpha) * p * a)[
                    :
                ]
            elif self.model.function == "GAMMA":
                self.output.matrix_view[i, :] = (
                    np.power(self.impedance.matrix_view[i, :], self.model.alpha)
                    * np.exp(-self.model.beta * self.impedance.matrix_view[i, :])
                    * p
                    * a
                )[:]

        # Deals with infinite and NaNs
        infinite = np.isinf(self.output.matrix_view[:, :]).astype(int)
        non_inf = np.ones_like(self.output.matrix_view[:, :]) - infinite
        self.output.matrix_view[:, :] = self.output.matrix_view[:, :] * non_inf
        np.nan_to_num(self.output.matrix_view[:, :])
