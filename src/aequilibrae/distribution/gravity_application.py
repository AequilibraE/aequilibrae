import logging
from datetime import datetime
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd

from aequilibrae.context import get_active_project
from aequilibrae.distribution.cython.gravity_core import deterrence_matrix
from aequilibrae.distribution.cython.ipf_core import ipf_core
from aequilibrae.distribution.synthetic_gravity_model import SyntheticGravityModel
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.parameters import Parameters
from aequilibrae.utils.core_setter import resolve_cores

logger = logging.getLogger(__name__)


class GravityApplication:
    """Applies a synthetic gravity model.

    Model is an instance of SyntheticGravityModel class.

    Impedance is an instance of AequilibraeMatrix.

    Vectors are a pandas DataFrame.

    .. code-block:: python

        >>> import pandas as pd
        >>> from aequilibrae.distribution import SyntheticGravityModel, GravityApplication

        >>> project = create_example(project_path)

        # We define the model we will use
        >>> model = SyntheticGravityModel()

        # Before adding a parameter to the model, you need to define the model functional form
        # You can select one of GAMMA, EXPO or POWER.
        >>> model.function = "GAMMA"

        # Only the parameter(s) applicable to the chosen functional form will have any effect
        >>> model.alpha = 0.1
        >>> model.beta = 0.0001

        # We load the impedance matrix
        >>> matrix = project.matrices.get_matrix("skims", subset=["distance_blended"])

        # We create the vectors we will use
        >>> query = "SELECT zone_id, population, employment FROM zones;"
        >>> with project.db_connection as conn:
        ...     df = pd.read_sql(query, conn)
        >>> df = df.sort_values(by="zone_id").set_index("zone_id")

        # You create the vectors you would have
        >>> df = df.assign(productions=df.population * 3.0)
        >>> df = df.assign(attractions=df.employment * 4.0)
        >>> vectors = df[["productions", "attractions"]]

        # Balance the vectors
        >>> vectors["attractions"] *= vectors["productions"].sum() / vectors["attractions"].sum()

        # Create the problem object
        >>> args = {"impedance": matrix,
        ...         "vectors": vectors,
        ...         "row_field": "productions",
        ...         "model": model,
        ...         "column_field": "attractions",
        ...         "nan_as_zero":True
        ...         }
        >>> gravity = GravityApplication(**args)

        # Solve and save the outputs
        >>> gravity.apply()
        >>> gravity.output.save_as_omx(project_path / 'matrices' / 'gravity_omx.omx')

        >>> project.close()
    """

    def __init__(self, project=None, **kwargs):
        """
        Instantiates the IPF problem

        :Arguments:
            **model** (:obj:`SyntheticGravityModel`): Synthetic gravity model to apply

            **impedance** (:obj:`AequilibraeMatrix`): Impedance matrix to be used

            **vectors** (:obj:`pd.DataFrame`): Dataframe with data for row and column totals

            **row_field** (:obj:`str`): Field name that contains the data for the row totals

            **column_field** (:obj:`str`): Field name that contains the data for the column totals

            **project** (:obj:`Project`, *Optional*): The Project to connect to. By default, uses the currently
            active project

            **core_name** (:obj:`str`, *Optional*): Name for the output matrix core. Defaults to "gravity"

            **parameters** (:obj:`str`, *Optional*): Convergence parameters. Defaults to those in the parameter file

            **nan_as_zero** (:obj:`bool`, *Optional*): If NaN values should be treated as zero. Defaults to ``True``

        :Results:
            **output** (:obj:`AequilibraeMatrix`): Result Matrix

            **report** (:obj:`list`): Iteration and convergence report

            **error** (:obj:`str`): Error description
        """

        self.project = project
        self.parameters = kwargs.get("parameters", self.__get_parameters())

        self.vectors = kwargs.get("vectors")
        self.rows_ = kwargs.get("row_field", None)

        self.cols_ = kwargs.get("column_field", None)

        self.impedance: AequilibraeMatrix | None = kwargs.get("impedance")  # type:
        self.model = kwargs.get("model")  # type: SyntheticGravityModel
        self.core_name: str = kwargs.get("output_core", "gravity")
        self.nan_as_zero = kwargs.get("nan_as_zero", False)
        self.output = None  # type: AequilibraeMatrix
        self.gap = np.inf
        self.procedure_date = ""
        self.procedure_id = ""
        self.cores = resolve_cores((project.parameters if project else Parameters().parameters)["system"])

    def apply(self):
        """Runs the Gravity Application instance as instantiated

        Resulting matrix is the *output* class member
        """
        self.__check_data()
        self.procedure_id = uuid4().hex
        self.procedure_date = str(datetime.today())
        t = perf_counter()

        assert self.impedance is not None
        cost = self.impedance[self.impedance_core]
        if self.nan_as_zero:
            cost = np.nan_to_num(cost)

        # The deterrence matrix, without trips at or above the max trip length, is the seed of the balancing
        trips = deterrence_matrix(
            cost,
            self.model.function,
            self.model.alpha,
            self.model.beta,
            self.parameters["max trip length"],
            cores=self.cores,
        )
        if self.model.k_factors is not None:
            groups = self.model.k_factors.index.get_indexer(self.model.districts.loc[self.impedance.index])
            factors = self.model.k_factors.to_numpy()[:, groups]
            for row, group in zip(trips, groups, strict=True):
                row *= factors[group]
        rows = np.nan_to_num(self.vectors[self.rows_].to_numpy(dtype=np.float64))
        columns = np.nan_to_num(self.vectors[self.cols_].to_numpy(dtype=np.float64))
        iterations, self.gap = ipf_core(
            trips,
            rows,
            columns * (rows.sum() / columns.sum()),
            self.parameters["max iterations"],
            self.parameters["convergence level"],
            self.cores,
        )
        self.output = AequilibraeMatrix._from_shared_arrays(
            {self.core_name: trips}, index=self.impedance.index, index_name=self.impedance.index_name
        )

        self.report.append(f"Iterations: {iterations}, convergence: {self.gap:.10f}")
        self.report.append("Total of matrix: " + "{:15,.4f}".format(float(trips.sum())))
        self.report.append("Intrazonal flow: " + "{:15,.4f}".format(float(np.diagonal(trips).sum())))
        self.report.append(f"Running time: {round(perf_counter() - t, 3)}")

    def save_to_project(self, name: str, file_name: str, project=None) -> None:
        """Saves the matrix output to the project file

        :Arguments:
            **name** (:obj:`str`): Name of the desired matrix record

            **file_name** (:obj:`str`): Name for the OMX matrix file

            **project** (:obj:`Project`, *Optional*): Project we want to save the results to.
            Defaults to the active project
        """

        project = project or get_active_project()
        project.matrices.create(
            name,
            file_name,
            self.output,
            procedure="Synthetic gravity trip distribution",
            procedure_id=self.procedure_id,
            timestamp=self.procedure_date,
            description=f"Synthetic gravity trip distribution. {self.model.function}",
        )

    def __get_parameters(self):
        par = self.project.parameters if self.project else Parameters().parameters
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
        if not isinstance(self.vectors, pd.DataFrame):
            raise TypeError("Row vector needs to be a Pandas DataFrame")

        if not isinstance(self.impedance, AequilibraeMatrix):
            raise TypeError("Impedance matrix needs to be an instance of AequilibraeMatrix")

        if not np.array_equal(self.impedance.index, self.vectors.index):
            raise ValueError("Indices from vectors do not match those from seed matrix")

        if len(self.impedance) != 1:
            raise ValueError("Impedance matrix must contain exactly one named matrix")
        self.impedance_core = next(iter(self.impedance))

        # check balancing:
        sum_rows = np.nansum(self.vectors[self.rows_])
        sum_cols = np.nansum(self.vectors[self.cols_])
        if abs(sum_rows - sum_cols) > self.parameters["balancing tolerance"]:
            raise ValueError("Vectors are not balanced")
