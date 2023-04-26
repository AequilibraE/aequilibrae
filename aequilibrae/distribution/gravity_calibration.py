"""
Algorithms to **calibrate** synthetic gravity models with power and exponential functions

The procedures implemented in this code are some of those suggested in
Modelling Transport, 4th Edition, Ortuzar and Willumsen, Wiley 2011
"""
from time import perf_counter

import numpy as np

from aequilibrae.distribution.gravity_application import GravityApplication, SyntheticGravityModel
from aequilibrae.matrix import AequilibraeMatrix, AequilibraeData
from aequilibrae.parameters import Parameters


class GravityCalibration:
    """Calibrate a traditional gravity model

    Available deterrence function forms are: 'EXPO' or 'POWER'. 'GAMMA'

    .. code-block:: python

        >>> from aequilibrae import Project
        >>> from aequilibrae.matrix import AequilibraeMatrix
        >>> from aequilibrae.distribution import GravityCalibration

        >>> project = Project.from_path("/tmp/test_project_gc")

        # We load the impedance matrix
        >>> matrix = AequilibraeMatrix()
        >>> matrix.load('/tmp/test_project_gc/matrices/demand.omx')
        >>> matrix.computational_view(['matrix'])

        # We load the impedance matrix
        >>> impedmatrix = AequilibraeMatrix()
        >>> impedmatrix.load('/tmp/test_project_gc/matrices/skims.omx')
        >>> impedmatrix.computational_view(['time_final'])

        # Creates the problem
        >>> args = {"matrix": matrix,
        ...         "impedance": impedmatrix,
        ...         "row_field": "productions",
        ...         "function": 'expo',
        ...         "nan_as_zero": True}
        >>> gravity = GravityCalibration(**args)

        # Solve and save outputs
        >>> gravity.calibrate()
        >>> gravity.model.save('/tmp/test_project_gc/dist_expo_model.mod')

        # To save the model report in a file
        # with open('/tmp/test_project_gc/report.txt', 'w') as f:
        #     for line in gravity.report:
        #         f.write(f'{line}\\n')
    """

    def __init__(self, project=None, **kwargs):
        """
        Instantiates the Gravity calibration problem

        :Arguments:
            **matrix** (:obj:`AequilibraeMatrix`): Seed/base trip matrix

            **impedance** (:obj:`AequilibraeMatrix`): Impedance matrix to be used

            **function** (:obj:`str`): Function name to be calibrated. "EXPO" or "POWER"

            **project** (:obj:`Project`, optional): The Project to connect to. By default, uses the currently active project

            **parameters** (:obj:`str`, optional): Convergence parameters. Defaults to those in the parameter file

            **nan_as_zero** (:obj:`bool`, optional): If Nan values should be treated as zero. Defaults to True

        :Results:
            **model** (:obj:`SyntheticGravityModel`): Calibrated model

            **report** (:obj:`list`): Iteration and convergence report

            **error** (:obj:`str`): Error description

        """

        self.project = project
        self.__required_parameters = ["max trip length", "max iterations", "max error"]
        self.parameters = kwargs.get("parameters", self.__get_parameters())

        self.nan_as_zero = kwargs.get("nan_as_zero", False)
        self.matrix = kwargs.get("matrix")  # type: AequilibraeMatrix
        self.impedance = kwargs.get("impedance")  # type: AequilibraeMatrix
        deterrence_function = str(kwargs.get("function", "")).upper()

        if self.nan_as_zero:
            self.matrix = self.matrix.copy(memory_only=True)
            self.impedance = self.impedance.copy(memory_only=True)

        self.result_matrix = None
        self.rows = None
        self.columns = None
        self.gap = np.inf

        self.error = None
        self.gravity = None

        self.comput_core = None
        self.impedance_core = None

        self.itera = 0
        self.max_iter = None
        self.max_error = None
        self.gap = np.inf

        self.report = ["  #####    GRAVITY CALIBRATION    #####  ", ""]
        self.report.append("Functional form: " + deterrence_function)
        self.model = SyntheticGravityModel()
        if deterrence_function not in self.model.valid_functions:
            raise ValueError("Function needs to be one of these: " + ", ".join(self.model.valid_functions))
        else:
            self.model.function = deterrence_function

    def __assemble_model(self, b1):
        # NEED TO SET PARAMETERS #
        if self.model.function == "EXPO":
            self.model.beta = float(b1)
        elif self.model.function == "POWER":
            self.model.alpha = float(b1)

    def calibrate(self):
        """Calibrate the model

        Resulting model is in *output* class member
        """
        t = perf_counter()
        # initialize auxiliary variables
        max_cost = self.parameters["max trip length"]
        self.max_iter = self.parameters["max iterations"]
        self.max_error = self.parameters["max error"]

        # Check the inputs
        self.__check_inputs()
        if self.model.function in ["EXPO", "POWER"]:
            # filtering for all costs over limit

            a = 1
            if max_cost > 0:
                a = (self.impedance.matrix_view[:, :] < max_cost).astype(int)

            # weighted average cost
            self.report.append("Iteration: 1")
            cstar = np.nansum(self.impedance.matrix_view[:, :] * self.result_matrix.gravity[:, :] * a) / np.nansum(
                self.result_matrix.gravity[:, :] * a
            )

            b0 = 1 / cstar

            self.__assemble_model(b0)
            c0 = self.__apply_gravity()
            for i in self.gravity.report:
                self.report.append("       " + i)
            self.report.append("")
            self.report.append("")

            bm1 = b0
            bm = b0 * c0 / cstar

            self.report.append("Iteration: 2")
            self.__assemble_model(bm)

            cm = self.__apply_gravity()
            for i in self.gravity.report:
                self.report.append("       " + i)
            self.report.append("Error: " + "{:.2E}".format(float(np.nansum(abs((bm / bm1) - 1)))))
            self.report.append("")
            cm1 = c0

        # While the max iterations has not been reached and the error is still too large
        self.itera = 2
        while self.itera < self.max_iter and self.gap > self.max_error:
            self.report.append("Iteration: " + str(self.itera + 1))
            aux = bm
            bm = ((cstar - cm1) * bm - (cstar - cm) * bm) / (cm - cm1)
            bm1 = aux
            cm1 = cm

            self.__assemble_model(bm1)
            cm = self.__apply_gravity()

            for i in self.gravity.report:
                self.report.append("       " + i)
            self.report.append("Error: " + "{:.2E}".format(float(np.nansum(abs((bm / bm1) - 1)))))
            self.report.append("")

            # compute convergence criteria
            self.gap = abs((bm / bm1) - 1)
            self.itera += 1

        if self.itera == self.max_iter:
            self.report.append(
                "DID NOT CONVERGE. Stopped in  " + str(self.itera) + "  with a global error of " + str(self.gap)
            )
        else:
            self.report.append("Converged in " + str(self.itera) + "  iterations to a global error of " + str(self.gap))
        s = perf_counter() - t
        m, s1 = divmod(s, 60)
        s -= m * 60
        h, m = divmod(m, 60)
        t = "%d:%02d:%2.4f" % (h, m, s)

        self.report.append("Running time: " + t)

    def __check_inputs(self):
        if not isinstance(self.impedance, AequilibraeMatrix):
            raise TypeError("Impedance matrix needs to be an instance of AequilibraEMatrix")

        if not isinstance(self.matrix, AequilibraeMatrix):
            raise TypeError("Observed matrix needs to be an instance of AequilibraEMatrix")

        # Check data dimensions
        if not np.array_equal(self.impedance.index, self.impedance.index):
            raise ValueError("Indices from impedance matrix do not match those from seed matrix")

        # Check if matrices were set for computation
        mats = [(self.matrix, "Observed matrix"), (self.impedance, "Impedance matrix")]
        for matrix, title in mats:
            if matrix.matrix_view is None:
                raise ValueError(f"{title} needs to be set for computation")
            if matrix.matrix_view.ndim > 2:
                raise ValueError(f"{title} computational view needs to be set for a single matrix core")
            if np.nansum(matrix.matrix_view.data) == 0:
                raise ValueError(f"{title} has only zero values")
            if np.nanmin(matrix.matrix_view.data) < 0:
                raise ValueError(f"{title} has negative values")

        # Augment parameters if we happen to have only passed one
        default_parameters = self.__get_parameters()
        for para in self.__required_parameters:
            if para not in self.parameters:
                self.parameters[para] = default_parameters[para]

        # Prepare the data for computation
        self.comput_core = self.matrix.view_names[0]

        self.result_matrix = self.matrix.copy(cores=[self.comput_core], names=["gravity"], memory_only=True)

        self.rows = AequilibraeData()
        self.rows.create_empty(entries=self.matrix.zones, field_names=["rows"], memory_mode=True)
        self.rows.index[:] = self.matrix.index[:]
        self.rows.rows[:] = self.matrix.rows()[:]

        self.columns = AequilibraeData()
        self.columns.create_empty(entries=self.matrix.zones, field_names=["columns"], memory_mode=True)
        self.columns.index[:] = self.matrix.index[:]
        self.columns.columns[:] = self.matrix.columns()[:]

        self.impedance_core = self.impedance.view_names[0]

    def __apply_gravity(self):
        args = {
            "impedance": self.impedance,
            "rows": self.rows,
            "row_field": "rows",
            "columns": self.columns,
            "column_field": "columns",
            "model": self.model,
            "parameters": self.parameters,
            "nan_as_zero": self.nan_as_zero,
        }

        self.gravity = GravityApplication(self.project, **args)
        self.gravity.apply()
        self.result_matrix = self.gravity.output

        return np.nansum(self.impedance.matrix_view[:, :] * self.result_matrix.gravity[:, :]) / np.nansum(
            self.result_matrix.gravity[:, :]
        )

    def __get_parameters(self):
        par = Parameters().parameters
        para = par["distribution"]["ipf"].copy()
        para.update(par["distribution"]["gravity"])
        return para
