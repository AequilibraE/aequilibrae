"""Algorithms to **calibrate** synthetic gravity models by maximum likelihood"""

from time import perf_counter

import numpy as np

from aequilibrae.distribution.cython.gravity_core import deterrence_matrix, weighted_cost_sums
from aequilibrae.distribution.cython.ipf_core import ipf_factors
from aequilibrae.distribution.synthetic_gravity_model import SyntheticGravityModel
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.parameters import Parameters
from aequilibrae.utils.core_setter import resolve_cores

# Balancing must be much tighter than the calibration, as the Jacobian comes from finite differences
IPF_TOLERANCE = 1e-10
FINITE_DIFFERENCE_STEP = 1e-5

# Sufficient statistic of each parameter, as weights of trip cost and log-cost, so that log(f) = theta . statistics
STATISTICS = {"EXPO": {"beta": (-1, 0)}, "POWER": {"alpha": (0, -1)}, "GAMMA": {"alpha": (0, 1), "beta": (-1, 0)}}


class GravityCalibration:
    """Calibrate a traditional gravity model

    Available deterrence function forms are: 'EXPO', 'POWER' or 'GAMMA'.

    Parameters are estimated by maximum likelihood, so the calibrated model reproduces the observed average
    trip cost (``EXPO``), the observed average logarithm of trip cost (``POWER``), or both (``GAMMA``).

    .. code-block:: python

        >>> from aequilibrae.distribution import GravityCalibration

        >>> project = create_example(project_path)

        # We load the demand matrix
        >>> demand = project.matrices.get_matrix("demand_omx", subset=["matrix"])

        # We load the skim matrix
        >>> skim = project.matrices.get_matrix("skims", subset=["time_final"])

        >>> args = {"matrix": demand,
        ...         "impedance": skim,
        ...         "function": 'gamma',
        ...         "nan_as_zero": True}
        >>> gravity = GravityCalibration(**args)

        # Solve and save outputs
        >>> gravity.calibrate()
        >>> gravity.model.save(project_path / 'dist_gamma_model.mod')

        >>> project.close()
    """

    def __init__(self, project=None, **kwargs):
        """
        Instantiates the Gravity calibration problem

        :Arguments:
            **matrix** (:obj:`AequilibraeMatrix`): Seed/base trip matrix

            **impedance** (:obj:`AequilibraeMatrix`): Impedance matrix to be used

            **function** (:obj:`str`): Function name to be calibrated. "EXPO", "POWER" or "GAMMA"

            **project** (:obj:`Project`, *Optional*): The Project to connect to.
            By default, uses the currently active project

            **parameters** (:obj:`str`, *Optional*): Convergence parameters. Defaults to those in the parameter file

            **nan_as_zero** (:obj:`bool`, *Optional*): If Nan values should be treated as zero. Defaults to ``False``

        :Results:
            **model** (:obj:`SyntheticGravityModel`): Calibrated model

            **result_matrix** (:obj:`AequilibraeMatrix`): Trips modelled with the calibrated model

            **report** (:obj:`list`): Iteration and convergence report

            **gap** (:obj:`float`): Largest relative change in a parameter in the last iteration
        """

        self.project = project
        self.__required_parameters = ["max trip length", "max iterations", "max error"]
        self.parameters = kwargs.get("parameters", self.__get_parameters())

        self.nan_as_zero = kwargs.get("nan_as_zero", False)
        self.matrix = kwargs.get("matrix")  # type: AequilibraeMatrix
        self.impedance = kwargs.get("impedance")  # type: AequilibraeMatrix
        deterrence_function = str(kwargs.get("function", "")).upper()

        self.result_matrix = None
        self.error = None
        self.comput_core = None
        self.impedance_core = None

        self.itera = 0
        self.max_iter = None
        self.max_error = None
        self.gap = np.inf
        self.cores = resolve_cores((project.parameters if project else Parameters().parameters)["system"])

        self.report = ["  #####    GRAVITY CALIBRATION    #####  ", ""]
        self.report.append("Functional form: " + deterrence_function)
        self.model = SyntheticGravityModel()
        if deterrence_function not in self.model.valid_functions:
            raise ValueError("Function needs to be one of these: " + ", ".join(self.model.valid_functions))
        else:
            self.model.function = deterrence_function

    def calibrate(self):
        """Calibrate the model

        Resulting model is in the *model* class member, and the trips it produces in *result_matrix*
        """
        t = perf_counter()
        self.__check_inputs()
        self.max_iter = self.parameters["max iterations"]
        self.max_error = self.parameters["max error"]

        names = list(STATISTICS[self.model.function])
        initial = {"alpha": float(self.model.function == "POWER"), "beta": 1 / self.__average_cost}
        theta = np.array([initial[name] for name in names])
        statistics, likelihood, _ = self.__evaluate(theta)
        self.itera, self.gap = 0, np.inf
        while self.itera < self.max_iter and self.gap > self.max_error:
            self.itera += 1
            gradient = self.__observed - statistics
            step = self.__newton_step(theta, statistics, gradient)

            # The profile log-likelihood is concave, so backtracking along the Newton direction improves it unless we
            # are at the optimum. Steps below the tolerance are within the precision of the balancing, so we take them
            converging = np.max(np.abs(step) / np.maximum(np.abs(theta), 1e-12)) < self.max_error
            for size in 0.5 ** np.arange(20):
                trial = self.__evaluate(theta + size * step)
                if converging or trial[1] >= likelihood + 1e-4 * size * (gradient @ step):
                    break
            else:
                self.report.append("Could not improve the likelihood any further")
                break

            self.gap = float(np.max(np.abs(size * step) / np.maximum(np.abs(theta + size * step), 1e-12)))
            theta = theta + size * step
            statistics, likelihood, _ = trial
            values = ", ".join(f"{name}={value:.6g}" for name, value in zip(names, theta, strict=True))
            self.report.append(f"Iteration {self.itera}: {values}, error {self.gap:.2E}")

        for name, value in zip(names, theta, strict=True):
            setattr(self.model, name, float(value))
        # The deterrence buffer may hold a rejected trial, so we evaluate the solution again
        _, _, (rows, columns) = self.__evaluate(theta)
        trips = self.__deterrence
        trips *= rows[:, np.newaxis]
        trips *= columns[np.newaxis, :]
        self.result_matrix = AequilibraeMatrix._from_shared_arrays(
            {"gravity": trips}, index=self.matrix.index, index_name=self.matrix.index_name
        )

        if self.gap > self.max_error:
            self.report.append(
                f"DID NOT CONVERGE. Stopped in {self.itera} iterations with a global error of {self.gap}"
            )
        else:
            self.report.append(f"Converged in {self.itera} iterations to a global error of {self.gap}")
        self.report.append(f"Running time: {round(perf_counter() - t, 3)}s")

    def __evaluate(self, theta):
        # Balances the model for these parameters, returning its statistics, profile log-likelihood and factors
        parameters = dict(zip(STATISTICS[self.model.function], theta, strict=True))
        shift = theta @ self.__weights @ self.__cheapest  # The deterrence of the cheapest cell is one
        deterrence_matrix(
            self.__cost,
            self.model.function,
            parameters.get("alpha"),
            parameters.get("beta"),
            self.__max_cost,
            shift,
            self.__deterrence,
            self.cores,
        )
        rows, columns, _, _ = ipf_factors(
            self.__deterrence,
            self.__productions,
            self.__attractions,
            self.__ipf_iterations,
            IPF_TOLERANCE,
            self.cores,
            False,
        )
        _, cost_sum, log_sum = weighted_cost_sums(self.__deterrence, rows, columns, self.__cost, self.cores)
        with np.errstate(divide="ignore"):
            likelihood = (
                self.__productions @ np.log(np.where(self.__productions > 0, rows, 1))
                + self.__attractions @ np.log(np.where(self.__attractions > 0, columns, 1))
                + theta @ self.__observed
                - shift * self.__total
            )
        return self.__weights @ (cost_sum, log_sum), likelihood if np.isfinite(likelihood) else -np.inf, (rows, columns)

    def __newton_step(self, theta, statistics, gradient):
        # The Jacobian of the modelled statistics, from forward differences, is the negative Hessian of the profile
        # log-likelihood, symmetric positive definite in exact arithmetic
        typical = {"alpha": 1.0, "beta": 1 / self.__average_cost}
        jacobian = np.empty((theta.size, theta.size))
        for k, name in enumerate(STATISTICS[self.model.function]):
            step = FINITE_DIFFERENCE_STEP * max(abs(theta[k]), typical[name])
            jacobian[:, k] = (self.__evaluate(theta + step * np.eye(theta.size)[k])[0] - statistics) / step

        # Flooring the eigenvalues guarantees an ascent direction even if differencing noise breaks definiteness
        values, vectors = np.linalg.eigh((jacobian + jacobian.T) / 2)
        values = np.maximum(values, max(values.max(), 0) * 1e-12 + np.finfo(float).tiny)
        return vectors @ ((vectors.T @ gradient) / values)

    def __check_inputs(self):
        if not isinstance(self.impedance, AequilibraeMatrix):
            raise TypeError("Impedance matrix needs to be an instance of AequilibraeMatrix")

        if not isinstance(self.matrix, AequilibraeMatrix):
            raise TypeError("Observed matrix needs to be an instance of AequilibraeMatrix")

        if not np.array_equal(self.impedance.index, self.matrix.index):
            raise ValueError("Indices from impedance and seed matrices do not match")

        if len(self.matrix) != 1 or len(self.impedance) != 1:
            raise ValueError("Observed and impedance matrices must each contain exactly one named matrix")
        self.comput_core = next(iter(self.matrix))
        self.impedance_core = next(iter(self.impedance))

        observed = np.array(self.matrix[self.comput_core], dtype=np.float64)
        cost = np.array(self.impedance[self.impedance_core], dtype=np.float64)
        if self.nan_as_zero:
            np.nan_to_num(observed, copy=False)
            np.nan_to_num(cost, copy=False)

        for values, title in ((observed, "Observed matrix"), (cost, "Impedance matrix")):
            if np.nanmin(values) < 0:
                raise ValueError(f"{title} has negative values")
            if not np.nanmax(values) > 0:
                raise ValueError(f"{title} has only zero values")

        # Augment parameters if we happen to have only passed one
        default_parameters = self.__get_parameters()
        for para in self.__required_parameters:
            if para not in self.parameters:
                self.parameters[para] = default_parameters[para]

        # Observed trips the deterrence function cannot reproduce are left out
        self.__max_cost = self.parameters["max trip length"]
        valid = np.isfinite(cost) & ((cost < self.__max_cost) | (self.__max_cost <= 0))
        valid &= (cost > 0) | (self.model.function == "EXPO")
        observed[np.isnan(observed)] = 0
        all_trips = observed.sum()
        observed[~valid] = 0
        self.__total = observed.sum()
        if self.__total < all_trips:
            self.report.append(
                f"{all_trips - self.__total:,.2f} observed trips are in cells with a missing, zero or excessive cost "
                "and were not used"
            )

        self.__productions, self.__attractions = observed.sum(axis=1), observed.sum(axis=0)
        _, cost_sum, log_sum = weighted_cost_sums(observed, np.ones(len(cost)), np.ones(len(cost)), cost, self.cores)
        self.__average_cost = cost_sum / self.__total
        self.__weights = np.array(list(STATISTICS[self.model.function].values()), dtype=np.float64)
        self.__observed = self.__weights @ (cost_sum, log_sum)
        min_cost = cost.min(where=valid, initial=np.inf)
        self.__cheapest = np.array([min_cost, np.log(min_cost) if min_cost > 0 else 0])
        self.__cost, self.__deterrence = cost, np.empty_like(cost)
        self.__ipf_iterations = Parameters().parameters["distribution"]["ipf"]["max iterations"]

    def __get_parameters(self):
        par = Parameters().parameters
        para = par["distribution"]["ipf"].copy()
        para.update(par["distribution"]["gravity"])
        return para
