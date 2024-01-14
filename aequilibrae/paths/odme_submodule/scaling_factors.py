"""
Implementation of ODME algorithms to obtain scaling matrices at each iteration:
"""

from typing import Tuple
import numpy as np
import scipy.stats as spstats

class ScalingFactors(object):
    """ ODME Algorithms (Scaling Factor Generation) """
    ALL_ALGORITHMS = ["gmean", "spiess", "reg_spiess"]

    # FIGURE OUT TYPEHINT FOR ODME
    def __init__(self, odme, algorithm: str) -> None:
        """
        Initialises necessary fields from odme object in order to generate
        a set of scaling matrices for the current iteration of the odme 
        procedure.
        
        Parameters:
            odme: the ODME object containing all fields pertaining to the odme procedure
            algorithm: the algorithm to use to generate scaling factors.
        """
        self.algo_name = algorithm
        self.__set_algorithm()

        self._count_volumes = odme.count_volumes
        self.class_names = odme.class_names
        self.num_classes = len(self.class_names)
        self._demand_dims = odme.demand_dims
        self._sl_matrices = odme._sl_matrices
        self.demand_matrices = odme.demand_matrices
        self.init_demand_matrices = odme.demand_matrices
        if algorithm in ["reg_spiess"]:
            self._alpha = odme.alpha
            self._beta = 1 - odme.alpha

        self.odme = odme

    def __set_algorithm(self) -> None:
        """
        Set the algorithm to be used to obtain scaling factors.
        """
        if self.algo_name == "gmean":
            self._algorithm = self.__geometric_mean

        elif self.algo_name == "spiess":
            self._algorithm = self.__spiess

        elif self.algo_name == "reg_spiess":
            self._algorithm = self.__reg_spiess
        
        else:
            raise ValueError(
                f"Invalid algorithm name: {self.algo_name}"
                "Valid algorithms are: "
                '\n'.join(self.ALL_ALGORITHMS)
            )

    def generate(self) -> list[np.ndarray]:
        """
        Returns scaling factors for this iteration of the ODME procedure.
        """
        return self._algorithm()

    # gmean (Geometric Mean):
    def __geometric_mean(self) -> list[np.ndarray]:
        """
        Calculates scaling factor based on geometric mean of ratio between 
        proportionally (via SL matrix) assigned flow & observed flows.

        MULTI-CLASS UNDER DEVELOPMENT! (REQUIRES TESTING)
        """
        # Steps:
        # 1. For each link create a factor f_a given by \hat v_a / v_a
        # 2. Create a matrix of factors for each link where for a given OD pair i,
        #    the factor at i is f_a if the corresponding value in the SL matrix
        #    is non-zero, and 1 otherwise.
        # 3. Return the geometric mean of all the factor matrices component-wise
        # NOTE - This may be slower due to holding all these matrices in memory
        # simultaneously. It is possible to do this e.g element-wise or row-wise
        # to save on memory usage. Need to test this later on.
        # NOTE - by not approximating step size we may over-correct massively.

        scaling_factors = []
        c_v = self._count_volumes
        # Steps 1 & 2:
        for i, name in enumerate(self.class_names):
            observed = c_v[c_v['class'] == name]

            # If there are no observations leave matrix unchanged
            if len(observed) == 0:
                scaling_factors.append(np.ones(self._demand_dims[i]))
                continue

            factors = np.empty((len(observed), *(self._demand_dims[i])))
            for j, row in self._count_volumes.iterrows():
                # Create factor matrix:
                if row["obs_volume"] != 0 and row['assign_volume'] != 0:

                    # Modulate factor by select link dependency:
                    link_factor = (row['obs_volume'] / row['assign_volume']) - 1
                    sl_matrix = self._sl_matrices[self.odme.get_sl_key(row)]
                    factor_matrix = (sl_matrix * link_factor)

                    # Apply factor limiting:
                    # factor_matrix = np.clip(factor_matrix, -self.GMEAN_LIMIT, self.GMEAN_LIMIT)

                    # Center factors at 1:
                    factor_matrix = factor_matrix + 1

                # If assigned or observed value is 0 we cannot do anything right now
                else:
                    factor_matrix = np.ones(self._demand_dims[i])
                
                # Add factor matrix
                factors[j, :, :] = factor_matrix

            # If the assigned volume was 0 (or both 0) no OD pair can have any effect
            factors = np.nan_to_num(factors, nan=1, posinf=1, neginf=1)

        # Add the factors for this class to the array:
        scaling_factors.append(spstats.gmean(factors, axis=0))

        # Step 3:
        return scaling_factors

    # spiess (Gradient Descent - Objective Function (2,0))
    def __spiess(self) -> list[np.ndarray]:
        """
        Calculates scaling factor based on gradient descent method via SL matrix,
        assigned flow & observed flows as described by Spiess (1990) - REFERENCE HERE

        MULTI-CLASS UNDER DEVELOPMENT!
        """
        # Derivative matrices for spiess algorithm:
        gradient_matrices = self.__get_derivative_matrices_spiess()

        # Get optimum step sizes for current iteration:
        step_sizes = self.__get_step_sizes_spiess(gradient_matrices)

        # Get scaling factors:
        scaling_factors = [1 - (step_sizes[i] * gradient_matrices[i]) for i in range(self.num_classes)]
        return scaling_factors

    def __get_derivative_matrices_spiess(self) -> list[np.ndarray]:
        """
        Returns derivative matrix (see Spiess (1990) - REFERENCE HERE)

        MULTI-CLASS UNDER DEVELOPMENT!
        """
        # THIS CAN BE SIGNIFICANTLY SIMPLIFIED - SEE NOTES

        # There are probably some numpy/cython ways to do this in parallel and
        # without storing too many things in memory.
        derivatives = []
        # Create a derivative matrix for each user class:
        for i, user_class in enumerate(self.class_names):
            observed = self._count_volumes[self._count_volumes['class'] == user_class]
            factors = np.empty((len(observed), *(self._demand_dims[i])))
            for i, row in observed.iterrows():
                sl_matrix = self._sl_matrices[self.odme.get_sl_key(row)]
                factors[i, :, :] = sl_matrix * (row['assign_volume'] - row['obs_volume'])

            # Add derivative matrix to list of derivatives:
            derivatives.append(np.sum(factors, axis=0))

        return derivatives

    def __get_step_sizes_spiess(self, gradients: list[np.ndarray]) -> list[float]:
        """
        Returns estimate of optimal step size (see Spiess (1990) - REFERENCE HERE)

        Parameters:
            gradients: The previously calculated gradient matrices - required for calculating 
                derivative of link flows with respect to step size.

        MULTI-CLASS UNDER DEVELOPMENT!
        """
        # THIS CAN BE SIGNIFICANTLY SIMPLIFIED - SEE NOTES

        bounds = self.__get_step_size_limits_spiess(gradients)

        # SOME MINOR OPTIMISATIONS CAN BE DONE HERE IN TERMS OF WHAT PRECISELY TO CALCULATE:
        # Calculate step-sizes (lambdas) for each gradient matrix:
        lambdas = []
        for i, user_class in enumerate(self.class_names):
            class_counts = self._count_volumes[self._count_volumes['class'] == user_class]

            # Calculating link flow derivatives:
            flow_derivatives = self.__get_flow_derivatives_spiess(
                user_class,
                self.demand_matrices[i],
                gradients[i]
            )

            # Calculate minimising step length:
            errors = class_counts['obs_volume'].to_numpy() - class_counts['assign_volume'].to_numpy()
            min_lambda = np.sum(flow_derivatives * errors) / np.sum(np.square(flow_derivatives))

            # This can only happen if all flow derivatives are 0 - ie we should not bother perturbing matrix
            if np.isnan(min_lambda):
                min_lambda = 0

            # Check minimising lambda is within bounds:
            lambdas.append(self.__enforce_bounds(min_lambda, *bounds[i]))

        return lambdas

    def __get_flow_derivatives_spiess(self,
        user_class: str,
        gradient: np.ndarray,
        demand: np.ndarray) -> np.ndarray:
        """
        Returns an array of flow derivatives (v_a' in paper in SMP Teams)
        for the particular class.

        Parameters:
            user_class: the name of the class from which to find flow derivatives
            gradient: the gradient for the relevant class
            demand: the demand matrix for this class
        """
        class_counts = self._count_volumes[self._count_volumes['class'] == user_class]

        # Calculating link flow derivatives:
        flow_derivatives = np.empty(len(class_counts))
        for j, row in class_counts.iterrows():
            sl_matrix = self._sl_matrices[self.odme.get_sl_key(row)]
            flow_derivatives[j] = -np.sum(demand * sl_matrix * gradient)
        
        return flow_derivatives


    def __enforce_bounds(self, value: float, upper: float, lower: float) -> float:
        """
        Given a value, and an upper and lower bounds returns the value
        if it sits between the bounds, and otherwise whichever bounds was
        violated.

        E.g. self.__enforce_bounds(1.1, 10, 2) = 2

        Parameters:
            value: the values to check
            upper: the upper bound
            lower: the lower bound
        """
        if value > upper:
            return upper # Upper Bound Violated
        elif value < lower:
            return lower # Lower Bound Violated
        else: # Bounds Not Violated
            return value

    def __get_step_size_limits_spiess(self,
            gradients: list[np.ndarray]) -> list[Tuple[float, float]]:
        """
        Returns bounds for step size in order of upper bound, then lower bound (see Spiess
        (1990) - REFERENCE HERE) for each gradient matrix.

        Parameters:
            gradient: The currently calculating gradient matrix - required for calculating 
                derivative of link flows with respect to step size.

        MULTI-CLASS UNDER DEVELOPMENT!
        """
        # THIS CAN BE SIGNIFICANTLY SIMPLIFIED - SEE NOTES
        bounds = []
        # Create each bound and check for edge cases with empty sets:
        for i, gradient in enumerate(gradients):
            # Upper bound:
            upper_mask = np.logical_and(self.demand_matrices[i] > 0, gradient > 0)
            if np.any(upper_mask):
                upper_lim = 1 / np.min(gradient[upper_mask])
            else:
                upper_lim = float('inf')

            # Lower bound:
            lower_mask = np.logical_and(self.demand_matrices[i] > 0, gradient < 0)
            if np.any(lower_mask):
                lower_lim = 1 / np.max(gradient[lower_mask])
            else:
                lower_lim = float('-inf')

            bounds.append((upper_lim, lower_lim)) # Tuple[float, float]

        return bounds

    # regularised spiess (Gradient Descent - Objective Function (2,2))
    def __reg_spiess(self) -> list[np.ndarray]:
        """
        Calculates scaling factor based on gradient descent method via SL matrix,
        assigned flow & observed flows for the algorithm by Spiess with a
        regularisation term attached.

        NOT YET IMPLEMENTED!
        CURRENTLY ONLY IMPLEMENTING FOR SINGLE CLASS!
        MULTI-CLASS NOT YET IMPLEMENTED!
        """
        # Derivative matrices for spiess algorithm:
        gradient_matrices = self.__get_derivative_matrices_reg_spiess()

        # Get optimum step sizes for current iteration:
        step_sizes = self.__get_step_sizes_spiess(gradient_matrices)

        # Get scaling factors:
        scaling_factors = [1 - (step_sizes[i] * gradient_matrices[i]) for i in range(self.num_classes)]
        return scaling_factors

    def __get_derivative_matrices_reg_spiess(self) -> list[np.ndarray]:
        """
        Returns derivative matrix (see notes in SMP Teams)

        NOT YET IMPLEMENTED!
        CURRENTLY ONLY IMPLEMENTING FOR SINGLE CLASS!
        MULTI-CLASS NOT YET IMPLEMENTED!
        NOTE - THIS RETURNS A LIST OF GRADIENT MATRICES, BUT IT HAS NOT BEEN DERIVED THAT THIS
        IS THE APPROPRIATE GRADIENT FOR MULTI-CLASS (EVEN IF IT LIKELY IS THE CASE)
        """
        spiess_grads = self.__get_derivative_matrices_spiess()
        g_hat = self.init_demand_matrices
        reg_grads = [demand - g_hat[i] for i, demand in enumerate(self.demand_matrices)]
        return [
            (self._alpha * spiess) + (self._beta * reg_grads[i]) 
            for i, spiess in enumerate(spiess_grads)
            ]
