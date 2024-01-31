"""
Results/statistics class to help record and analyse the ODME procedure:
"""

import time
import numpy as np
import pandas as pd

class ODMEResults(object):
    """ Results and statistics of an ODME procedure """
    # Columns for various dataframes:
    DATA_COLS = ["Outer Loop #", "Inner Loop #", "Total Iteration #", "Total Run Time (s)",
        "Loop Time (s)", "Convergence", "Inner Convergence", "class", "link_id", "direction",
        "obs_volume", "assign_volume", "Assigned - Observed", "Flow Objective", "Reg Objective"]
    FACTOR_COLS = ['class', 'Outer Loop #', 'Inner Loop #', 'Total Inner Iteration #',
        'mean', 'median', 'std_deviation', 'variance', 'sum', 'min', 'max']

    # For logging different iterations:
    INNER, OUTER, FINAL_LOG = 0, 1, 2

    # FIGURE OUT HOW TO DO TYPEHINT PROPERLY
    def __init__(self, odme) -> None:
        """
        Initialises necessary fields from odme object in order to generate
        statistics and results.
        """
        # Dataframes to log statistical information:
        self.statistics = []

        # Stats on scaling matrices
        self.factor_stats = pd.DataFrame(columns=self.FACTOR_COLS)

        # Iteration number data:
        self.total_iter, self.total_inner, self.outer, self.inner = 0, 0, 0, 0

        # ODME object:
        self.odme = odme

        # Time data for logging information
        self.total_time = 0
        self.time = None

    # Statistics:
    def get_cumulative_factors(self) -> pd.DataFrame:
        """
        Return the cumulative factors (ratio of final to initial matrix) in a dataframe.
        """
        cumulative_factors = []
        for initial, final, name in zip(
            self.odme.original_demands,
            self.odme.demands,
            self.odme.class_names
            ):
            # Get cumulative factors for this demand matrix and store them:
            factors = np.nan_to_num(final / initial, nan=1)
            cumulative_factors.append(
                pd.DataFrame({
                    "class": [name for _ in range(final.size)],
                    "Factors": factors.ravel()
                })
            )

        return pd.concat(cumulative_factors, ignore_index=True)

    def get_all_statistics(self) -> pd.DataFrame:
        """
        Returns dataframe of all assignment values across iterations,
        along with other statistical information (see self.FACTOR_COLS) 
        per iteration, per count volume.
        """
        return pd.concat(self.statistics, ignore_index=True)

    def log_iter(self, iter_type: int) -> None:
        """
        Logs statistics for a given iteration type (inner/outer/final log).

        Parameters:
            iter_type: the type of iteration to log
        """
        if iter_type == self.INNER:
            self.__increment_inner()
        elif iter_type == self.OUTER:
            self.__increment_outer()
        elif iter_type == self.FINAL_LOG:
            self.outer += 1
            self.inner = 0
        else:
            raise ValueError(
                f"\'{iter_type}\' is not a valid type of iteration!"
            )

        self.log_stats()

    def log_stats(self) -> None:
        """
        Computes statistics regarding previous iteration and stores them in the statistics list.
        """
        # Compute Timing:
        old_time = self.time
        self.time = time.time()
        loop_time = self.time - old_time
        self.total_time += loop_time

        # Create Data:
        data = self.odme.count_volumes.copy(deep=True)
        data["Loop Time (s)"] = [loop_time for _ in range(len(data))]
        data["Total Run Time (s)"] = [self.total_time for _ in range(len(data))]
        data["Convergence"] = [self.odme.last_convergence for _ in range(len(data))]
        data["Inner Convergence"] = [self.odme.convergence_change for _ in range(len(data))]
        data["Flow Objective"] = [self.odme.flow_obj for _ in range(len(data))]
        data["Reg Objective"] = [self.odme.reg_obj for _ in range(len(data))]

        data["Total Iteration #"] = [self.total_iter for _ in range(len(data))]
        data["Outer Loop #"] = [self.outer for _ in range(len(data))]
        data["Inner Loop #"] = [self.inner for _ in range(len(data))]

        data["Assigned - Observed"] = (
            self.odme.count_volumes['assign_volume'].to_numpy() -
            self.odme.count_volumes["obs_volume"].to_numpy()
            )

        # Add data to current list of dataframes
        self.statistics.append(data)

    def record_factor_stats(self, factors: list[np.ndarray]) -> None:
        """
        Logs information on the current scaling matrix (ie
        factor statistics per iteration per class).
        """
        # Create statistics on all new factors:
        data = []
        for cls_name, factor in zip(self.odme.class_names, factors):
            data.append([
                cls_name,
                self.outer,
                self.inner,
                self.total_inner,
                np.mean(factor),
                np.median(factor),
                np.std(factor),
                np.var(factor),
                np.sum(factor),
                np.min(factor),
                np.max(factor)
            ])
        new_stats = pd.DataFrame(data, columns=self.FACTOR_COLS)

        # Add the new data to the current list of factor statistics
        self.factor_stats = pd.concat([self.factor_stats, new_stats], ignore_index=True)

    # Extra Utilities:

    def init_timer(self) -> None:
        """
        Initialises the internal times (for statistics purposes).
        
        Should be run when the ODME procedure begins execution.
        """
        self.time = time.time()

    def __increment_outer(self) -> None:
        """
        Increments outer iteration number, increments total iterations and zeros inner
        iteration number.
        """
        self.outer += 1
        self.inner = 0
        self.total_iter += 1

    def __increment_inner(self) -> None:
        """
        Increments inner iteration number and total iteration and total inner iteration number.
        """
        self.inner += 1
        self.total_iter += 1
        self.total_inner += 1
