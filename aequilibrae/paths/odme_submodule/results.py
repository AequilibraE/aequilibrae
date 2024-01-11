"""
Implementation of results/statistics class to help record and analyse the ODME procedure:
"""

import time
import numpy as np
import pandas as pd

class ODMEResults(object):
    """ Results and statistics of an ODME procedure """
    # Columns for various dataframes:
    # FIGURE OUT WHICH ONES OF THESE ARE ACTUALLY NECESSARY
    DATA_COLS = ["Outer Loop #", "Inner Loop #", "Total Iteration #", "Total Run Time (s)",
        "Loop Time (s)", "Convergence", "Inner Convergence", "class", "link_id", "direction",
        "obs_volume", "assign_volume", "Assigned - Observed"]
    FACTOR_COLS = ['class', 'Outer Loop #', 'Inner Loop #', 'Total Inner Iteration #',
        'mean', 'median', 'std_deviation', 'variance', 'sum', 'min', 'max']

    # For logging different iterations:
    OUTER, INNER = 1, 0

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

        # Time data for logging information:
        #self._total_time = 0
        #self._time = None

        # ODME object:
        self.odme = odme

        # Set up statistics structures
        # Move statistic recording functions to this class
        # Move results objects to this class
        # Clean up stuff
        # From here to continue decide how to more appropriately record results and statistics
        # and make sure the two classes have as little coupling as possible

    # Statistics:
    def get_cumulative_factors(self) -> pd.DataFrame:
        """
        Return the cumulative factors (ratio of final to initial matrix) in a dataframe.
        """
        # Get cumulative factors for each demand matrix
        cumulative_factors = []
        for i, demand_matrix in enumerate(self.odme.demand_matrices):
            factors = np.nan_to_num(demand_matrix / self.odme.init_demand_matrices[i], nan=1)
            cumulative_factors.append(
                pd.DataFrame({
                    "class": [self.odme.class_names[i] for _ in range(demand_matrix.size)],
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

    def log_stats(self) -> None:
        """
        Computes statistics regarding previous iteration and stores them in the statistics list.
        """
        # Compute Statistics:
        old_time = self.odme.time
        self.odme.time = time.time()
        loop_time = self.odme.time - old_time
        self.odme.total_time += loop_time

        # Create Data:
        data = self.odme.count_volumes.copy(deep=True)
        data["Loop Time (s)"] = [loop_time for _ in range(self.odme.num_counts)]
        data["Total Run Time (s)"] = [self.odme.total_time for _ in range(self.odme.num_counts)]
        data["Convergence"] = [self.odme.last_convergence for _ in range(self.odme.num_counts)]
        data["Inner Convergence"] = [self.odme.convergence_change for _ in range(self.odme.num_counts)]
        data["Total Iteration #"] = [self.odme.total_iter for _ in range(self.odme.num_counts)]
        data["Outer Loop #"] = [self.odme.outer for _ in range(self.odme.num_counts)]
        data["Inner Loop #"] = [self.odme.inner for _ in range(self.odme.num_counts)]
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
        for i, factor in enumerate(factors):
            data.append([
                self.odme.class_names[i],
                self.odme.outer,
                self.odme.inner,
                self.odme.total_inner,
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