"""
Results/statistics class to help record and analyse the ODME procedure:
"""

import time
import numpy as np
import pandas as pd


# ADD FUNCTIONALITY TO SPLIT DATA INTO A: DEPENDENT ON ITERATION, B: DEPENDENT ON COUNT VOLUME
# ADD A WAY TO SAVE B TO CSV's
class ODMEResults(object):
    """ Results and statistics of an ODME procedure """
    # Columns for various dataframes:

    # This one get written to the procedure_report
    ITERATION_COLS = ["class", "Outer Loop #", "Inner Loop #", "Total Iteration #",
        "Total Run Time (s)", "Loop Time (s)", "Convergence", "Inner Convergence",
        "Flow Objective", "Reg Objective",
        'mean_factor', 'median_factor', 'std_deviation_factor',
        'variance_factor', 'min_factor', 'max_factor']

    # This only for debugging
    LINK_COLS = ["class", "link_id", "direction", "Outer Loop #", "Inner Loop #",
                 "obs_volume", "assign_volume", "Assigned - Observed"]

    # For logging different iterations:
    INNER, OUTER, FINAL_LOG = 0, 1, 2

    def __init__(self, odme: 'ODME') -> None:
        """
        Initialises necessary fields from odme object in order to generate
        statistics and results.
        """
        # ODME object:
        self.odme = odme

        # Statistics depending on each iterations
        self.iteration_stats = []
        # Information on factors to be logged with implicit ordering by ODME classes
        self.current_factors = None

        # Statistics depending on each link
        self.link_stats = []

        # Iteration number data:
        self.total_iter, self.outer, self.inner = 0, 0, 0

        # Time data for logging information
        self.total_time = 0
        self.loop_time = None
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

    def get_iteration_statistics(self) -> pd.DataFrame:
        """
        Returns dataframe of all statistics relevant to each iteration.
        See self.ITERATION_COLS.
        """
        if len(self.iteration_stats):
            return pd.concat(self.iteration_stats, ignore_index=True)
        else:
            return pd.DataFrame(columns=self.ITERATION_COLS)

    def get_link_statistics(self) -> pd.DataFrame:
        """
        Returns dataframe of all statistics relevant to each link.
        See self.LINK_COLS.
        """
        if len(self.link_stats):
            return pd.concat(self.link_stats, ignore_index=True)
        else:
            return pd.DataFrame(columns=self.LINK_COLS)

    def log_iter(self, iter_type: int) -> None:
        """
        Logs statistics for a given iteration type (inner/outer/final log).

        Parameters:
            iter_type: the type of iteration to log

        NEEDS UPDATING TO ALLOW FOR US TO RECORD FACTOR STATS AT APPROPRIATE TIMES!!!
        """
        if iter_type == self.INNER:
            self.__prepare_inner()
        elif iter_type == self.OUTER:
            self.__prepare_outer()
        elif iter_type == self.FINAL_LOG:
            self.__prepare_final()
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
        self.__update_times()

        # Update Iteration Statistics
        self.__update_iteration_stats()

        # Update Link Statistics
        self.__update_link_stats()

    def __update_iteration_stats(self) -> None:
        """
        Appends the newest set of statistics for the last iteration.
        """
        # Create Data:
        for cls_name, factor_stats in zip(self.odme.class_names, self.current_factors):
            data = dict()
            
            data["class"] = [cls_name]
            data["Outer Loop #"] = [self.outer]
            data["Inner Loop #"] = [self.inner]
            data["Total Iteration #"] = [self.total_iter]
            data["Total Run Time (s)"] = [self.total_time]
            data["Loop Time (s)"] = [self.loop_time]
            data["Convergence"] = [self.odme.last_convergence]
            data["Inner Convergence"] = [self.odme.convergence_change]
            data["Flow Objective"] = [self.odme.flow_obj]
            data["Reg Objective"] = [self.odme.reg_obj] # Only relevant for reg_spiess
            data["mean_factor"] = factor_stats["mean_factor"]
            data["median_factor"] = factor_stats["median_factor"]
            data["std_deviation_factor"] = factor_stats["std_deviation_factor"]
            data["variance_factor"] = factor_stats["variance_factor"]
            data["min_factor"] = factor_stats["min_factor"]
            data["max_factor"] = factor_stats["max_factor"]

            # Add the new row of statistics
            self.iteration_stats.append(pd.DataFrame(data))

    def __update_times(self):
        """
        Updates the times for the last iteration.
        """
        old_time = self.time
        self.time = time.time()
        self.loop_time = self.time - old_time
        self.total_time += self.loop_time

    def __update_link_stats(self) -> None:
        """
        Appends the newest set of link statistics.
        """
        data = self.odme.count_volumes.copy(deep=True)
        data[ "Outer Loop #"] = [self.outer for _ in range(len(data))]
        data["Inner Loop #"] = [self.inner for _ in range(len(data))]
        data["Assigned - Observed"] = (
            self.odme.count_volumes['assign_volume'].to_numpy() -
            self.odme.count_volumes["obs_volume"].to_numpy()
            )
        self.link_stats.append(data)

    def record_factor_stats(self, factors: list[np.ndarray]) -> None:
        """
        Logs information on the current scaling matrix (ie
        factor statistics per iteration per class).
        """
        # Create statistics on all new factors:
        self.current_factors = []
        for factor in factors:
            self.current_factors.append({
                'mean_factor' : np.mean(factor),
                'median_factor': np.median(factor),
                'std_deviation_factor' : np.std(factor),
                'variance_factor' : np.var(factor),
                'min_factor' : np.min(factor),
                'max_factor' : np.max(factor)
            })

    # Extra Utilities:
    def init_timer(self) -> None:
        """
        Initialises the internal times (for statistics purposes).
        
        Should be run when the ODME procedure begins execution.
        """
        self.time = time.time()

    def __prepare_outer(self) -> None:
        """
        Prepares logging of outer iteration
        """
        self.outer += 1
        self.inner = 0
        self.total_iter += 1
        self.__reset_current_factors()

    def __prepare_inner(self) -> None:
        """
        Prepares logging of inner iteration
        """
        self.inner += 1
        self.total_iter += 1

    def __prepare_final(self) -> None:
        """
        Prepares logging of final iteration
        """
        self.outer += 1
        self.inner = 0
        self.__reset_current_factors()

    def __reset_current_factors(self) -> None:
        """
        Resets the set of current factors to be missing values.
        """
        self.current_factors = []
        for _ in self.odme.classes:
            self.current_factors.append({
                    'mean_factor' : None,
                    'median_factor': None,
                    'std_deviation_factor' : None,
                    'variance_factor' : None,
                    'min_factor' : None,
                    'max_factor' : None
                })
