"""Internal shared assignment lifecycle and reporting support."""

from abc import ABC, abstractmethod
from datetime import datetime
from uuid import uuid4

import numpy as np
import pandas as pd

from aequilibrae.parameters import Parameters
from aequilibrae.utils.core_setter import (
    clamp_cores,
    resolve_cores,
    resolve_elementwise_cores,
    resolve_threading_threshold,
)


class AssignmentBase(ABC):
    def __init__(self, project=None):
        self.procedure_id = uuid4().hex
        self.procedure_date = str(datetime.today())

        self.project = project
        self.parameters = project.parameters if project is not None else Parameters().parameters

        self.classes: list = []
        self.algorithm: str = None
        self.time_field: str = None
        self.assignment = None
        self.free_flow_tt: np.ndarray = None
        self.total_flow: np.ndarray = None
        self.cores: int = None
        self.elementwise_cores: int = None
        self.threading_threshold: int = None
        self._config = {}

        self.description: str = ""

    def _resolve_project(self, project):
        project = self.project if project is None else project
        if project is None:
            raise ValueError("An explicit project is required to save assignment results")
        return project

    def algorithms_available(self) -> list:
        """
        Returns all algorithms available for use

        :Returns:
            :obj:`list`: Available algorithm names
        """
        return self.all_algorithms

    def execute(self, log_specification=True) -> None:
        """Processes assignment"""
        if log_specification:
            self.log_specification()
        self.assignment.execute()

    @abstractmethod
    def log_specification(self):
        pass

    @abstractmethod
    def save_results(self, table_name: str, keep_zero_flows=True, project=None) -> None:
        pass

    @abstractmethod
    def results(self) -> pd.DataFrame:
        pass

    def report(self) -> pd.DataFrame:
        """Returns the assignment convergence report

        :Returns:
           **DataFrame** (:obj:`pd.DataFrame`): Convergence report
        """
        if self.assignment is None:
            raise RuntimeError("Run the assignment before requesting its report")
        return pd.DataFrame(self.assignment.convergence_report)

    @abstractmethod
    def info(self) -> dict:
        pass

    def get_skim_results(self) -> dict:
        """Return assignment skims by class name."""
        return {cls._id: cls.results.skims for cls in self.classes}


class AssignmentResultsBase(ABC):
    """Shared reporting and thread settings for traffic and transit results."""

    def __init__(self):
        self._system_parameters = Parameters().parameters["system"]
        self.set_cores(resolve_cores(self._system_parameters), resolve_threading_threshold(self._system_parameters))

    def set_cores(self, cores, threading_threshold=None, elementwise_cores=None):
        """Set thread counts without reallocating or clearing existing results."""
        self.cores = clamp_cores(cores)
        if threading_threshold is not None:
            if not isinstance(threading_threshold, int):
                raise ValueError("Threading threshold needs to be an integer")
            self.threading_threshold = threading_threshold
        self.elementwise_cores = (
            clamp_cores(elementwise_cores)
            if elementwise_cores is not None
            else resolve_elementwise_cores(self._system_parameters, self.cores)
        )

    @abstractmethod
    def reset(self):
        pass
