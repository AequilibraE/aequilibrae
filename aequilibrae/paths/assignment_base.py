from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Union
from uuid import uuid4

import numpy as np
import pandas as pd

from aequilibrae.context import get_active_project
from aequilibrae.parameters import Parameters

if False:
    from aequilibrae.transit_assignment.optimal_strategies import OptimalStrategies
    from aequilibrae.traffic_assignment.linear_approximation import LinearApproximation
    from aequilibrae.traffic_assignment.traffic_class import TransportClassBase


class AssignmentBase(ABC):
    def __init__(self, project=None):
        self.procedure_id = uuid4().hex
        self.procedure_date = str(datetime.today())

        proj = project or get_active_project(must_exist=False)
        self.project = proj

        self.parameters = proj.parameters if proj else Parameters().parameters
        self.logger = proj.logger if proj else logging.getLogger("aequilibrae")

        self.classes: List[TransportClassBase] = []
        self.algorithm: str = None
        self.time_field: str = None
        self.assignment: Union[LinearApproximation, OptimalStrategies] = None
        self.free_flow_tt: np.ndarray = None
        self.total_flow: np.ndarray = None
        self.cores: int = None
        self._config = {}

        self.description: str = ""

    def algorithms_available(self) -> list:
        """
        Returns all algorithms available for use

        :Returns:
            :obj:`list`: List of string values to be used with **set_algorithm**
        """
        return self.all_algorithms

    @abstractmethod
    def set_algorithm(self, algorithm: str):
        pass

    @abstractmethod
    def set_cores(self, cores: int) -> None:
        pass

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
        return pd.DataFrame(self.assignment.convergence_report)

    @abstractmethod
    def info(self) -> dict:
        pass

    def set_classes(self, classes: List[TransportClassBase]) -> None:
        """
        Sets Transport classes to be assigned

        :Arguments:
            **classes** (:obj:`List[TransportClassBase]`): List of TransportClass's for assignment
        """

        ids = {x._id for x in classes}
        if len(ids) < len(classes):
            raise ValueError("Classes need to be unique. Your list of classes has repeated items/IDs")
        self.classes = classes  # type: List[TransportClassBase]

    def add_class(self, transport_class: TransportClassBase) -> None:
        """
        Adds a Transport class to the assignment

        :Arguments:
            **transport_class** (:obj:`TransportClassBase`): Transport class
        """

        ids = [x._id for x in self.classes if x._id == transport_class._id]
        if len(ids) > 0:
            raise ValueError("Transport class already in the assignment")

        self.classes.append(transport_class)

    def _check_field(self, field: str, allow_zeros=False) -> None:
        """Throws expection if field is invalid."""
        if not self.classes:
            raise ValueError("You need add at least one transport class first")

        for c in self.classes:
            if field not in c.graph.graph.columns:
                raise ValueError(f"'{field}' not in graph for '{c._id}'")

            if np.any(np.isnan(c.graph.graph[field].values)):
                raise ValueError(f"At least one link for {field} is NaN for '{c._id}'")

            if c.graph.graph[field].values.min() <= 0 and not allow_zeros:
                raise ValueError(f"There is at least one link with zero or negative {field} for '{c._id}'")

    def set_time_field(self, time_field: str) -> None:
        self._check_field(time_field)
        c = self.classes[0]
        self.free_flow_tt = np.zeros(c.graph.graph.shape[0], c.graph.default_types("float"))
        self.free_flow_tt[c.graph.graph.__supernet_id__] = c.graph.graph[time_field]
        self.total_flow = np.zeros(self.free_flow_tt.shape[0], np.float64)
        self.time_field = time_field

    def get_skim_results(self) -> list:
        """Prepares the assignment skim results for all classes

        :Returns:
            **skim list** (:obj:`list`): Lists of all skims with the results for each class
        """
        return {cls._id: cls.results.skims for cls in self.classes}
