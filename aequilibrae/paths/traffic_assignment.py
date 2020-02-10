from typing import List
from copy import deepcopy
import numpy as np
from aequilibrae.paths.all_or_nothing import allOrNothing
from aequilibrae.paths.linear_approximation import LinearApproximation
from aequilibrae.paths.vdf import VDF
from aequilibrae.paths.traffic_class import TrafficClass

bpr_parameters = ["alpha", "beta"]
all_algorithms = ["all-or-nothing", "msa", "frank-wolfe", "cfw", "bfw"]


class TrafficAssignment(object):
    def __init__(self) -> None:
        self.__dict__["vdf"] = None  # type: VDF
        self.__dict__["classes"] = None  # type: List[TrafficClass]
        self.__dict__["algorithm"] = None  # type: str
        self.__dict__["vdf_parameters"] = None  # type: list
        self.__dict__["time_field"] = None  # type: str
        self.__dict__["capacity_field"] = None  # type: str
        self.__dict__["assignment"] = None  # type: LinearApproximation
        self.__dict__["capacity"] = None  # type: np.ndarray
        self.__dict__["free_flow_tt"] = None  # type: np.ndarray
        self.__dict__["total_flow"] = None  # type: np.ndarray
        self.__dict__["congested_time"] = None  # type: np.ndarray

    def __setattr__(self, instance, value) -> None:
        if instance == "assignment":
            pass
        elif instance == "vdf":
            if value not in ["BPR"]:
                raise ValueError("Volume-delay function {} is not available".format(value))
            value = VDF()
            value.function = "BPR"
        elif instance == "classes":
            if isinstance(value, TrafficClass):
                value = [value]
            elif isinstance(value, list):
                for v in value:
                    if not isinstance(v, TrafficClass):
                        raise ValueError("Traffic classes need to be proper AssignmentClass objects")
            else:
                raise ValueError("Traffic classes need to be proper AssignmentClass objects")
        elif instance == "vdf_parameters":
            if not self.__validate_parameters(value):
                raise ValueError("Parameter set is not valid:  ".format(value))
        elif instance == "time_field":
            if not isinstance(value, str):
                raise ValueError("Value for time field is not string")
        elif instance == "capacity_field":
            if not isinstance(value, str):
                raise ValueError("Value for capacity field is not string")
        else:
            raise ValueError("trafficAssignment class does not have property {}".format(instance))
        self.__dict__[instance] = value

    def set_vdf(self, vdf_function: str) -> None:
        self.vdf = vdf_function

    def set_classes(self, classes: List[TrafficClass]) -> None:
        self.classes = classes

    def available_algorithms(self) -> list:
        return all_algorithms

    # TODO: Create procedure to check that travel times, capacities and vdf parameters are equal across all graphs
    # TODO: We also need procedures to check that all graphs are compatible (i.e. originated from the same network)
    def set_algorithm(self, algorithm: str):
        """
        Chooses the assignment algorithm. e.g. 'frank-wolfe', 'bfw', 'msa'
        """

        # First we instantiate the arrays we will be using over and over
        c = self.classes[0]
        self.__dict__["free_flow_tt"] = np.array(c.graph.graph[self.time_field], copy=True).astype(np.float64)
        self.__dict__["capacity"] = np.array(c.graph.graph[self.capacity_field], copy=True).astype(np.float64)
        self.__dict__["total_flow"] = np.zeros(self.free_flow_tt.shape[0]).astype(np.float64)
        self.__dict__["congested_time"] = np.array(self.free_flow_tt, copy=True).astype(np.float64)

        if algorithm.lower() == "all-or-nothing":
            self.assignment = allOrNothing(self)
        elif algorithm.lower() in ["msa", "frank-wolfe", "cfw", "bfw"]:
            self.assignment = LinearApproximation(self, algorithm.lower())
        else:
            raise AttributeError("Assignment algorithm not available. Choose from: {}".format(",".join(all_algorithms)))

    def set_vdf_parameters(self, par: dict) -> None:
        """
        Sets the parameters for the Volume-delay function. e.g. {'alpha': 0.15, 'beta':4.0}
        """
        self.__dict__['vdf_parameters'] = par
        pars = []
        if self.vdf.function == "BPR":
            for par in ['alpha', 'beta']:
                if isinstance(self.vdf_parameters[par], str):
                    array = np.array(self.classes[0].graph.graph[par], copy=True).astype(np.float64)
                else:
                    array = np.zeros(self.classes[0].graph.graph.shape[0], np.float64)
                    array.fill(self.vdf_parameters[par])
                pars.append(array)
        self.__dict__["vdf_parameters"] = pars

    def set_time_field(self, time_field: str) -> None:
        """
        Sets the graph field that contains free flow travel time -> e.g. 'fftime'
        """
        self.time_field = time_field

    def set_capacity_field(self, capacity_field: str) -> None:
        """
        Sets the graph field that contains link capacity for the assignment period -> e.g. 'capacity1h'
        """
        self.capacity_field = capacity_field

    def load_assignment_spec(self, specs: dict) -> None:
        pass

    def get_spec(self, specs: dict) -> dict:
        """Gets the entire specification of the assignment"""
        return deepcopy(self.__dict__)

    def __validate_parameters(self, kwargs) -> bool:
        if self.vdf == "":
            raise ValueError("First you need to set the Volume-Delay Function to use")

        par = list(kwargs.keys())
        if self.vdf.function == "BPR":
            q = [x for x in par if x not in bpr_parameters] + [x for x in bpr_parameters if x not in par]
        if len(q) > 0:
            raise ValueError("List of functions {} for vdf {} has an inadequate set of parameters".format(q, self.vdf))
        return True

    def execute(self) -> None:
        self.assignment.execute()
