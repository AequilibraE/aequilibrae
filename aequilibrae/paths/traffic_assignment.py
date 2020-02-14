from typing import List
from copy import deepcopy
from warnings import warn
import numpy as np
from aequilibrae.paths.all_or_nothing import allOrNothing
from aequilibrae.paths.linear_approximation import LinearApproximation
from aequilibrae.paths.vdf import VDF, all_vdf_functions
from aequilibrae.paths.traffic_class import TrafficClass
from aequilibrae import Parameters


class TrafficAssignment(object):
    bpr_parameters = ["alpha", "beta"]
    all_algorithms = ["all-or-nothing", "msa", "frank-wolfe", "cfw", "bfw"]

    def __init__(self) -> None:
        parameters = Parameters().parameters["assignment"]["equilibrium"]
        self.__dict__["rgap_target"] = parameters["rgap"]
        self.__dict__["max_iter"] = parameters["maximum_iterations"]
        self.__dict__["vdf"] = VDF()
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
        self.__dict__["cores"] = None  # type: int

    def __setattr__(self, instance, value) -> None:

        check, value, message = self.__check_attributes(instance, value)
        if check:
            self.__dict__[instance] = value
        else:
            raise ValueError(message)

    def __check_attributes(self, instance, value):
        if instance == "rgap_target":
            if not isinstance(value, float):
                return False, value, 'Relative gap needs to be a float'
            if isinstance(self.assignment, LinearApproximation):
                self.assignment.rgap_target = value
        elif instance == "max_iter":
            if not isinstance(value, int):
                return False, value, 'Number of iterations needs to be an integer'
            if isinstance(self.assignment, LinearApproximation):
                self.assignment.max_iter = value
        elif instance == "vdf":
            v = value.lower()
            if v not in all_vdf_functions:
                return False, value, f"Volume-delay function {value} is not available"
            value = VDF()
            value.function = v
        elif instance == "classes":
            if isinstance(value, TrafficClass):
                value = [value]
            elif isinstance(value, list):
                for v in value:
                    if not isinstance(v, TrafficClass):
                        return False, value, "Traffic classes need to be proper AssignmentClass objects"
            else:
                raise ValueError("Traffic classes need to be proper AssignmentClass objects")
        elif instance == "vdf_parameters":
            if not self.__validate_parameters(value):
                return False, value, f"Parameter set is not valid: {value} "
        elif instance in ["time_field", "capacity_field"]:
            if not isinstance(value, str):
                return False, value, f"Value for {instance} is not string"
        elif instance == 'cores':
            if not isinstance(value, int):
                return False, value, f"Value for {instance} is not integer"
        if instance not in self.__dict__:
            return False, value, f"trafficAssignment class does not have property {instance}"
        return True, value, ''

    def set_vdf(self, vdf_function: str) -> None:
        self.vdf = vdf_function

    def set_classes(self, classes: List[TrafficClass]) -> None:
        self.classes = classes
        self.__collect_data()

    def algorithms_available(self) -> list:
        return self.all_algorithms

    # TODO: Create procedure to check that travel times, capacities and vdf parameters are equal across all graphs
    # TODO: We also need procedures to check that all graphs are compatible (i.e. originated from the same network)
    def set_algorithm(self, algorithm: str):
        """
        Chooses the assignment algorithm. e.g. 'frank-wolfe', 'bfw', 'msa'
        """

        # First we instantiate the arrays we will be using over and over
        if algorithm not in self.all_algorithms:
            raise AttributeError(f"Assignment algorithm not available. Choose from: {','.join(self.all_algorithms)}")

        if algorithm.lower() == "all-or-nothing":
            self.assignment = allOrNothing(self)
        elif algorithm.lower() in ["msa", "frank-wolfe", "cfw", "bfw"]:
            self.assignment = LinearApproximation(self, algorithm.lower())
        else:
            raise Exception('Algorithm not listed in the case selection')

        self.__collect_data()

    def __collect_data(self):
        if not isinstance(self.classes, list):
            return

        c = self.classes[0]
        if self.time_field not in c.graph.graph.dtype.names:
            return

        self.__dict__["free_flow_tt"] = np.array(c.graph.graph[self.time_field], copy=True).astype(np.float64)
        self.__dict__["total_flow"] = np.zeros(self.free_flow_tt.shape[0]).astype(np.float64)
        self.__dict__["congested_time"] = np.array(self.free_flow_tt, copy=True).astype(np.float64)
        self.__dict__["cores"] = c.results.cores

        if self.capacity_field not in c.graph.graph.dtype.names:
            return

        self.__dict__["capacity"] = np.array(c.graph.graph[self.capacity_field], copy=True).astype(np.float64)

    def set_vdf_parameters(self, par: dict) -> None:
        """
        Sets the parameters for the Volume-delay function. e.g. {'alpha': 0.15, 'beta':4.0}
        """
        if self.classes is None or self.vdf.function.lower() not in all_vdf_functions:
            raise Exception('Before setting vdf parameters, you need to set traffic classes and choose a VDF function')
        self.__dict__['vdf_parameters'] = par
        pars = []
        if self.vdf.function in ["BPR"]:
            for p1 in ['alpha', 'beta']:
                if p1 not in par:
                    raise ValueError(f'{p1} should exist in the set of parameters provided')
                p = par[p1]
                if isinstance(self.vdf_parameters[p1], str):
                    array = np.array(self.classes[0].graph.graph[p], copy=True).astype(np.float64)
                else:
                    array = np.zeros(self.classes[0].graph.graph.shape[0], np.float64)
                    array.fill(self.vdf_parameters[p])
                pars.append(array)

                if np.any(np.isnan(array)):
                    warn(f'At least one {p} is NaN. Results will make no sense')

                if p1 == 'alpha':
                    if array.min() < 0:
                        warn(f'At least one {p1} is smaller than zero. Results will make no sense')
                else:
                    if array.min() < 1:
                        warn(f'At least one {p1} is smaller than one. Results will make no sense')

        self.__dict__["vdf_parameters"] = pars

    def set_cores(self, cores: int) -> None:
        """Allows one to set the number of cores to be used AFTER traffic classes have been added"""
        self.cores = cores
        if self.classes is not None:
            for c in self.classes:
                c.results.set_cores(cores)
                c._aon_results.set_cores(cores)
        else:
            raise Exception('You need load traffic classes before overwriting the number of cores')

    def set_time_field(self, time_field: str) -> None:
        """
        Sets the graph field that contains free flow travel time -> e.g. 'fftime'
        """
        self.time_field = time_field
        self.__collect_data()

    def set_capacity_field(self, capacity_field: str) -> None:
        """
        Sets the graph field that contains link capacity for the assignment period -> e.g. 'capacity1h'
        """
        self.capacity_field = capacity_field
        self.__collect_data()

    # def load_assignment_spec(self, specs: dict) -> None:
    #     pass

    # TODO: This function actually needs to return a human-readable dictionary, and not one with
    #       tons of classes. Feeds into the class above
    # def get_spec(self) -> dict:
    #     """Gets the entire specification of the assignment"""
    #     return deepcopy(self.__dict__)

    def __validate_parameters(self, kwargs) -> bool:
        if self.vdf == "":
            raise ValueError("First you need to set the Volume-Delay Function to use")

        par = list(kwargs.keys())
        if self.vdf.function == "BPR":
            q = [x for x in par if x not in self.bpr_parameters] + [x for x in self.bpr_parameters if x not in par]
        if len(q) > 0:
            raise ValueError("List of functions {} for vdf {} has an inadequate set of parameters".format(q, self.vdf))
        return True

    def execute(self) -> None:
        self.assignment.execute()
