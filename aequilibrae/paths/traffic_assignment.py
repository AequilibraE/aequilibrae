from typing import List
from copy import deepcopy
import numpy as np
from aequilibrae.paths.all_or_nothing import allOrNothing
from aequilibrae.paths.msa import MSA
from aequilibrae.paths.fw import FW
from aequilibrae.paths.vdf import VDF
from aequilibrae.paths.traffic_class import TrafficClass

bpr_parameters = ['alpha', 'beta']
all_algorithms = ['all-or-nothing', 'msa', 'frank-wolfe', 'bfw']


class TrafficAssignment(object):
    def __init__(self) -> None:
        self.__dict__['vdf'] = None  # type: VDF
        self.__dict__['classes'] = None  # type: List[TrafficClass]
        self.__dict__['algorithm'] = None  # type: str
        self.__dict__['vdf_parameters'] = None  # type: dict
        self.__dict__['time_field'] = None  # type: str
        self.__dict__['capacity_field'] = None  # type: str
        self.__dict__['assignment'] = None  # type: MSA

    def __setattr__(self, instance, value) -> None:
        if instance == 'assignment':
            pass
        elif instance == 'vdf':
            if value not in ['BPR']:
                raise ValueError('Volume-delay function {} is not available'.format(value))
            value = VDF()
            value.function = 'BPR'
        elif instance == 'classes':
            if isinstance(value, TrafficClass):
                value = [value]
            elif isinstance(value, list):
                for v in value:
                    if not isinstance(v, TrafficClass):
                        raise ValueError('Traffic classes need to be proper AssignmentClass objects')
            else:
                raise ValueError('Traffic classes need to be proper AssignmentClass objects')
        elif instance == 'vdf_parameters':
            if not self.__validate_parameters(value):
                raise ValueError('Parameter set is not valid:  '.format(value))
        elif instance == 'time_field':
            if not isinstance(value, str):
                raise ValueError('Value for time field is not string')
        elif instance == 'capacity_field':
            if not isinstance(value, str):
                raise ValueError('Value for capacity field is not string')
        else:
            raise ValueError('trafficAssignment class does not have property {}'.format(instance))
        self.__dict__[instance] = value

    def set_vdf(self, vdf_function: str) -> None:
        self.vdf = vdf_function

    def set_classes(self, classes: List[TrafficClass]) -> None:
        self.classes = classes

    def available_algorithms(self) -> list:
        return (all_algorithms)

    # TODO: Create procedure to check that travel times, capacities and vdf parameters are equal across all graphs
    # TODO: We also need procedures to check that all graphs are compatible (i.e. originated from the same network)
    def set_algorithm(self, algorithm: str):
        """
        Chooses the assignment algorithm. e.g. 'frank-wolfe'
        """
        if algorithm.lower() == 'all-or-nothing':
            self.assignment = allOrNothing(self)
        elif algorithm.lower() == 'msa':
            self.assignment = MSA(self)
        elif algorithm.lower() == 'frank-wolfe':
            self.assignment = FW(self)
        else:
            raise AttributeError("Assignment algorithm not available. Choose from: {}".format(','.join(all_algorithms)))

    def set_vdf_parameters(self, **kwargs) -> None:
        """
        Sets the parameters for the Volume-delay function. e.g. {'alpha': 0.15, 'beta':4.0}
        """
        self.vdf_parameters = kwargs

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
        if self.vdf == '':
            raise ValueError('First you need to set the Volume-Delay Function to use')

        par = list(kwargs.keys())
        if self.vdf.function == 'BPR':
            q = [x for x in par if x not in bpr_parameters] + [x for x in bpr_parameters if x not in par]
        if len(q) > 0:
            raise ValueError('List of functions {} for vdf {} has an inadequate set of parameters'.format(q, self.vdf))
        return True

    def execute(self) -> None:
        self.assignment.execute()
