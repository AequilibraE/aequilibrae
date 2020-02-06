import numpy as np


class VDF:
    def __init__(self):
        self.__dict__['function'] = ''

    def __setattr__(self, instance, value) -> None:
        if instance not in self.__dict__:
            raise AttributeError('This class does not have "{}" attribute'.format(instance))
        self.__dict__[instance] = value

    def apply_vdf(self, **kwargs):
        if self.function == "BPR":
            return self.BPR(**kwargs)
        else:
            raise ValueError("Sorry, only BPR allowed")

    def apply_derivative(self, vdf: str, **kwargs):
        if self.function == "BPR":
            return self.BPR_delta(**kwargs)
        else:
            raise ValueError("Sorry, only BPR allowed")

    def BPR(self, link_flows: np.array, capacity: np.array, fftime: np.array, alpha: float, beta: float) -> np.array:
        congested_time = fftime * (1 + alpha * np.power(link_flows / capacity, beta))
        return congested_time

    def BPR_delta(self, link_flows: np.array, capacity: np.array, fftime: np.array, alpha: float,
                  beta: float) -> np.array:
        dbpr = fftime * (alpha * beta * np.power(link_flows / capacity, beta - 1)) / capacity
        return dbpr
