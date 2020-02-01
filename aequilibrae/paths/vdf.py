import numpy as np


class vdf:
    def __init__(self):
        pass

    def apply_vdf(self, vdf: str, **kwargs):
        if vdf == "BPR":
            return self.BPR(**kwargs)
        else:
            raise ValueError("Sorry, only BPR allowed")

    def apply_derivative(self, vdf: str, **kwargs):
        if vdf == "BPR":
            return self.BPR_delta(**kwargs)
        else:
            raise ValueError("Sorry, only BPR allowed")

    def BPR(self, link_flows: np.array, capacity: np.array, fftime: np.array) -> np.array:
        alpha = 0.15
        beta = 4
        congested_time = fftime * (1 + alpha * np.power(link_flows / capacity, beta))
        return congested_time

    def BPR_delta(self, link_flows: np.array, capacity: np.array, fftime: np.array) -> np.array:
        alpha = 0.15
        beta = 4
        dbpr = fftime * (alpha * beta * np.power(link_flows / capacity, beta - 1)) / capacity
        return dbpr
