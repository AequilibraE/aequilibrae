from typing import Callable
from aequilibrae.paths.cython.AoN import (
    bpr,
    delta_bpr,
    bpr2,
    delta_bpr2,
    conical,
    delta_conical,
    inrets,
    delta_inrets,
    akcelik,
    delta_akcelik,
)


all_vdf_functions = ["bpr", "bpr2", "conical", "inrets", "akcelik"]


class VDFsManager:
    def __init__(self):
        # look in

        pass

    def add_vdf(self):
        # project.add_vdf(name="bpr_tyler", function="bpr", spec=bpr_spec)
        # have wrapper in project that passes all arguments here
        pass

    def comparison_plots(self):
        #
        pass


class VDF:
    """Volume-Delay function
    spec = {
        "graph_column_a": {"default": 5, "bounds": (0, 10)},
        "graph_column_b": {"bounds": (0, float("inf")},
    }

    ***SUPPORTS multiplicative and additive delays, needs to return the absolute congested time rather than a
    delay or factor***

    .. code-block:: python

        >>> from aequilibrae.paths import VDF



    """

    def __init__(self, name: str, func: Callable, spec: dict, d_func: Callable | None = None):
        self.name = name
        self.func = func
        self.spec = spec
        self.d_func = d_func

        self.check_valid()

    def check_valid(self):
        # check vdf is non-negative

        # check monotone increasing - evaluate at some values of flow and check it is increasing
        # also derivative is positive

        # check parameters are within the specified bounds
        pass

    def plot_vdf(self):
        # put code from documentation of vdfs
        pass

    def apply_vdf(self, congested_time, link_flows, capacity, fftime, cores: int, **link_attributes):
        self.func(congested_time, link_flows, capacity, fftime, cores, **link_attributes)

    def apply_derivative(self, congested_time, link_flows, capacity, fftime, cores: int, **link_attributes):
        self.func(congested_time, link_flows, capacity, fftime, cores, **link_attributes)


class VDF_old:
    """Volume-Delay function old

    .. code-block:: python

        >>> from aequilibrae.paths import VDF

        >>> vdf = VDF_old()
        >>> vdf.functions_available()
        ['bpr', 'bpr2', 'conical', 'inrets', 'akcelik']

    """

    def __init__(self):
        self.__dict__["function"] = ""
        self.__dict__["apply_vdf"] = None
        self.__dict__["apply_derivative"] = None

    def __setattr__(self, instance, value) -> None:
        if instance == "function":
            value = value.upper()
            self.__dict__[instance] = value
            if value == "BPR":
                self.__dict__["apply_vdf"] = bpr
                self.__dict__["apply_derivative"] = delta_bpr
            elif value == "BPR2":
                self.__dict__["apply_vdf"] = bpr2
                self.__dict__["apply_derivative"] = delta_bpr2
            elif value == "CONICAL":
                self.__dict__["apply_vdf"] = conical
                self.__dict__["apply_derivative"] = delta_conical
            elif value == "INRETS":
                self.__dict__["apply_vdf"] = inrets
                self.__dict__["apply_derivative"] = delta_inrets
            elif value == "AKCELIK":
                self.__dict__["apply_vdf"] = akcelik
                self.__dict__["apply_derivative"] = delta_akcelik
            else:
                raise ValueError("VDF function not available")
        else:
            raise AttributeError("This class only allows you to set the VDF to use")

    def functions_available(self) -> list:
        """returns a list of all functions available"""
        return all_vdf_functions
