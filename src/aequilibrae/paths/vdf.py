from __future__ import annotations
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

import numpy as np
import numexpr as ne


FUNCTION_MAP: dict[str, tuple[Callable, Callable]] = {
    "bpr": (bpr, delta_bpr),
    "bpr2": (bpr2, delta_bpr2),
    "conical": (conical, delta_conical),
    "inrets": (inrets, delta_inrets),
    "akcelik": (akcelik, delta_akcelik),
}

DEFAULT_PRESET_SPECS = {
    "bpr": {
        "alpha": {"fill_NA": 0.15, "bounds": (0.0, float("inf"))},
        "beta": {"fill_NA": 4.0, "bounds": (1.0, float("inf"))},
        # "fftime": {"bounds": (0, float("inf"))},
        "capacity": {"bounds": (0, float("inf"))},
    },
    "bpr2": {
        "alpha": {"fill_NA": 0.15, "bounds": (0.0, float("inf"))},
        "beta": {"fill_NA": 4.0, "bounds": (1.0, float("inf"))},
        # "fftime": {"bounds": (0, float("inf"))},
        "capacity": {"bounds": (0, float("inf"))},
    },
    "conical": {
        "alpha": {"fill_NA": 1.0, "bounds": (1.0, float("inf"))},
        # "fftime": {"bounds": (0, float("inf"))},
        "capacity": {"bounds": (0, float("inf"))},
    },
    "inrets": {
        "alpha": {"fill_NA": 1.0, "bounds": (0.0, 1.0)},
        # "fftime": {"bounds": (0, float("inf"))},
        "capacity": {"bounds": (0, float("inf"))},
    },
    "akcelik": {
        "alpha": {"fill_NA": 0.25, "bounds": (0.0, 1.0)},
        "tau": {"fill_NA": 0.8, "bounds": (0.0, float("inf"))},
        "length": {"bounds": (0, float("inf"))},
        # "fftime": {"bounds": (0, float("inf"))},
        "capacity": {"bounds": (0, float("inf"))},
    },
}


class VDFsManager:
    def __init__(self, add_preset_vdfs: bool = False, vdf_data_from_parameters: dict | None = None):
        self.vdfs: dict[str, VDF] = {}
        if add_preset_vdfs:
            for name in FUNCTION_MAP.keys():
                self.add_preset_vdf(name)
        if vdf_data_from_parameters is not None:
            self._load_from_parameters(vdf_data_from_parameters)

    def _load_from_parameters(self, vdf_data: dict):
        """Populate self.vdfs from the parsed "vdfs" section of parameters.yml, e.g.:

        vdfs:
            default: "bpr_tyler"
            bpr_tyler:
                function: bpr
                spec:
                    alpha:
                        fillNA: 0.15
                        bounds: [0, 10]
                    beta: 4
            quadratic:
                functional_form: "fftime * (a * (link_flows/capacity)**2 + b * (link_flows/capacity) + 1)"
                derivative_functional_form: "fftime * (2 * a * (link_flows/capacity) + b) / capacity"
                spec:
                    a:
                        fillNA: 0.15
                        bounds: [0, .inf]
                    b:
                        fillNA: 1.0
                        bounds: [0, .inf]
        """
        vdf_data = dict(vdf_data)  # don't mutate the caller's dict
        self.default = vdf_data.pop("default", None)

        for name, entry in vdf_data.items():
            if entry is None:
                continue

            if "function" in entry:
                function_name = str(entry["function"]).lower()
                if function_name not in FUNCTION_MAP:
                    raise ValueError(
                        f"VDF '{name}' references unknown preset function '{entry['function']}'. "
                        f"Available presets are: {', '.join(FUNCTION_MAP.keys())}."
                    )
                func, derivative = FUNCTION_MAP[function_name]
                self.add_vdf(name, func, entry["spec"], derivative, override_existing=True)

            elif "functional_form" in entry:
                func = entry["functional_form"]
                derivative = None
                if "derivative_functional_form" in entry:
                    derivative = entry["derivative_functional_form"]
                self.add_vdf(name, func, entry["spec"], derivative, override_existing=True)

            else:
                raise ValueError(
                    f"VDF '{name}' must define either 'function' (a preset name) or "
                    "'functional_form' (a custom expression)."
                )

    @staticmethod
    def convert_str_function_into_function(function_def: str) -> Callable:
        def func(out: np.ndarray, link_flows, fftime, cores, **link_attributes):
            # def bpr(congested_times, link_flows, capacity, fftime, cores, alpha, beta):

            ne.evaluate(
                function_def,
                out=out,
                local_dict={"link_flows": link_flows, "fftime": fftime} | link_attributes,
                global_dict={},
                # disable_cache=True,
            )
            # MAKE fftime, capacity into parameters into spec, link flows is in assignment so keep it called "colume"

        return func

    def add_vdf(
        self,
        name: str,
        function: Callable | str,
        spec: dict,
        derivative: Callable | str | None,
        override_existing: bool = False,
    ):
        # bpr_spec = {"alpha": {"fill_NA": 0.15}, "beta": {"fill_NA": 4.0}}
        # project.add_vdf(name="bpr_tyler", function="bpr", spec=bpr_spec)
        # have wrapper in project that passes all arguments here
        if isinstance(function, str):
            function: Callable = VDFsManager.convert_str_function_into_function(function)
        if isinstance(derivative, str):
            derivative: Callable = VDFsManager.convert_str_function_into_function(derivative)
        if not override_existing and name in self.vdfs:
            raise ValueError(f"A volume delay function of name {name} is already stored.")
        new_vdf = VDF(name, function, spec, derivative)
        self.vdfs[name] = new_vdf

    def get_vdf(self, name) -> VDF:
        name_lower = name.lower()
        if name_lower in FUNCTION_MAP:
            name = name_lower

        if name not in self.vdfs:
            raise ValueError(f"VDF of name {name} is not stored.\nAvailable are {', '.join(self.vdfs.keys())}")
        return self.vdfs[name]

    def add_preset_vdf(self, name: str, custom_name: str = "", spec: dict | None = None):
        name_lower = name.lower()
        if name_lower not in FUNCTION_MAP:
            raise ValueError(f"A preset volume delay function of name {name_lower} does not exist.")
        if spec is None:
            spec = DEFAULT_PRESET_SPECS[name_lower]
        func, derivative = FUNCTION_MAP[name_lower]
        self.add_vdf(custom_name if custom_name else name, func, spec, derivative)

    @staticmethod
    def make_preset_vdf(name: str, custom_name: str = "", spec: dict | None = None):
        name_lower = name.lower()
        if name_lower not in FUNCTION_MAP:
            raise ValueError(f"A preset volume delay function of name {name_lower} does not exist.")
        if spec is None:
            spec = DEFAULT_PRESET_SPECS[name_lower]
        func, derivative = FUNCTION_MAP[name_lower]
        return VDF(custom_name if custom_name else name, func, spec, d_func=derivative)

    def comparison_plots(self):
        #
        pass


class VDF:
    """Volume-Delay function
    spec = {
        "graph_column_a": {"fill_NA": 5, "bounds": (0, 10)},
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

    def apply_vdf(self, congested_time, link_flows, fftime, cores: int, **link_attributes):
        self.func(congested_time, link_flows, fftime, cores, **link_attributes)

    def apply_derivative(self, congested_time, link_flows, fftime, cores: int, **link_attributes):
        self.d_func(congested_time, link_flows, fftime, cores, **link_attributes)


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
            value = value.lower()
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
