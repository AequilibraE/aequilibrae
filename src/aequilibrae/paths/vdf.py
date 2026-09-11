from __future__ import annotations

from typing import Any, Callable

import numexpr as ne
import numpy as np

from aequilibrae.paths.cython.vdf_core import (
    akcelik as _akcelik,
)
from aequilibrae.paths.cython.vdf_core import (
    bpr as _bpr,
)
from aequilibrae.paths.cython.vdf_core import (
    bpr2 as _bpr2,
)
from aequilibrae.paths.cython.vdf_core import (
    conical as _conical,
)
from aequilibrae.paths.cython.vdf_core import (
    delta_akcelik as _delta_akcelik,
)
from aequilibrae.paths.cython.vdf_core import (
    delta_bpr as _delta_bpr,
)
from aequilibrae.paths.cython.vdf_core import (
    delta_bpr2 as _delta_bpr2,
)
from aequilibrae.paths.cython.vdf_core import (
    delta_conical as _delta_conical,
)
from aequilibrae.paths.cython.vdf_core import (
    delta_inrets as _delta_inrets,
)
from aequilibrae.paths.cython.vdf_core import (
    inrets as _inrets,
)

FUNCTION_MAP: dict[str, tuple[Callable, Callable]] = {
    "bpr": (_bpr, _delta_bpr),
    "bpr2": (_bpr2, _delta_bpr2),
    "conical": (_conical, _delta_conical),
    "inrets": (_inrets, _delta_inrets),
    "akcelik": (_akcelik, _delta_akcelik),
}

DEFAULT_PRESET_SPECS = {
    "bpr": {
        "alpha": {"fill_NA": 0.15, "bounds": (0.0, float("inf"))},
        "beta": {"fill_NA": 4.0, "bounds": (1.0, float("inf"))},
    },
    "bpr2": {
        "alpha": {"fill_NA": 0.15, "bounds": (0.0, float("inf"))},
        "beta": {"fill_NA": 4.0, "bounds": (1.0, float("inf"))},
    },
    "conical": {
        "alpha": {"fill_NA": 2.0, "bounds": (1.0, float("inf")), "inclusive_lower": False},
        "beta": {"fill_NA": 1.5, "bounds": (1.0, float("inf")), "inclusive_lower": False},
    },
    "inrets": {
        "alpha": {"fill_NA": 1.0, "bounds": (0.0, 1.0)},
    },
    "akcelik": {
        "alpha": {"fill_NA": 0.25, "bounds": (0.0, 1.0)},
        "tau": {"fill_NA": 0.8, "bounds": (0.0, float("inf"))},
        "length": {"bounds": (0, float("inf"))},
    },
}


class VDF:
    """Volume-Delay function that describes the total travel time across a link for a based on its
    free flow travel time and capacity for a given volume of traffic. Supports multiplicative and
    additive delays, and returns the absolute congested time rather than a delay or factor***

    .. code-block:: python

        >>> from aequilibrae.paths import VDF
    """

    def __init__(
        self,
        name: str,
        function: Callable | str,
        spec: dict,
        derivative: Callable | str | None = None,
    ):
        """Creates a Volume Delay Function (VDF) that describes the total travel time across a
        link for a based on its free flow travel time and capacity for a given volume of traffic.

        :Arguments:
            **name** (:obj:`str`): the name of this VDF.

            **function** (:obj:`Callable | str`): the function computing travel time given
            volume, capacity and free flow travel time, as well as the parameters in spec.
            Can be passed directly as a callable, or as a string expression to be interpreted by
            NumExpr.

            **spec** (:obj:`dict`): mapping of parameter names to values used by function (and
            derivative, if applicable) when computing travel time.

            **derivative** (:obj:`Callable | str`, *Optional*): the derivative of function with
            respect to volume. Can be passed directly as a callable, or as a string expression
            that will be converted into a callable via NumExpr. Defaults to None, in which case
            the derivative is instead approximated with a finite difference method.

        :Raises:
            **ValueError**: if function or derivative is passed as a string that cannot be
            converted into a callable.
        """
        if isinstance(function, str):
            function: Callable = self.convert_str_function_into_function(function)

        if isinstance(derivative, str):
            derivative: Callable = self.convert_str_function_into_function(derivative)

        self.name = name
        self.spec = spec

        self.func = function

        if derivative is None:
            self.d_func = self.make_finite_difference_derivative()
        else:
            self.d_func = derivative

    def make_finite_difference_derivative(self, eps: float = 1e-4) -> Callable:
        def finite_diff(delta, link_flows, fftime, capacity, cores, **link_attributes):
            minus_epsilon_congested_time = np.zeros_like(link_flows)
            plus_epsilon_congested_time = np.zeros_like(link_flows)

            self.apply_vdf(
                congested_time=minus_epsilon_congested_time,
                link_flows=link_flows - eps,
                fftime=fftime,
                capacity=capacity,
                cores=cores,
                **link_attributes,
            )
            self.apply_vdf(
                congested_time=plus_epsilon_congested_time,
                link_flows=link_flows + eps,
                fftime=fftime,
                capacity=capacity,
                cores=cores,
                **link_attributes,
            )
            np.subtract(plus_epsilon_congested_time, minus_epsilon_congested_time, out=delta)
            np.divide(delta, 2 * eps, out=delta)

        return finite_diff

    @staticmethod
    def convert_str_function_into_function(function_def: str) -> Callable:
        def func(out: np.ndarray, link_flows, fftime, capacity, cores, **link_attributes):
            # def bpr(congested_times, link_flows, capacity, fftime, cores, alpha, beta):

            ne.evaluate(
                function_def,
                out=out,
                local_dict={"link_flows": link_flows, "fftime": fftime, "capacity": capacity} | link_attributes,
                global_dict={},
                # disable_cache=True,
            )
            # MAKE fftime, capacity into parameters into spec, link flows is in assignment so keep it called "colume"

        return func

    def check_valid(self, num_points, link_attributes: dict[str, Any], from_voc: float = 0.0, to_voc: float = 3.0):
        """Checks if the VDF starts at 1 for 0 volume, is increasing, its derivative is positive, and if it is convex
        via checking that the derivative is increasing. Returns a tuple of bools that are true if it is satisfied, and
        false if these are violated respectively.

        """
        voc_range = np.linspace(from_voc, to_voc, num_points)

        function_values, derivative_values = self._get_fake_vdf_values(voc_range, link_attributes)

        # check monotone increasing - evaluate at some values of flow and check it is increasing
        # also derivative is positive
        # check that f'(x) is strictly increasing -> convex
        decreasing_points = []
        negative_derivative_points = []
        derivative_non_convex_points = []
        for i, voc in enumerate(voc_range):
            if i != 0 and function_values[i] < function_values[i - 1]:
                decreasing_points.append(voc)
            if derivative_values[i] < 0:
                negative_derivative_points.append(voc)
            if i != 0 and derivative_values[i] < derivative_values[i - 1]:
                # non-convex
                derivative_non_convex_points.append(voc)

        # at 0 v/c, it is just the free flow travel time, ie 1
        value_at_0 = function_values[0]

        # should this be a logger instead?
        EPSILON = 1e-4
        vdf_valid_0_value: bool = abs(value_at_0 - 1.0) <= EPSILON
        vdf_increasing_f_vals: bool = not decreasing_points
        vdf_nonnegative_derivative: bool = not negative_derivative_points
        vdf_convex: bool = not derivative_non_convex_points

        if abs(value_at_0 - 1.0) > EPSILON:
            print(f"The value of the VDF for 0 volume was not 1, it was {value_at_0}")
        if decreasing_points:
            print(f"The VDF decreased for these values of volume/capacity: {decreasing_points}")
        if negative_derivative_points:
            print(
                f"The VDF had a negative derivative for these values of volume/capacity: {negative_derivative_points}"
            )
        if derivative_non_convex_points:
            print(
                "The VDF is non-convex due to its derivative decreasing at these values of volume/capacity: "
                f"{derivative_non_convex_points}"
            )
        return vdf_valid_0_value, vdf_increasing_f_vals, vdf_nonnegative_derivative, vdf_convex

    def _get_fake_vdf_values(self, voc_range: np.ndarray, link_attributes: dict[str, Any]):
        size = voc_range.shape[0]
        function_values = np.zeros(size, dtype=np.float64)
        derivative_values = np.zeros(size, dtype=np.float64)
        fftime = np.ones(size, dtype=np.float64)
        capacity = np.ones(size, dtype=np.float64)

        self.apply_vdf(
            congested_time=function_values,
            link_flows=voc_range,
            fftime=fftime,
            capacity=capacity,
            cores=1,
            **link_attributes,
        )
        self.apply_derivative(
            delta=derivative_values,
            link_flows=voc_range,
            fftime=fftime,
            capacity=capacity,
            cores=1,
            **link_attributes,
        )

        return function_values, derivative_values

    def plot_vdf(self, num_points: int, link_attributes: dict[str, Any]):
        """Creates a plot of the vdf using num_points for velocity/capacity in the range [0,3]. Link attributes are
        directly passed self.apply_vdf() and self.apply_derivative, so they may need to be a numpy array the size of
        num_points."""

        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as e:
            e.add_note("VDF plotting requires matplotlib be available in the environment, make sure it is installed")
            raise

        from_voc, to_voc = 0.0, 3.0
        name = self.name
        voc_range = np.linspace(from_voc, to_voc, num_points)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        function_values, derivative_values = self._get_fake_vdf_values(voc_range, link_attributes)

        # Left plot: Function values
        ax1.plot(voc_range, function_values, linewidth=2.5, color="#1f77b4")
        ax1.axvline(x=1.0, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
        ax1.fill_between(voc_range, 0, function_values, alpha=0.1)
        ax1.set_xlabel("Volume / Capacity Ratio", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Travel Time Multiplier (t / t₀)", fontsize=11, fontweight="bold")
        ax1.set_title(f"{name} VDF: Travel Time", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3, linestyle="--")
        ax1.set_xlim(0, 3)
        ax1.text(1.05, ax1.get_ylim()[1] * 0.665, "Capacity", fontsize=9, color="red", rotation=90)

        # Right plot: Derivative (marginal cost)
        ax2.plot(voc_range, derivative_values, linewidth=2.5, color="#ff7f0e")
        ax2.axvline(x=1.0, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
        ax2.fill_between(voc_range, 0, derivative_values, alpha=0.1, color="#ff7f0e")
        ax2.set_xlabel("Volume / Capacity Ratio", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Marginal Travel Time (dt/dv)", fontsize=11, fontweight="bold")
        ax2.set_title(f"{name} VDF: Marginal Cost", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3, linestyle="--")
        ax2.set_xlim(0, 3)
        ax2.text(1.05, ax2.get_ylim()[1] * 0.665, "Capacity", fontsize=9, color="red", rotation=90)

        # Add formula and description
        fig.suptitle(f"{name}")

        plt.tight_layout()

        return fig

    def apply_vdf(self, *, congested_time, link_flows, fftime, capacity, cores: int, **link_attributes):
        self.func(congested_time, link_flows, fftime, capacity, cores, **link_attributes)

    def apply_derivative(self, *, delta, link_flows, fftime, capacity, cores: int, **link_attributes):
        self.d_func(delta, link_flows, fftime, capacity, cores, **link_attributes)


def load_from_parameters(
    vdf_data: dict,
    function_map: dict[str, tuple[Callable, Callable]] | None = None,
) -> dict[str, VDF]:
    """Creates Volume Delay Functions from the given vdf_data dictionary.

    Each entry in vdf_data must specify either a preset function (via "function", referencing
    a name in function_map/FUNCTION_MAP, with an optional "spec" dict overriding the preset's
    default parameters) or a custom functional form (via "functional_form", with an optional
    "derivative_functional_form" and a required "spec" dict). An entry named "default" is
    skipped. If a derivative of a custom VDF is not specified, it will use a finite difference
    scheme to calculate the derivative.

    :Arguments:
        **vdf_data** (:obj:`dict`): Mapping of VDF names to their definitions, which could be
        originally specified in the parameters yaml file. Each entry is a dict containing
        either:
            - "function" (:obj:`str`): the name of a preset VDF function, and optionally
              "spec" (:obj:`dict`) with parameter overrides for that preset; or
            - "functional_form" (:obj:`Callable`): a custom VDF function, optionally paired
              with "derivative_functional_form" (:obj:`Callable`), and a required "spec"
              (:obj:`dict`) of parameters.

        **function_map** (:obj:`dict[str, tuple[Callable, Callable]]`, *Optional*): mapping of
        user supplied function names to (function, derivative) tuples, merged with (and taking
        precedence over) FUNCTION_MAP. Defaults to None, in which case FUNCTION_MAP alone is used.

    :Returns:
        **results** (:obj:`dict[str, VDF]`): mapping of VDF names to their constructed VDF
        objects.

    :Raises:
        **ValueError**: if a "function" entry references an unknown preset vdf, if a "spec"
        contains keys not present in the preset's default spec, or if an entry defines
        neither "function" nor "functional_form".
    """
    results = {}

    function_map = FUNCTION_MAP if function_map is None else FUNCTION_MAP | function_map

    for name, entry in vdf_data.items():
        if name == "default":
            continue

        if "function" in entry:
            function_name = str(entry["function"]).lower()
            if function_name not in function_map:
                raise ValueError(
                    f"VDF '{name}' references unknown preset function '{entry['function']}'. "
                    f"Available presets are: {', '.join(function_map.keys())}."
                )
            func, derivative = function_map[function_name]
            default_spec = DEFAULT_PRESET_SPECS[function_name]

            spec = entry.get("spec")
            if spec is not None:
                if extra := entry["spec"].keys() - default_spec.keys():
                    raise ValueError(f"found unexpected keys in the specification for '{name}': {extra}")

                spec = default_spec | entry["spec"]
            else:
                spec = default_spec

            results[name] = VDF(name, func, spec, derivative)

        elif "functional_form" in entry:
            func = entry["functional_form"]
            derivative = None
            if "derivative_functional_form" in entry:
                derivative = entry["derivative_functional_form"]
            results[name] = VDF(name, func, entry["spec"], derivative)

        else:
            raise ValueError(
                f"VDF '{name}' must define either 'function' (a preset name) or "
                "'functional_form' (a custom expression)."
            )

    return results


# Built-in VDF objects are shared so callers can import the one they need directly.
bpr = VDF("bpr", _bpr, DEFAULT_PRESET_SPECS["bpr"], _delta_bpr)
bpr2 = VDF("bpr2", _bpr2, DEFAULT_PRESET_SPECS["bpr2"], _delta_bpr2)
conical = VDF("conical", _conical, DEFAULT_PRESET_SPECS["conical"], _delta_conical)
inrets = VDF("inrets", _inrets, DEFAULT_PRESET_SPECS["inrets"], _delta_inrets)
akcelik = VDF("akcelik", _akcelik, DEFAULT_PRESET_SPECS["akcelik"], _delta_akcelik)

BPR = bpr
BPR2 = bpr2
CONICAL = conical
INRETS = inrets
AKCELIK = akcelik

_BUILTIN_VDFS = {
    "bpr": bpr,
    "bpr2": bpr2,
    "conical": conical,
    "inrets": inrets,
    "akcelik": akcelik,
}


def builtin_vdfs() -> dict[str, VDF]:
    return _BUILTIN_VDFS.copy()
