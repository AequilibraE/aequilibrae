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

# Maps the VDF name to the pair of kernels that evaluate the curve and its derivative
VDF_KERNELS = {
    "BPR": (bpr, delta_bpr),
    "BPR2": (bpr2, delta_bpr2),
    "CONICAL": (conical, delta_conical),
    "INRETS": (inrets, delta_inrets),
    "AKCELIK": (akcelik, delta_akcelik),
}

all_vdf_functions = [name.lower() for name in VDF_KERNELS]


class VDF:
    """Volume-Delay function

    .. code-block:: python

        >>> from aequilibrae.paths import VDF

        >>> vdf = VDF()
        >>> vdf.functions_available()
        ['bpr', 'bpr2', 'conical', 'inrets', 'akcelik']

    """

    def __init__(self):
        self.__dict__["function"] = ""
        self.__dict__["apply_vdf"] = None
        self.__dict__["apply_derivative"] = None

    def __setattr__(self, instance, value) -> None:
        if instance != "function":
            raise AttributeError("This class only allows you to set the VDF to use")

        value = value.upper()
        self.__dict__[instance] = value
        if value not in VDF_KERNELS:
            raise ValueError("VDF function not available")
        self.__dict__["apply_vdf"], self.__dict__["apply_derivative"] = VDF_KERNELS[value]

    def functions_available(self) -> list:
        """returns a list of all functions available"""
        return all_vdf_functions
