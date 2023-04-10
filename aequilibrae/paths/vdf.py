from aequilibrae import global_logger

try:
    from aequilibrae.paths.AoN import bpr, delta_bpr, bpr2, delta_bpr2, conical, delta_conical, inrets, delta_inrets
except ImportError as ie:
    global_logger.warning(f"Could not import procedures from the binary. {ie.args}")

all_vdf_functions = ["bpr", "bpr2", "conical", "inrets"]


class VDF:
    """Volume-Delay function

    .. code-block:: python

        >>> from aequilibrae.paths import VDF

        >>> vdf = VDF()
        >>> vdf.functions_available()
        ['bpr', 'bpr2', 'conical', 'inrets']

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
            else:
                raise ValueError("VDF function not available")
        else:
            raise AttributeError("This class only allows you to set the VDF to use")

    def functions_available(self) -> list:
        """returns a list of all functions available"""
        return all_vdf_functions
