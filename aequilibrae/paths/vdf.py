from aequilibrae import logger

try:
    from aequilibrae.paths.AoN import bpr, delta_bpr
except ImportError as ie:
    logger.warning(f'Could not import procedures from the binary. {ie.args}')

all_vdf_functions = ['bpr']


class VDF:
    """Volume-Delay function

    ::
        from aequilibrae.paths import VDF

        vdf = VDF()
        vdf.functions_available()
      ['bpr']
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
            else:
                raise ValueError('VDF function not available')
        else:
            raise AttributeError('This class only allows you to set the VDF to use')

    def functions_available(self) -> list:
        """returns a list of all functions available"""
        return all_vdf_functions
