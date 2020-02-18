from aequilibrae.paths.AoN import bpr, delta_bpr

all_vdf_functions = ['bpr']


class VDF:
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

    def functions_available(self):
        return all_vdf_functions
