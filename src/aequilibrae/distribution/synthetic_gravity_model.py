import pandas as pd
import yaml

valid_functions = ["EXPO", "GAMMA", "POWER"]
members = ["function", "alpha", "beta", "districts", "k_factors"]
model_type = "SyntheticGravityModel"


class SyntheticGravityModel:
    """Simple class object to represent synthetic gravity models

    K-factors are a ``pd.DataFrame`` with origin districts as index and destination districts as columns, and
    ``districts`` a ``pd.Series`` with the district of each zone, indexed by zone ID.
    """

    def __init__(self):
        self.function = None
        self.alpha = None
        self.beta = None
        self.districts = None
        self.k_factors = None

    def __setattr__(self, key, value):
        if (value is None and key in members) or key in ["districts", "k_factors"]:
            self.__dict__[key] = value
        else:
            if key == "function":
                self.alpha = None
                self.beta = None
                self.districts = None
                self.k_factors = None
                if value not in self.valid_functions:
                    raise ValueError("Function needs to be one of these: " + ", ".join(self.valid_functions))
            else:
                if isinstance(value, float) or isinstance(value, int):
                    if key == "alpha":
                        if self.__dict__.get("function") == "EXPO":
                            raise ValueError("Exponential function does not have an alpha parameter")

                    if key == "beta":
                        if self.function == "POWER":
                            raise ValueError("Inverse power function does not have a beta parameter")
                else:
                    raise ValueError("Parameter needs to be numeric")

            self.__dict__[key] = value

    def __getattr__(self, key):
        if key == "valid_functions":
            return valid_functions
        elif key == "members":
            return members
        elif key == "model_type":
            return model_type
        else:
            return self.__dict__[key]

    def load(self, file_name):
        """Loads model from disk. Extension is \\*.mod"""
        try:
            with open(file_name, "r") as f:
                model = yaml.safe_load(f)[self.model_type]
                for key, value in model.items():
                    if key in self.members:
                        if key == "districts":
                            value = pd.Series(value)
                        elif key == "k_factors":
                            value = pd.DataFrame.from_dict(value, orient="index")
                        self.__dict__[key] = value
                    else:
                        raise ValueError("Model has unknown parameters: " + str(key))
        except ValueError as err:
            raise ValueError("File provided is not a valid Synthetic Gravity Model - {}".format(err.__str__())) from err

    def save(self, file_name):
        """Saves model to disk in yaml format. Extension is \\*.mod"""
        file_name = str(file_name)
        if file_name[-4:].upper() != ".MOD":
            file_name += ".mod"

        model = {model_type: {"function": self.function, "alpha": self.alpha, "beta": self.beta}}
        if self.k_factors is not None:
            model[model_type]["districts"] = self.districts.to_dict()
            model[model_type]["k_factors"] = self.k_factors.to_dict(orient="index")

        mod = open(file_name, "w")
        yaml.dump(model, mod, default_flow_style=False)
        mod.flush()
        mod.close()
