import yaml

valid_functions = ["EXPO", "GAMMA", "POWER"]
members = ["function", "alpha", "beta"]
model_type = "SyntheticGravityModel"


class SyntheticGravityModel:
    """Simple class object to represent synthetic gravity models"""

    def __init__(self):
        self.function = None
        self.alpha = None
        self.beta = None

    def __setattr__(self, key, value):
        if value is None and key in ["function", "alpha", "beta"]:
            self.__dict__[key] = value
        else:
            if key == "function":
                self.alpha = None
                self.beta = None
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
        R"""Loads model from disk. Extension is \*.mod"""
        try:
            with open(file_name, "r") as f:
                model = yaml.safe_load(f)[self.model_type]
                for key, value in model.items():
                    if key in self.members:
                        self.__dict__[key] = value
                    else:
                        raise ValueError("Model has unknown parameters: " + str(key))
        except ValueError as err:
            raise ValueError("File provided is not a valid Synthetic Gravity Model - {}".format(err.__str__()))

    def save(self, file_name):
        R"""Saves model to disk in yaml format. Extension is \*.mod"""
        file_name = str(file_name)
        if file_name[-4:].upper() != ".MOD":
            file_name += ".mod"

        model = {model_type: {"function": self.function, "alpha": self.alpha, "beta": self.beta}}

        mod = open(file_name, "w")
        yaml.dump(model, mod, default_flow_style=False)
        mod.flush()
        mod.close()
