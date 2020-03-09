import os
import yaml


class Parameters:
    """
    Global parameters module

    Parameters are used in many procedures, and are often defined only in thi parameters.yml file ONLY
    Parameters are organized in the following groups:

    assignment:
    distribution:

    system:
    cpus: Maximum threads to be used in any procedure
    default_directory: If is the directory QGIS file opening/saving dialogs will try to open as standard
    driving side: For purposes of plotting on QGIS
    logging: Level of logging to be written to temp/aequilibrae.log: Levels are those from the Python logging library
                0: 'NOTSET'
                10: 'DEBUG'
                20: 'INFO'
                30: 'WARNING'
                40: 'ERROR'
                50: 'CRITICAL'
            both numeric and text accepted
    report zeros:
    temp directory:

    """

    def __init__(self):
        """ Loads parameters from file. The place is always the same. The root of the package"""
        path = os.path.dirname(os.path.realpath(__file__))
        self.file = os.path.join(path, "parameters.yml")
        with open(self.file, "r") as yml:
            self.parameters = yaml.load(yml, Loader=yaml.SafeLoader)

    def write_back(self):
        """Writes the parameters back to file"""
        stream = open(self.file, "w")
        yaml.dump(self.parameters, stream, default_flow_style=False)
        stream.close()

    def restore_default(self):
        """Restores parameters to generic default"""
        path = os.path.dirname(os.path.realpath(__file__))
        default_file = os.path.join(path, "parameter_default.yml")
        with open(default_file, "r") as yml:
            self.parameters = yaml.load(yml, Loader=yaml.SafeLoader)
        self.write_back()
