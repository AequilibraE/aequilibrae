import os
import yaml


class Parameters:
    """
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
        self.path = os.path.dirname(os.path.realpath(__file__))

        file = os.path.join(self.path, "parameters.yml")
        with open(file, "r") as yml:
            self.parameters = yaml.load(yml, Loader=yaml.SafeLoader)

    def write_back(self):
        stream = open(self.path + "/parameters.yaml", "w")
        yaml.dump(self.parameters, stream, default_flow_style=False)
        stream.close()
