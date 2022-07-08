import os
import yaml
from warnings import warn
from copy import deepcopy
from aequilibrae.context import get_active_project


class Parameters:
    """
    Global parameters module

    Parameters are used in many procedures, and are often defined only in the parameters.yml file ONLY
    Parameters are organized in the following groups:

    * assignment
    * distribution

    * system
        - cpus: Maximum threads to be used in any procedure

        - default_directory: If is the directory QGIS file opening/saving dialogs will try to open as standard

        - driving side: For purposes of plotting on QGIS

        - logging: Level of logging to be written to temp/aequilibrae.log: Levels are those from the Python logging library
                    - 0: 'NOTSET'
                    - 10: 'DEBUG'
                    - 20: 'INFO'
                    - 30: 'WARNING'
                    - 40: 'ERROR'
                    - 50: 'CRITICAL'
    * report zeros
    * temp directory
    """

    def __init__(self, proj_path=None):
        """Loads parameters from file. The place is always the same. The root of the package"""
        if not proj_path:
            proj = get_active_project(must_exist=False)
            proj_path = proj.project_base_path if proj is not None else ""

        default_path = os.path.dirname(os.path.realpath(__file__))

        self.file = os.path.join(proj_path, "parameters.yml")
        self.file_default = os.path.join(default_path, "parameters.yml")

        with open(self.file_default, "r") as yml:
            self._default = yaml.load(yml, Loader=yaml.SafeLoader)

        if os.path.isfile(self.file):
            with open(self.file, "r") as yml:
                self.parameters = yaml.load(yml, Loader=yaml.SafeLoader)
        else:
            self.parameters = deepcopy(self._default)
            if proj_path:
                warn("No pre-existing parameter file exists for this project. Will use default")

    def write_back(self):
        """Writes the parameters back to file"""
        with open(self.file, "w") as stream:
            yaml.dump(self.parameters, stream, default_flow_style=False)

    def restore_default(self):
        """Restores parameters to generic default"""
        self.parameters = self._default
        self.write_back()
