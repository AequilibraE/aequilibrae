import os
import yaml
from copy import deepcopy
import logging
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

    _default: dict
    file_default: str

    def __init__(self, project=None):
        """Loads parameters from file. The place is always the same. The root of the package"""
        project = project or get_active_project(must_exist=False)
        proj_path = project.project_base_path if project is not None else ""

        self.file = os.path.join(proj_path, "parameters.yml")

        if os.path.isfile(self.file):
            with open(self.file, "r") as yml:
                self.parameters = yaml.load(yml, Loader=yaml.SafeLoader)
        else:
            if project is not None:
                logger = logging.getLogger("aequilibrae")
                logger.warning("No pre-existing parameter file exists for this project. Will use default")

            self.parameters = deepcopy(self._default)

    def write_back(self):
        """Writes the parameters back to file"""
        with open(self.file, "w") as stream:
            yaml.dump(self.parameters, stream, default_flow_style=False)

    def restore_default(self):
        """Restores parameters to generic default"""
        self.parameters = self._default
        self.write_back()


Parameters.file_default = os.path.join(os.path.dirname(os.path.realpath(__file__)), "parameters.yml")
with open(Parameters.file_default, "r") as yml:
    Parameters._default = yaml.safe_load(yml)
