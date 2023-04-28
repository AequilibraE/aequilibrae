import os
import yaml
from copy import deepcopy
import logging
from aequilibrae.context import get_active_project


class Parameters:
    """Global parameters module

    Parameters are used in many procedures, and are often defined only in the parameters.yml file ONLY
    Parameters are organized in the following groups:

    * assignment
    * distribution
    * system
    * report zeros
    * temp directory

    .. code-block:: python

        >>> from aequilibrae import Project, Parameters

        >>> project = Project.from_path("/tmp/test_project")

        >>> p = Parameters(project)

        >>> p.parameters['system']['logging_directory'] =  "/tmp/other_folder"
        >>> p.parameters['osm']['overpass_endpoint'] = "http://192.168.0.110:32780/api"
        >>> p.parameters['osm']['max_query_area_size'] = 10000000000
        >>> p.parameters['osm']['sleeptime'] = 0
        >>> p.write_back()

        >>> # You can also restore the software default values
        >>> p.restore_default()
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
