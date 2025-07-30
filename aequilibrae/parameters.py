import os
import yaml
from copy import deepcopy
import logging
import pathlib
from typing import Optional
from aequilibrae.context import get_active_project


class Parameters:
    """Global parameters module.

    Parameters are used in many procedures, and are often defined in the ``parameters.yml`` file ONLY.

    Parameters are organized in the following groups:

    * assignment
    * distribution
    * network
      * links
      * modes
      * nodes
      * osm
      * gmns
    * osm
    * system

    Please observe that OSM information handled on network is not the same on the OSM group.

    .. code-block:: python

        >>> from aequilibrae import Parameters

        >>> project = Project()
        >>> project.new(project_path)

        >>> p = Parameters()

        >>> p.parameters['system']['logging_directory'] =  "/path_to/other_logging_directory"
        >>> p.parameters['osm']['overpass_endpoint'] = "http://192.168.0.110:32780/api"
        >>> p.parameters['osm']['max_query_area_size'] = 10000000000
        >>> p.parameters['osm']['sleeptime'] = 0
        >>> p.write_back()

        >>> # You can also restore the software default values
        >>> p.restore_default()

        >>> project.close()
    """

    _default: dict
    file_default: str

    def __init__(self, path: Optional[pathlib.Path] = None):
        """Loads parameters from file."""
        self.file = None
        if path is not None:
            self.file = path / "parameters.yml"
        else:
            proj = get_active_project(must_exist=False)
            if proj is not None:
                self.file = proj.project_base_path / "parameters.yml"

        if self.file is not None and self.file.is_file():
            with open(self.file, "r") as yml:
                self.parameters = yaml.load(yml, Loader=yaml.SafeLoader)
        else:
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
