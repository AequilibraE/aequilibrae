import logging
from copy import deepcopy
from os import PathLike
from pathlib import Path
from typing import Callable, Optional

import yaml

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

        >>> from aequilibrae.parameters import Parameters

        >>> project = Project()
        >>> project.new(project_path)

        >>> p = Parameters()

        >>> p.parameters['system']['logging_directory'] =  "/path_to/other_logging_directory"
        >>> p.parameters['osm']['overpass_endpoint'] = "http://192.168.0.110:32780/api"
        >>> p.parameters['osm']['timeout'] = 180
        >>> p.write_back()

        >>> # You can also restore the software default values
        >>> p.restore_default()

        >>> project.close()
    """

    file_default: Path = Path(__file__).parent / "parameters.yml"
    _default: dict

    def __init__(self, path: Optional[Path] = None):
        """Loads parameters from file."""
        self.file: PathLike | str = None
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
            logger.debug("No pre-existing parameter file exists for this project. Will use default")

            self.parameters = deepcopy(self._default)

    def write_back(self):
        """Writes the parameters back to file"""
        with open(self.file, "w") as stream:
            yaml.dump(self.parameters, stream, default_flow_style=False)

    def restore_default(self):
        """Restores parameters to generic default"""
        self.parameters = deepcopy(self._default)
        self.write_back()

    def get_vdfs(
        self,
        exclude_builtins: bool = False,
        function_map: dict[str, tuple[Callable, Callable]] | None = None,
    ):
        """Gets Volume Delay Functions (VDFs) specified in the parameters "vdfs" entry, as well
        as preset VDFs if exclude_builtins is False.

        Each entry in the "vdfs" entry of the parameters must either be specifying a default vdf, or
        either a preset function via "function" or a custom functional form "functional_form".
        If "function" is specified, the name of a preset VDF or VDF in function_map must be specified,
        with an optional "spec" dict overriding the preset's default parameters. If "functional_form"
        is specified with a string representation of the VDF (to be interpreted by NumExpr), the
        specification is required in "spec" and its derivative can be optionally included by its
        string representation in "derivative_functional_form". If a derivative of a custom VDF is not
        specified, it will use a finite difference scheme to calculate the derivative.

        :Arguments:
            **exclude_builtins** (:obj:`bool`, *Optional*): Setting to exclude the built in preset
            VDFs, for example bpr. Defaults to False, so presets are included
            **function_map** (:obj:`dict[str, tuple[Callable, Callable]]`, *Optional*): mapping of
            user supplied function names to (function, derivative) tuples, taking precedence over
            preset vdf definitions. Then the parameters can reference these functions
            by their name and spec in the "function" entry.

        :Returns:
            **results** (:obj:`dict[str, VDF]`): mapping of VDF names to their constructed VDF
            objects.

        :Raises:
            **ValueError**: if a "function" entry references a preset not found in function_map
            or a built in preset, if a "spec" contains keys not present in the preset's default
            spec, or if an entry defines neither "function" nor "functional_form".
        """
        from aequilibrae.paths.vdf import builtin_vdfs, load_from_parameters

        vdfs = self.parameters.get("vdfs", None)
        if vdfs is None:
            raise ValueError("no 'vdfs' entry in parameters file")

        vdfs = load_from_parameters({k: v for k, v in vdfs.items() if k != "default"}, function_map=function_map)

        if exclude_builtins:
            return vdfs

        builtins = builtin_vdfs()
        if conflicts := builtins.keys() & vdfs.keys():
            raise ValueError(f"cannot name VDF in parameters the same as a built in VDF, found conflicts {conflicts}")

        return builtins | vdfs

    @classmethod
    def load_default(cls):
        with open(cls.file_default, "r") as yml:
            return yaml.safe_load(yml)


Parameters._default = Parameters.load_default()
