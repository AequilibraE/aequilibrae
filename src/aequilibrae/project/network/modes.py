from typing import Any

from aequilibrae.project.project_table import NonSpatialProjectTable


class Modes(NonSpatialProjectTable):
    """
    Access to the API resources to manipulate the modes table in the network

    .. code-block:: python

        >>> project = create_example(project_path)

        >>> modes = project.network.modes

        # We can get a mode as an immutable record
        >>> car_mode = modes.get('c')

        # or by name
        >>> car_mode = modes.get_by_name('car')

        # and write changes explicitly
        >>> modes.update('c', description='personal autos only', alpha=0.95)

        # Adding a new mode to the model is a single insert
        >>> modes.insert(mode_id='k', mode_name='flying_car')
        'k'

        >>> project.close()
    """

    name = "modes"
    key = "mode_id"
    record_name = "ModeRecord"

    def get_by_name(self, mode_name: str) -> Any:
        """Get a mode record by its descriptive name.

        :Arguments:
            **mode_name** (:obj:`str`): Descriptive mode name.

        :Returns:
            **mode** (:obj:`Any`): Generated frozen record for the mode.
        """
        for mode in self:
            if mode.mode_name == mode_name:
                return mode
        raise ValueError(f"Mode {mode_name} does not exist in the model")
