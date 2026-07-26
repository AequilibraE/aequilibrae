import string

from aequilibrae.project.project_table import ProjectTable


class Modes(ProjectTable):
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

    __allowed_characters = string.ascii_letters + "_"

    def __init__(self, net):
        super().__init__(net.project)

    def get_by_name(self, mode_name: str):
        """Get a mode record from the network by its *mode_name*"""
        for mode in self:
            if mode.mode_name == mode_name:
                return mode
        raise ValueError(f"Mode {mode_name} does not exist in the model")

    def _check_mode_id(self, value) -> str:
        if not isinstance(value, str) or len(value) != 1 or value not in string.ascii_letters:
            raise ValueError("Mode IDs must be a single ascii character")
        return value

    def _check_mode_name(self, value) -> str:
        if value is None:
            raise ValueError("mode_name cannot be None")
        for letter in value:
            if letter not in self.__allowed_characters:
                raise ValueError('mode_name can only contain letters and "_"')
        return value
