from sqlite3 import IntegrityError, Connection
from aequilibrae.project.network.mode import Mode
from aequilibrae.project.field_editor import FieldEditor


class Modes:
    """
    Access to the API resources to manipulate the modes table in the network

    .. code-block:: python

        >>> from aequilibrae import Project

        >>> p = Project.from_path("/tmp/test_project")

        >>> modes = p.network.modes

        # We can get a dictionary of all modes in the model
        >>> all_modes = modes.all_modes()

        # And do a bulk change and save it
        >>> for mode_id, mode_obj in all_modes.items():
        ...     mode_obj.beta = 1
        ...     mode_obj.save()

        # or just get one mode in specific
        >>> car_mode = modes.get('c')

        # or just get this same mode by name
        >>> car_mode = modes.get_by_name('car')

        # We can change the description of the mode
        >>> car_mode.description = 'personal autos only'

        # Let's say we are using alpha to store the PCE for a future year with much smaller cars
        >>> car_mode.alpha = 0.95

        # To save this mode we can simply
        >>> car_mode.save()

        # We can also create a completely new mode and add to the model
        >>> new_mode = modes.new('k')
        >>> new_mode.mode_name = 'flying_car'  # Only ASCII letters and *_* allowed # other fields are not mandatory

        # We then explicitly add it to the network
        >>> modes.add(new_mode)

        # we can even keep editing and save it directly once we have added it to the project
        >>> new_mode.description = 'this is my new description'
        >>> new_mode.save()
    """

    def __init__(self, net):
        self.__all_modes = []
        self.__items = {}
        self.project = net.project
        self.logger = net.logger
        self.conn = net.conn  # type: Connection
        self.curr = net.conn.cursor()
        self.__update_list_of_modes()

    def add(self, mode: Mode) -> None:
        """We add a mode to the project"""
        self.__update_list_of_modes()
        if mode.mode_id in self.__all_modes:
            raise ValueError("Mode already exists in the model")

        self.curr.execute("insert into 'modes'(mode_id, mode_name) Values(?,?)", [mode.mode_id, mode.mode_name])
        self.conn.commit()
        self.logger.info(f"mode {mode.mode_name}({mode.mode_id}) was added to the project")
        mode.save()
        self.__update_list_of_modes()

    def delete(self, mode_id: str) -> None:
        """Removes the mode with *mode_id* from the project"""
        try:
            self.curr.execute(f'delete from modes where mode_id="{mode_id}"')
            self.conn.commit()
        except IntegrityError as e:
            self.logger.error(f"Failed to remove mode {mode_id}. {e.args}")
            raise e
        self.logger.warning(f"Mode {mode_id} was successfully removed from the database")
        self.__update_list_of_modes()

    @property
    def fields(self) -> FieldEditor:
        """Returns a FieldEditor class instance to edit the Modes table fields and their metadata"""
        return FieldEditor(self.project, "modes")

    def get(self, mode_id: str) -> Mode:
        """Get a mode from the network by its *mode_id*"""
        self.__update_list_of_modes()
        if mode_id not in self.__all_modes:
            raise ValueError(f"Mode {mode_id} does not exist in the model")
        return Mode(mode_id, self.project)

    def get_by_name(self, mode: str) -> Mode:
        """Get a mode from the network by its *mode_name*"""
        self.__update_list_of_modes()
        self.curr.execute(f"select mode_id from 'modes' where mode_name='{mode}'")
        found = self.curr.fetchone()
        if len(found) == 0:
            raise ValueError(f"Mode {mode} does not exist in the model")
        return Mode(found[0], self.project)

    def all_modes(self) -> dict:
        """Returns a dictionary with all mode objects available in the model. mode_id as key"""
        self.__update_list_of_modes()
        return {x: Mode(x, self.project) for x in self.__all_modes}

    def new(self, mode_id: str) -> Mode:
        """Returns a new mode with *mode_id* that can be added to the model later"""
        if mode_id in self.__all_modes:
            raise ValueError("Mode already exists in the model. Creating a new one does not make sense")

        return Mode(mode_id, self.project)

    def __update_list_of_modes(self) -> None:
        self.curr.execute("select mode_id from 'modes'")
        self.__all_modes = [x[0] for x in self.curr.fetchall()]

    def __copy__(self):
        raise Exception("Modes object cannot be copied")

    def __deepcopy__(self, memodict=None):
        raise Exception("Modes object cannot be copied")

    def __del__(self):
        self.__items.clear()

    def __has_mode(self):
        curr = self.conn.cursor()
        curr.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return any(["modes" in x[0] for x in curr.fetchall()])
