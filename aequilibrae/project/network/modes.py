from sqlite3 import IntegrityError, Connection
from aequilibrae.project.network.mode import Mode
from aequilibrae import logger


class Modes:
    """
    Access to the API resources to manipulate the modes table in the network

    ::

        from aequilibrae import Project
        from aequilibrae.project.network import Mode

        p = Project()
        p.open('path/to/project/folder')

        modes = p.network.modes

        # We can get a dictionary of all modes in the model
        all_modes = modes.all_modes()

        #And do a bulk change and save it
        for mode_id to mode_obj in all_modes.items():
            mode_obj.beta = 1
            mode_obj.save()

        # or just get one mode in specific
        car_mode = modes.get('c')

        # We can change the description of the mode
        car_mode.description = 'personal autos only'

        # Let's say we are using alpha to store the PCE for a future year with much smaller cars
        car_mode.alpha = 0.95

        # To save this mode we can simply
        car_mode.save()

        # We can also create a completely new mode and add to the model
        new_mode = Mode('k')
        new_mode.mode_name = 'flying_car'  # Only ASCII letters and *_* allowed
        # other fields are not mandatory

        # We then explicitly add it to the network
        modes.add(new_mode)

        # we can even keep editing and save it
        new_mode.description = 'this is my new description'
        new_mode.save()
    """

    def __init__(self, net):
        self.__all_modes = []
        self.conn = net.conn  # type: Connection
        self.curr = net.conn.cursor()

        self.__update_list_of_modes()

    def add(self, mode: Mode) -> None:
        """ We add a mode to the project"""
        self.__update_list_of_modes()
        if mode.mode_id in self.__all_modes:
            raise ValueError("Mode already exists in the model")

        self.curr.execute("insert into 'modes'(mode_id, mode_name) Values(?,?)", [mode.mode_id, mode.mode_name])
        self.conn.commit()
        logger.info(f'mode {mode.mode_name}({mode.mode_id}) was added to the project')

        mode.save()

    def drop(self, mode_id: str) -> None:
        """Remove the mode with **mode_id** from the project"""
        try:
            self.curr.execute(f'delete from modes where mode_id="{mode_id}"')
            self.conn.commit()
        except IntegrityError as e:
            logger.error(f'Failed to remove mode {mode_id}. {e.args}')
            raise e
        logger.warning(f'Mode {mode_id} was successfully removed from the database')

    def get(self, mode_id: str) -> Mode:
        """Get a mode from the network by its **mode_id**"""
        self.__update_list_of_modes()
        if mode_id not in self.__all_modes:
            raise ValueError(f'Mode {mode_id} does not exist in the model')
        return Mode(mode_id)

    def all_modes(self) -> dict:
        """Returns a dictionary with all mode objects available in the model. mode_id as key"""
        self.__update_list_of_modes()
        return {x: Mode(x) for x in self.__all_modes}

    def __update_list_of_modes(self) -> None:
        self.curr.execute("select mode_id from 'modes'")
        self.__all_modes = [x[0] for x in self.curr.fetchall()]
