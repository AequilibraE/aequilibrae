from typing import Union
from .safe_class import SafeClass
from aequilibrae.project.network.mode import Mode


class Link(SafeClass):
    """A Link object represents a single record in the *links* table

    .. code-block:: python

        >>> from aequilibrae import Project

        >>> proj = Project.from_path("/tmp/test_project")

        >>> all_links = proj.network.links

        # Let's get a mode to work with
        >>> modes = proj.network.modes
        >>> car_mode = modes.get('c')

        # We can just get one link in specific
        >>> link1 = all_links.get(3)
        >>> link2 = all_links.get(17)

        # We can find out which fields exist for the links
        >>> which_fields_do_we_have = link1.data_fields()

        # And edit each one like this
        >>> link1.lanes_ab = 3
        >>> link1.lanes_ba = 2

        # we can drop a mode from the link
        >>> link1.drop_mode(car_mode)  # or link1.drop_mode('c')

        # we can add a mode to the link
        >>> link2.add_mode(car_mode)  # or link2.add_mode('c')

        # Or set all modes at once
        >>> link2.set_modes('cbtw')

        # We can just save the link
        >>> link1.save()
        >>> link2.save()
    """

    def __init__(self, dataset, project):
        super().__init__(dataset, project)
        self.__fields = list(dataset.keys())

        self.__new = dataset["geometry"] is None
        self.__stil_exists = True
        self._table = "links"

    def delete(self):
        """Deletes link from database"""
        conn = self.connect_db()
        curr = conn.cursor()
        curr.execute(f'DELETE FROM links where link_id="{self.link_id}"')
        conn.commit()
        self.__stil_exists = False

    def save(self):
        """Saves link to database"""
        conn = self.connect_db()
        curr = conn.cursor()

        if self.__new:
            data, sql = self._save_new_with_geometry()
        else:
            data, sql = self.__save_existing_link()

        if data:
            curr.execute(sql, data)

        conn.commit()
        conn.close()
        self.__new = False

        for key in self.__original__.keys():
            self.__original__[key] = self.__dict__[key]

    def set_modes(self, modes: str):
        """Sets the modes acceptable for this link

        :Arguments:
            **modes** (:obj:`str`): string with all mode_ids to be assigned to this link
        """

        if not isinstance(modes, str):
            raise ValueError("Modes field needs to be a string")
        if modes == "":
            raise ValueError("Modes field needs to have at least one mode")

        self.__dict__["modes"] = modes

    def add_mode(self, mode: Union[str, Mode]):
        """Adds a new mode to this link

        Raises a warning if mode is already allowed on the link, and fails if mode does not exist

        :Arguments:
            **mode_id** (:obj:`str` or `Mode`): Mode_id of the mode or mode object to be added to the link
        """
        mode_id = self.__validate(mode)

        if mode_id in self.modes:
            self._logger.warning("Mode already active for this link")
            return

        self.__dict__["modes"] += mode_id

    def drop_mode(self, mode: Union[str, Mode]):
        """Removes a mode from this link

        Raises a warning if mode is already NOT allowed on the link, and fails if mode does not exist

        :Arguments:
            **mode_id** (:obj:`str` or `Mode`): Mode_id of the mode or mode object to be removed from the link
        """

        mode_id = self.__validate(mode)

        if mode_id not in self.modes:
            self._logger.warning("Mode already inactive for this link")
            return

        if len(self.modes) == 1:
            raise ValueError("Link needs to have at least one mode")

        self.__dict__["modes"] = self.modes.replace(mode_id, "")

    def data_fields(self) -> list:
        """lists all data fields for the link, as available in the database

        :Returns:
            **data fields** (:obj:`list`): list of all fields available for editing
        """

        return list(self.__original__.keys())

    def _exists(self):
        return self.__stil_exists

    def __validate(self, mode: Union[str, Mode]) -> str:
        if isinstance(mode, Mode):
            mode_id = mode.mode_id
        elif isinstance(mode, str):
            if len(mode) > 1:
                raise ValueError("A mode_id is a single character")
            mode_id = mode
        else:
            raise TypeError("You should provide a mode id (string) or a Mode object")
        return mode_id

    def __save_existing_link(self):
        data = []
        if self.link_id != self.__original__["link_id"]:
            raise ValueError("One cannot change the link_id")

        txts = []
        for key, val in self.__dict__.items():
            if key not in self.__original__:
                continue
            if val != self.__original__[key]:
                if key == "geometry" and val is not None:
                    data.extend([val.wkb, self.__srid__])
                    txts.append("geometry=GeomFromWKB(?, ?)")
                else:
                    data.append(val)
                    txts.append(f'"{key}"=?')

        if not data:
            self._logger.warning(f"Nothing to update for link {self.link_id}")
            return [], ""

        txts = ",".join(txts) + " where link_id=?"
        data.append(self.link_id)
        sql = f"Update Links set {txts}"
        return data, sql

    def __setattr__(self, instance, value) -> None:
        if instance not in self.__dict__ and instance[:1] != "_":
            raise AttributeError(f'"{instance}" is not a valid attribute for a link')
        if instance == "modes":
            self.set_modes(value)
        elif instance == "a_node":
            raise AttributeError("Setting a_node is not allowed")
        elif instance == "b_node":
            raise AttributeError("Setting b_node is not allowed")
        elif instance == "link_id":
            raise ValueError("Changing a link_id is not supported. Create a new one and delete this")
        else:
            self.__dict__[instance] = value
