from .safe_class import SafeClass
from aequilibrae.project.database_connection import database_connection
from aequilibrae.project.network.link_types import LinkTypes
from aequilibrae.project.network.modes import Modes
from aequilibrae import logger


class Link(SafeClass):
    """A link object represents a single record in the *links* table"""

    def __init__(self, dataset, link_types: LinkTypes, modes: Modes):
        super().__init__(dataset)
        self.__link_types = link_types
        self.__modes = modes
        self.__fields = list(dataset.keys())

        self.__new = dataset['geometry'] is None

    def delete(self):
        """Deletes link from database"""
        conn = database_connection()
        curr = conn.cursor()
        curr.execute(f'DELETE FROM links where link_id="{self.link_id}"')
        conn.commit()
        del self

    def save(self):
        """Saves link to database"""
        conn = database_connection()
        curr = conn.cursor()

        data = []
        if self.__new:
            up_keys = []
            for key, val in self.__dict__.items():
                if key not in self.__original__ or key == 'geometry':
                    continue
                up_keys.append(f'"{key}"')
                data.append(val)
            markers = ','.join(['?'] * len(up_keys)) + ',GeomFromWKB(?)'
            up_keys.append('geometry')
            data.append(self.geometry.wkb)
            sql = f'Insert into links ({",".join(up_keys)}) values({markers})'
        else:
            if self.link_id != self.__original__['link_id']:
                raise ValueError('One cannot change the link_id')

            txts = []
            for key, val in self.__dict__.items():
                if key not in self.__original__:
                    continue
                if val != self.__original__[key]:
                    if key == 'geometry' and val is not None:
                        data.append(val.wkb)
                    else:
                        data.append(val)
                    txts.append(f'"{key}"=?')

            if not data:
                logger.warn(f'Nothing to update for link {self.link_id}')
                return

            txts = ','.join(txts) + f' where link_id=?'
            data.append(self.link_id)
            sql = f'Update Links set {txts}'

        logger.error(sql)
        curr.execute(sql, data)
        conn.commit()
        conn.close()
        self.__new = False

    def set_modes(self, modes: str):
        """Sets the modes acceptable for this link

        Args:
            *modes* (:obj:`str`): string with all mode_ids to be assigned to this link
        """

        if not isinstance(modes, str):
            raise ValueError('Modes field needs to be a string')
        if modes == '':
            raise ValueError('Modes field needs to have at least one mode')
        all_modes = self.__modes.all_modes()
        missing = [x for x in modes if x not in all_modes]
        if missing:
            raise ValueError(f'Mode(s) {",".join(missing)} do not exist in the model')

        self.__dict__["modes"] = modes

    def add_mode(self, mode_id: str):
        """Adds a new mode to this link

        Raises a warning if mode is already allowed on the link, and fails if mode does not exist

        Args:
            *mode_id* (:obj:`str`): Mode_id of the mode to be added to the link
        """

        if mode_id in self.modes:
            logger.warn('Mode already active for this link')
            return

        if mode_id not in self.__modes.all_modes():
            raise ValueError('Mode does not exist in the model')

        self.__dict__["modes"] += mode_id

    def drop_mode(self, mode_id: str):
        """Removes a mode from this link

        Raises a warning if mode is already NOT allowed on the link, and fails if mode does not exist

        Args:
            *mode_id* (:obj:`str`): Mode_id of the mode to be removed from the link
        """

        if mode_id not in self.modes:
            logger.warn('Mode already inactive for this link')
            return

        if mode_id not in self.__modes.all_modes():
            raise ValueError('Mode does not exist in the model')

        if len(self.modes) == 1:
            raise ValueError('Link needs to have at least one mode')

        self.__dict__['modes'] = self.modes.replace(mode_id, '')

    def fields(self) -> list:
        """lists all data fields for the link, as available in the database

        Returns:
            *data fields* (:obj:`list`): list of all fields available for editing
        """

        return list(self.__original__.keys())

    def __setattr__(self, instance, value) -> None:
        if instance not in self.__dict__ and instance[:1] != "_":
            raise AttributeError(f'"{instance}" is not a valid attribute for a link')
        if instance == 'modes':
            self.set_mode(value)
        elif instance == 'link_type':
            raise NotImplementedError('Setting link_type is a little tricky')
        elif instance == 'link_id':
            raise ValueError('Changing a link_id is not supported. Create a new one and delete this')
        else:
            self.__dict__[instance] = value
