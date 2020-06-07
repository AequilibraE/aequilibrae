import string
from aequilibrae.project.database_connection import database_connection


class LinkType:
    """A link_type object represents a single record in the *link_types* table"""
    __alowed_characters = string.ascii_letters + '_'

    def __init__(self, link_type_id: str) -> None:
        if link_type_id is None:
            raise ValueError('Link type IDs cannot be None')

        if len(link_type_id) != 1 or link_type_id not in string.ascii_letters:
            raise ValueError('Link Type IDs must be a single ascii character')
        conn = database_connection()
        curr = conn.cursor()

        curr.execute('pragma table_info(link_types)')
        table_struct = curr.fetchall()
        self.__fields = [x[1] for x in table_struct]
        self.__original__ = {}

        # data for the link_type
        curr.execute(f"select * from 'link_types' where link_type_id='{link_type_id}'")
        dt = curr.fetchone()
        if dt is None:
            # if the link_type was not found, we return a new one
            for k in self.__fields:
                self.__dict__[k] = None
                self.__original__[k] = None
            self.__dict__['link_type_id'] = link_type_id
            self.__original__['link_type_id'] = link_type_id
        else:
            for k, v in zip(self.__fields, dt):
                self.__dict__[k] = v
                self.__original__[k] = v
        conn.close()

    def __setattr__(self, instance, value) -> None:
        if instance == 'link_type' and value is None:
            raise ValueError('link_type cannot be None')

        if instance == 'link_type_id':
            raise ValueError('Changing a link_type_id is not supported. Create a new one and delete this one')
        else:
            self.__dict__[instance] = value

    def save(self):
        if self.link_type_id not in self.__alowed_characters:
            raise ValueError('link_type_id needs to be a ascii letter')

        for letter in self.link_type:
            if letter not in self.__alowed_characters:
                raise ValueError('link_type can only contain letters and "_"')

        conn = database_connection()
        curr = conn.cursor()

        curr.execute(f'select count(*) from link_types where link_type_id="{self.link_type_id}"')
        if curr.fetchone()[0] == 0:
            raise ValueError("Link type does not exist in the model. You need to explicitly add it")

        curr.execute('pragma table_info(link_types)')
        table_struct = [x[1] for x in curr.fetchall()]

        for key, value in self.__dict__.items():
            if key in table_struct and key != 'link_type_id':
                v_old = self.__original__.get(key, None)
                if value != v_old and value is not None:
                    self.__original__[key] = value
                    curr.execute(f"update 'link_types' set '{key}'=? where link_type_id='{self.link_type_id}'", [value])
        conn.commit()
        conn.close()
