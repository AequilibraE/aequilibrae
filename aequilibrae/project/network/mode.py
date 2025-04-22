import string


class Mode:
    """A mode object represents a single record in the *modes* table"""

    __alowed_characters = string.ascii_letters + "_"

    def __init__(self, mode_id: str, project) -> None:
        self.project = project
        if mode_id is None:
            raise ValueError("Mode IDs cannot be None")

        if len(mode_id) != 1 or mode_id not in string.ascii_letters:
            raise ValueError("Mode IDs must be a single ascii character")

        with self.project.db_connection as conn:
            table_struct = conn.execute("pragma table_info(modes)").fetchall()
            self.__fields = [x[1] for x in table_struct]
            self.__original__ = {}
            # data for the mode
            dt = conn.execute(f"select * from 'modes' where mode_id='{mode_id}'").fetchone()

        if dt is None:
            # if the mode was not found, we return a new one
            for k in self.__fields:
                self.__dict__[k] = None
                self.__original__[k] = None
            self.__dict__["mode_id"] = mode_id
            self.__original__["mode_id"] = mode_id
        else:
            for k, v in zip(self.__fields, dt):
                self.__dict__[k] = v
                self.__original__[k] = v

    def __setattr__(self, instance, value) -> None:
        if instance == "mode_name" and value is None:
            raise ValueError("mode_name cannot be None")

        if instance == "mode_id":
            raise ValueError("Changing a mode_id is not supported. Create a new one and delete this one")
        else:
            self.__dict__[instance] = value

    def save(self):
        if self.mode_id not in self.__alowed_characters:
            raise ValueError("mode_id needs to be a ascii letter")

        for letter in self.mode_name:
            if letter not in self.__alowed_characters:
                raise ValueError('mode_name can only contain letters and "_"')

        with self.project.db_connection as conn:
            if conn.execute(f'select count(*) from modes where mode_id="{self.mode_id}"').fetchone()[0] == 0:
                raise ValueError("Mode does not exist in the model. You need to explicitly add it")

            table_struct = [x[1] for x in conn.execute("pragma table_info(modes)").fetchall()]

            for key, value in self.__dict__.items():
                if key in table_struct and key != "mode_id":
                    v_old = self.__original__.get(key, None)
                    if value != v_old and value is not None:
                        self.__original__[key] = value
                        conn.execute(f"update 'modes' set '{key}'=? where mode_id='{self.mode_id}'", [value])
