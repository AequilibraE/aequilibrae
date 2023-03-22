import string
from .safe_class import SafeClass


class LinkType(SafeClass):
    """A link_type object represents a single record in the *link_types* table"""

    __alowed_characters = string.ascii_letters + "_"

    def delete(self):
        conn = self.connect_db()
        curr = conn.cursor()
        curr.execute(f'DELETE FROM link_types where link_type_id="{self.link_type_id}"')
        conn.commit()
        del self

    def save(self):
        conn = self.connect_db()

        try:
            sql = f'select count(*) from link_types where link_type_id="{self.link_type_id}"'
            if conn.execute(sql).fetchone()[0] == 0:
                data = [self.link_type_id, self.link_type]
                conn.execute("Insert into link_types (link_type_id, link_type) values(?,?)", data)

            for key, value in self.__dict__.items():
                if key != "link_type_id" and key in self.__original__:
                    v_old = self.__original__.get(key, None)
                    if value != v_old and value is not None:
                        self.__original__[key] = value
                        conn.execute(
                            f"update 'link_types' set '{key}'=? where link_type_id='{self.link_type_id}'", [value]
                        )
        finally:
            conn.commit()
            conn.close()

    def __setattr__(self, instance, value) -> None:
        if instance == "link_type":
            if isinstance(value, str):
                if not len(value):
                    raise ValueError("link_type cannot be zero-length")
                for letter in value:
                    if letter not in self.__alowed_characters:
                        raise ValueError('link_type can only contain letters and "_"')
            else:
                raise ValueError("link_type must be string")
        if instance == "link_type_id":
            raise ValueError("Changing a link_type_id is not supported. Create a new one and delete this")
        else:
            self.__dict__[instance] = value
