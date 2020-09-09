import string
from .network.safe_class import SafeClass
from aequilibrae.project.database_connection import database_connection


class Zone(SafeClass):
    def __init__(self, data_set: dict, zoning):
        super().__init__(data_set)
        self.__zoning = zoning

    def delete(self):
        conn = database_connection()
        curr = conn.cursor()
        curr.execute(f'DELETE FROM zones where zone_id="{self.zone_id}"')
        conn.commit()
        self.__zoning._remove_zone(self.zone_id)
        del self

    def save(self):
        conn = database_connection()
        curr = conn.cursor()

        curr.execute(f'select count(*) from zones where zone_id="{self.zone_id}"')
        if curr.fetchone()[0] == 0:
            data = [self.zone_id, self.geometry.wkb]
            curr.execute('Insert into zones (zone_id, geometry) values(?, ST_Multi(GeomFromWKB(?, 4326)))', data)

        for key, value in self.__dict__.items():
            if key != 'zone_id' and key in self.__original__:
                v_old = self.__original__.get(key, None)
                if value != v_old and value is not None:
                    self.__original__[key] = value
                    if key == 'geometry':
                        sql = "update 'zones' set geometry=ST_Multi(GeomFromWKB(?, 4326)) where zone_id=?"
                        curr.execute(sql, [value.wkb, self.zone_id])
                    else:
                        curr.execute(f"update 'zones' set '{key}'=? where zone_id=?", [value, self.zone_id])
        conn.commit()
        conn.close()
