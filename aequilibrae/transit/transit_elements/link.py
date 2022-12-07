from sqlite3 import Connection
from shapely.geometry import LineString, Point
from shapely.ops import substring

from aequilibrae.transit.constants import constants, TRANSIT_LINK_RANGE


class Link:
    """Transit link element to feed into Transit_links

    :Database class members:

        * transit_link (:obj:`int`): ID of the transit link (updated when inserted in the database)
        * from_node (:obj:`str`): Origin of the transit connection
        * to_node (:obj:`str`): Destination of the transit connection
        * pair (:obj:`str`): Identifier of the stop pair as FROM_ID##TO_ID. For identification only
        * geo (:obj:`LineString`): Geometry of the transit link as direct connection between stops
        * length (:obj:`float`): Link length measured directly from the geometry object
        * type (:obj:`int`): Route type (mode) for this transit link
        * srid (:obj:`int`): srid of our working database
    """

    def __init__(self, srid) -> None:
        """
        :Args:
            *srid* (:obj:`int`): srid of our working database
        """

        self.__dict__["from_node"] = ""
        self.__dict__["to_node"] = ""
        self.__dict__["pattern_id"] = -1
        self.__dict__["geo"] = None
        self.transit_link = -1
        self.pattern_id = -1
        self.from_node = ""
        self.to_node = ""
        self.key = "##"
        self.geo = None  # type: LineString
        self.length = -1
        self.type = -1
        self.srid = srid

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        self.__dict__["key"] = f"{self.from_node}##{self.to_node}##{self.pattern_id}"
        if self.geo is not None:
            self.__dict__["length"] = self.geo.length

    def build_geo(self, from_point: Point, to_point: Point, gtfs_shape: LineString, previous_end: Point):
        geo = LineString([from_point, to_point])
        if gtfs_shape is not None:
            fpos = gtfs_shape.project(from_point)
            tpos = gtfs_shape.project(to_point)
            if fpos < tpos:
                geo2 = substring(gtfs_shape, fpos, tpos)
                geo = geo if geo2.length / geo.length > 2 else geo2

        self.geo = geo if geo.touches(previous_end) else LineString([previous_end] + list(geo.coords))

    def save_to_database(self, conn: Connection, commit=True) -> None:
        """Saves Transit link to the database"""

        data = [
            self.transit_link,
            self.pattern_id,
            self.from_stop,
            self.to_stop,
            self.length,
            self.type,
            self.geo.wkb,
            self.srid,
        ]
        sql = """insert into route_links (transit_link, pattern_id, from_stop, to_stop, "length", "type", geo)
                                            values (?, ?, ?, ?, ?, ?, GeomFromWKB(?, ?));"""
        conn.execute(sql, data)
        if commit:
            conn.commit()

    def get_link_id(self):
        c = constants()
        self.transit_link = c.transit_links.get(1, TRANSIT_LINK_RANGE) + 1
        c.transit_links[1] = self.transit_link
