from contextlib import closing
from sqlite3 import Connection

from aequilibrae.transit.constants import constants, WALK_AGENCY_ID
from aequilibrae.transit.functions.transit_connection import transit_connection
from aequilibrae.transit.transit_elements.basic_element import BasicPTElement


class Agency(BasicPTElement):
    """Transit Agency to load into the database

    :Database class members:

    * agency_id (:obj:`int`): ID for the transit agency
    * agency (:obj:`str`): Name of the transit agency
    * feed_date (:obj:`str`): Date for the transit feed using in the import
    * service_date (:obj:`str`): Date for the route services being imported
    * description (:obj:`str`): Description of the feed"""

    def __init__(self):
        self.agency = ""
        self.feed_date = ""
        self.service_date = ""
        self.description = 0
        self.agency_id = self.__get_agency_id()

    def save_to_database(self, conn: Connection) -> None:
        """Saves route to the database"""

        data = [self.agency_id, self.agency, self.feed_date, self.service_date, self.description]
        sql = """insert into agencies (agency_id, agency, feed_date, service_date, description)
                 values (?, ?, ?, ?, ?);"""
        conn.execute(sql, data)
        conn.commit()

    def __get_agency_id(self):
        with closing(transit_connection()) as conn:
            sql = "Select coalesce(max(distinct(agency_id)), 0) from agencies where agency_id<?;"
            data = [x[0] for x in conn.execute(sql, [WALK_AGENCY_ID])]

        c = constants()
        c.agencies["agencies"] = max(c.agencies.get("agencies", 1) + 1, data[0] + 1)
        return c.agencies["agencies"]
