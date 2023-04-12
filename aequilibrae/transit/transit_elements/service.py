from typing import List
from aequilibrae.transit.date_tools import create_days_between, day_of_week


class Service:
    """Transit service built with data from calendar.txt and calendar_dates.txt from GTFS

    * service_id (:obj:`str`):
    * monday (:obj:`int`): Flag if the route runs on mondays (1 for **True**, 0 for **False**)
    * tuesday (:obj:`int`): Flag if the route runs on tuesdays (1 for **True**, 0 for **False**)
    * wednesday (:obj:`int`): Flag if the route runs on wednesdays (1 for **True**, 0 for **False**)
    * thursday (:obj:`int`): Flag if the route runs on thursdays (1 for **True**, 0 for **False**)
    * friday (:obj:`int`): Flag if the route runs on fridays (1 for **True**, 0 for **False**)
    * saturday (:obj:`int`): Flag if the route runs on saturdays (1 for **True**, 0 for **False**)
    * sunday (:obj:`int`): Flag if the route runs on sundays (1 for **True**, 0 for **False**)
    * start_date (:obj:`str`): Start date for this service
    * end_date (:obj:`str`): End date for this service
    * dates (:obj:`List[str]`): List of all dates for which this service is active between its start and end dates
    """

    def __init__(self) -> None:
        self.service_id = ""
        self.monday = 0
        self.tuesday = 0
        self.wednesday = 0
        self.thursday = 0
        self.friday = 0
        self.saturday = 0
        self.sunday = 0
        self.start_date = ""
        self.end_date = ""

        # Not part of GTFS
        self.dates = []  # type: List[str]

    def _populate(self, record: tuple, headers: list) -> None:
        for key, value in zip(headers, record):
            if key not in self.__dict__.keys():
                raise KeyError(f"{key} field in calendar.txt is unknown field for that file on GTFS")
            self.__dict__[key] = value

        if self.end_date < self.start_date:
            raise ValueError(f"Service {self.service_id} has start date after end date")

        days = [self.monday, self.tuesday, self.wednesday, self.thursday, self.friday, self.saturday, self.sunday]
        dates = create_days_between(self.start_date, self.end_date)

        for date in dates:
            if days[day_of_week(date)]:
                self.dates.append(date)
