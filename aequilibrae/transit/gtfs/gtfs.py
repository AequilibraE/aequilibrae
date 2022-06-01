import os
import numpy as np
from .agency import Agency
from .calendar_dates import CalendarDates
from .stop import Stop
from .route import Route
from .parse_csv import parse_csv


class GTFS:
    """
    Reader for GTFS (from https://developers.google.com/transit/gtfs/reference/)

    Provides a memory container for GTFS that can be:

       * Passed to transit assignment algorithms in memory
       * Edited and saved back to disk
       * Displayed in a GIS environment
    """

    def __init__(self):
        self.database = None
        self.source_folder = None
        self.agency = Agency()
        self.trips = None
        self.num_routes = None
        self.routes = {}
        self.stops = {}
        self.calendar_dates = {}
        self.available_files = {}
        self.shapes = {}
        self.schedule_exceptions = None

    def load_from_file(self, file_path, save_db=False, memory_db=False):
        raise NotImplementedError

    def load(self, path_to_folder):
        self.source_folder = path_to_folder

        self.load_agency()
        self.load_stops()
        self.load_routes()
        self.load_trips()
        self.load_stop_times()
        self.load_calendar()
        self.load_calendar_dates()
        self.load_shapes()

        self.get_routes_shapes()

    def load_agency(self) -> None:
        agency_file = os.path.join(self.source_folder, "agency.txt")
        self.available_files["agency.txt"] = True
        data = parse_csv(agency_file)
        if not len(data):
            return
        # TODO: Transfer to the database style
        self.agency.email = str(data["agency_id"][0])
        self.agency.name = str(data["agency_name"][0])
        self.agency.url = str(data["agency_url"][0])
        self.agency.timezone = str(data["agency_timezone"][0])
        self.agency.lang = str(data["agency_lang"][0])
        self.agency.phone = str(data["agency_phone"][0])
        del data

    def load_stops(self):
        stops_file = os.path.join(self.source_folder, "stops.txt")
        self.available_files["stops.txt"] = True
        data = parse_csv(stops_file)

        # Iterate over all the stops and puts them in the stops dictionary
        for i in range(data.shape[0]):
            stop = Stop()
            # Required fields
            stop.id = data["stop_id"][i]
            stop.name = data["stop_name"][i]
            stop.lat = data["stop_lat"][i]
            stop.lon = data["stop_lon"][i]

            # optional fields
            available_fields = data.dtype.names
            if "stop_code" in available_fields:
                stop.code = data["stop_code"][i]
            if "stop_desc" in available_fields:
                stop.desc = data["stop_desc"][i]
            if "zone_id" in available_fields:
                stop.zone_id = data["zone_id"][i]
            if "stop_url" in available_fields:
                stop.url = data["stop_url"][i]
            if "zone_id" in available_fields:
                stop.zone_id = data["zone_id"][i]
            if "location_type" in available_fields:
                stop.location_type = data["location_type"][i]
            if "parent_station" in available_fields:
                stop.parent_station = data["parent_station"][i]
            if "timezone" in available_fields:
                stop.timezone = data["timezone"][i]
            if "wheelchair_boarding" in available_fields:
                stop.wheelchair_boarding = data["wheelchair_boarding"][i]

            self.stops[stop.id] = stop
        del data

    def load_routes(self):
        routes_file = os.path.join(self.source_folder, "routes.txt")
        self.available_files["routes.txt"] = True
        data = parse_csv(routes_file)

        # Iterate over all the stops and puts them in the stops dictionary
        for i in range(data.shape[0]):
            r = Route()
            # Required fields
            r.id = data["route_id"][i]
            r.short_name = data["route_short_name"][i]
            r.long_name = data["route_long_name"][i]
            r.type = data["route_type"][i]

            # optional fields
            available_fields = data.dtype.names
            if "agency_id" in available_fields:
                r.agency_id = data["agency_id"][i]
            if "route_desc" in available_fields:
                r.desc = data["route_desc"][i]
            if "route_url" in available_fields:
                r.url = data["route_url"][i]
            if "route_color" in available_fields:
                r.color = data["route_color"][i]
            if "route_text_color" in available_fields:
                r.text_color = data["route_text_color"][i]
            if "route_sort_order" in available_fields:
                r.sort_order = data["route_sort_order"][i]
            self.routes[r.id] = r

        del data

    def load_trips(self):
        trips_file = os.path.join(self.source_folder, "trips.txt")
        self.available_files["trips.txt"] = True

        self.trips = parse_csv(trips_file)

    def load_stop_times(self):
        stop_times_file = os.path.join(self.source_folder, "stop_times.txt")
        self.available_files["stop_times.txt"] = True
        self.stop_times = parse_csv(stop_times_file)

    def load_calendar(self):
        raise NotImplementedError

    def load_calendar_dates(self):
        agency_file = os.path.join(self.source_folder, "calendar_dates.txt")
        if not os.path.isfile(agency_file):
            self.available_files["calendar_dates.txt"] = False
            return

        self.available_files["calendar_dates.txt"] = True
        data = parse_csv(agency_file)
        all_exceptions = []
        for i in range(data.shape[0]):
            cd = CalendarDates()
            # Required fields
            cd.service_id = data["service_id"][i]
            cd.date = data["date"][i]
            cd.exception_type = data["exception_type"][i]
            all_exceptions.append(cd.service_id)
            self.calendar_dates[i] = cd
        self.schedule_exceptions = set(all_exceptions)
        del all_exceptions
        del data

    def load_shapes(self):
        # TODO: Add the info from field "shape_dist_traveled"
        shapes_file = os.path.join(self.source_folder, "shapes.txt")
        if not os.path.isfile(shapes_file):
            self.available_files["shapes.txt"] = False
            return

        self.available_files["shapes.txt"] = True
        data = parse_csv(shapes_file)

        all_shapes = list(np.unique(data["shape_id"]))

        for shp in all_shapes:
            trace = data[data["shape_id"] == shp]
            trace = np.sort(trace, order=["shape_pt_sequence"])
            coords = np.core.defchararray.add(trace["shape_pt_lon"].astype(str), " ")
            coords = np.core.defchararray.add(coords, trace["shape_pt_lat"].astype(str))
            coords = ", ".join(list(coords))
            self.shapes[shp] = '"LINESTRING(' + coords + ')"'

    def get_routes_shapes(self):
        for rt in self.routes.keys():
            trips = self.trips[self.trips["route_id"] == rt]["shape_id"]
            if self.available_files["shapes.txt"]:
                self.routes[rt].shapes = {t: self.shapes[t] for t in trips}
            # else:
            #     for t in trips:
            #         stop_times = self.stop_times[self.stop_times["trip_id"] == t]

    def get_routes_stops(self):
        raise NotImplementedError
