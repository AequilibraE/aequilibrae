import numpy as np
import os
from collections import OrderedDict
import codecs
from agency import Agency
from calendar_dates import CalendarDates
from stop import Stop
from route import Route
from gtfs_sqlite_db import create_gtfsdb
import copy

class GTFS:
    """
     Reader for GTFS (from https://developers.google.com/transit/gtfs/reference/)

     .

     Objective
     _________
     To provide a memory container for GTFS that can be:
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
        pass

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

    def load_agency(self):
        agency_file = os.path.join(self.source_folder, 'agency.txt')
        self.available_files['agency.txt'] = True
        data = self.open(agency_file)
        #TODO: Transfer to the database style
        self.agency.email = data['agency_id']
        self.agency.name = data['agency_name']
        self.agency.url = data['agency_url']
        self.agency.timezone = data['agency_timezone']
        self.agency.lang = data['agency_lang']
        self.agency.phone = data['agency_phone']
        del(data)

    def load_stops(self):
        stops_file = os.path.join(self.source_folder, 'stops.txt')
        self.available_files['stops.txt'] = True
        data = self.open(stops_file)

        # Iterate over all the stops and puts them in the stops dictionary
        for i in range(data.shape[0]):
            stop = Stop()
            # Required fields
            stop.id = data['stop_id'][i]
            stop.name = data['stop_name'][i]
            stop.lat = data['stop_lat'][i]
            stop.lon = data['stop_lon'][i]

            # optional fields
            available_fields = data.dtype.names
            if 'stop_code' in available_fields: stop.code = data['stop_code'][i]
            if 'stop_desc' in available_fields: stop.desc = data['stop_desc'][i]
            if 'zone_id' in available_fields: stop.zone_id = data['zone_id'][i]
            if 'stop_url' in available_fields: stop.url = data['stop_url'][i]
            if 'zone_id' in available_fields: stop.zone_id = data['zone_id'][i]
            if 'location_type' in available_fields: stop.location_type = data['location_type'][i]
            if 'parent_station' in available_fields: stop.parent_station = data['parent_station'][i]
            if 'timezone' in available_fields: stop.timezone = data['timezone'][i]
            if 'wheelchair_boarding' in available_fields: stop.wheelchair_boarding = data['wheelchair_boarding'][i]

            self.stops[stop.id] = stop
        del(data)

    def load_routes(self):
        routes_file = os.path.join(self.source_folder, 'routes.txt')
        self.available_files['routes.txt'] = True
        data = self.open(routes_file)

        # Iterate over all the stops and puts them in the stops dictionary
        for i in range(data.shape[0]):
            r = Route()
            # Required fields
            r.id = data['route_id'][i]
            r.short_name = data['route_short_name'][i]
            r.long_name = data['route_long_name'][i]
            r.type = data['route_type'][i]

            # optional fields
            available_fields = data.dtype.names
            if 'agency_id' in available_fields: r.agency_id = data['agency_id'][i]
            if 'route_desc' in available_fields: r.desc = data['route_desc'][i]
            if 'route_url' in available_fields: r.url = data['route_url'][i]
            if 'route_color' in available_fields: r.color = data['route_color'][i]
            if 'route_text_color' in available_fields: r.text_color = data['route_text_color'][i]
            if 'route_sort_order' in available_fields: r.sort_order = data['route_sort_order'][i]
            self.routes[r.id] = r

        del data

    def load_trips(self):
        trips_file = os.path.join(self.source_folder, 'trips.txt')
        self.available_files['trips.txt'] = True

        self.trips = self.open(trips_file)

    def load_stop_times(self):
        stop_times_file = os.path.join(self.source_folder, 'stop_times.txt')
        self.available_files['stop_times.txt'] = True
        self.stop_times = self.open(stop_times_file)

    def load_calendar(self):
        pass

    def load_calendar_dates(self):
        agency_file = os.path.join(self.source_folder, 'calendar_dates.txt')
        if not os.path.isfile(agency_file):
            self.available_files['calendar_dates.txt'] = False
            return

        self.available_files['calendar_dates.txt'] = True
        data = self.open(agency_file)
        all_exceptions = []
        for i in range(data.shape[0]):
            cd = CalendarDates()
            # Required fields
            cd.service_id = data['service_id'][i]
            cd.date = data['date'][i]
            cd.exception_type = data['exception_type'][i]
            all_exceptions.append(cd.service_id)
            self.calendar_dates[i] = cd
        self.schedule_exceptions = set(all_exceptions)
        del all_exceptions
        del data

    def load_shapes(self):
        # TODO: Add the info from field "shape_dist_traveled"
        shapes_file = os.path.join(self.source_folder, 'shapes.txt')
        if not os.path.isfile(shapes_file):
            self.available_files['shapes.txt'] = False
            return

        self.available_files['shapes.txt'] = True
        data = self.open(shapes_file)

        all_shapes = list(np.unique(data['shape_id']))

        for shp in all_shapes:
            trace = data[data['shape_id']==shp]
            trace = np.sort(trace,order=['shape_pt_sequence'])
            coords = np.core.defchararray.add(trace['shape_pt_lon'].astype(str), ' ')
            coords = np.core.defchararray.add(coords, trace['shape_pt_lat'].astype(str))
            coords = ', '.join(list(coords))
            self.shapes[shp] = '"LINESTRING(' + coords + ')"'

    def get_routes_shapes(self):
        for rt in self.routes.keys():
            trips = self.trips[self.trips['route_id']==rt]['shape_id']
            if self.available_files['shapes.txt']:
                self.routes[rt].shapes = {t: self.shapes[t] for t in trips}
            else:
                for t in trips:
                    stop_times = self.stop_times[self.stop_times['trip_id']==t]


    def get_routes_stops(self):
        pass

    @staticmethod
    def open(file_name, column_order=False):
        # Read the stops and cleans the names of the columns
        data = np.genfromtxt(file_name, delimiter=',', names=True, dtype=None,)
        content = [str(unicode(x.strip(codecs.BOM_UTF8), 'utf-8')) for x in data.dtype.names]
        data.dtype.names = content
        if column_order:
            col_names = [x for x in column_order.keys() if x in content]
            data = data[col_names]

            # Define sizes for the string variables
            column_order = copy.deepcopy(column_order)
            for c in col_names:
                if column_order[c] is str:
                    if data[c].dtype.char.upper() == "S":
                        column_order[c] = data[c].dtype
                    else:
                        column_order[c] = "S16"

            new_data_dt = [(f, column_order[f]) for f in col_names]

            if int(data.shape.__len__())> 0:
                new_data = np.array(data, new_data_dt)
            else:
                new_data = data


        else:
            new_data = data
        return new_data
