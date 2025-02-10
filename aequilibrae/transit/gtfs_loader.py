import hashlib
import zipfile
from copy import deepcopy
from datetime import datetime
from os.path import splitext, basename
from typing import Dict

import numpy as np
import pandas as pd
import pyproj
from pyproj import Transformer
from shapely.geometry import LineString

from aequilibrae.context import get_logger
from aequilibrae.transit.column_order import column_order
from aequilibrae.transit.constants import AGENCY_MULTIPLIER
from aequilibrae.transit.date_tools import to_seconds, create_days_between, format_date
from aequilibrae.transit.functions.get_srid import get_srid
from aequilibrae.transit.parse_csv import parse_csv
from aequilibrae.transit.transit_elements import Fare, Agency, FareRule, Service, Trip, Stop, Route
from aequilibrae.utils.aeq_signal import SIGNAL, simple_progress
from aequilibrae.utils.interface.worker_thread import WorkerThread


class GTFSReader(WorkerThread):
    signal = SIGNAL(object)

    """Loader for GTFS data. Not meant to be used directly by the user"""

    def __init__(self):
        WorkerThread.__init__(self, None)

        self.__capacities__ = {}
        self.__pces__ = {}
        self.__max_speeds__ = {}
        self.feed_date = ""
        self.agency = Agency()
        self.services = {}
        self.routes: Dict[int, Route] = {}
        self.trips: Dict[int, Dict[Route]] = {}
        self.stops: Dict[int, Stop] = {}
        self.stop_times = {}
        self.shapes = {}
        self.trip_data = {}
        self.fare_rules = []
        self.fare_attributes = {}
        self.feed_dates = []
        self.data_arrays = {}
        self.wgs84 = pyproj.Proj("epsg:4326")
        self.srid = get_srid()
        self.transformer = Transformer.from_crs("epsg:4326", f"epsg:{self.srid}", always_xy=False)
        self.logger = get_logger()

    def set_feed_path(self, file_path):
        """Sets GTFS feed source to be used

        :Arguments:
            **file_path** (:obj:`str`): Full path to the GTFS feed (e.g. 'D:/project/my_gtfs_feed.zip')
        """

        self.archive_dir = file_path
        self.zip_archive = zipfile.ZipFile(self.archive_dir)
        ret = self.zip_archive.testzip()
        if ret is not None:
            self.__fail(f"GTFS feed {file_path} is not valid")

        self.__load_feed_calendar()
        self.zip_archive.close()

        self.feed_date = splitext(basename(file_path))[0]

    def _set_capacities(self, capacities: dict):
        self.__capacities__ = capacities

    def _set_pces(self, pces: dict):
        self.__pces__ = pces

    def _set_maximum_speeds(self, max_speeds: dict):
        self.__max_speeds__ = max_speeds

    def load_data(self, service_date: str):
        """Loads the data for a respective service date.

        :Arguments:
            **service_date** (:obj:`str`): service date. e.g. "2020-04-01".
        """
        ag_id = self.agency.agency
        self.logger.info(f"Loading data for {service_date} from the {ag_id} GTFS feed. This may take some time")

        self.__load_date()

    def __load_date(self):
        self.logger.debug("Starting __load_date")
        self.zip_archive = zipfile.ZipFile(self.archive_dir)

        self.__load_routes_table()

        self.__load_stops_table()

        self.__load_stop_times()

        self.__load_shapes_table()

        self.__load_trips_table()

        self.__deconflict_stop_times()

        self.__load_fare_data()

        self.zip_archive.close()
        self.signal = SIGNAL(object)

    def __deconflict_stop_times(self) -> None:
        self.logger.info("Starting deconflict_stop_times")

        msg = "De-conflicting stop times (Step: 6/12)"
        total_fast = 0
        for route in simple_progress(self.trips, self.signal, msg):
            max_speeds = self.__max_speeds__.get(self.routes[route].route_type, pd.DataFrame([]))
            for pattern in self.trips[route]:  # type: Trip
                for trip in self.trips[route][pattern]:
                    self.logger.debug(f"De-conflicting stops for route/trip {route}/{trip.trip}")
                    stop_times = self.stop_times[trip.trip]
                    if stop_times.shape[0] != len(trip.stops):
                        self.logger.error(
                            f"Trip {trip.trip_id} has a different number of stop_times than actual stops."
                        )

                    if not stop_times.arrival_time.is_monotonic_increasing:
                        stop_times.loc[stop_times.arrival_time == 0, "arrival_time"] = np.nan
                        stop_times.arrival_time.fillna(method="ffill", inplace=True)
                    diffs = np.diff(stop_times.arrival_time.values)

                    stop_geos = [self.stops[x].geo for x in trip.stops]
                    distances = np.array([x.distance(y) for x, y in zip(stop_geos[:-1], stop_geos[1:])])

                    times = np.copy(stop_times.arrival_time.values)
                    source_time = np.zeros_like(stop_times.arrival_time.values)

                    if times[-1] == times[-2]:
                        self.logger.info(f"De-conflicting stops for route/trip {route}/{trip.trip}")
                        self.logger.info("    Had conflicting stop times in its end")
                        times[-1] += 1
                        source_time[-1] = 1
                        diffs = np.diff(times)

                    to_override = np.argwhere(diffs == 0)[:, 0] + 1
                    if to_override.shape[0] > 0:
                        self.logger.info(f"De-conflicting stops for route/trip {route}/{trip.trip}")
                        self.logger.info("     Had consecutive stops with the same timestamp")
                        for i in to_override:
                            source_time[i] = 1
                            times[i:] += 1
                        diffs = np.diff(times)

                    if max_speeds.shape[0] > 0:
                        speeds = distances / diffs
                        df = pd.DataFrame(
                            {
                                "speed": speeds,
                                "max_speed": max_speeds.speed.max(),
                                "dist": distances,
                                "elapsed_time": diffs,
                                "add_time": np.zeros(diffs.shape[0], dtype=int),
                                "source_time": source_time[1:],
                            }
                        )

                        for _, r in max_speeds.iterrows():
                            df.loc[(df.dist >= r.min_distance) & (df.dist < r.max_distance), "max_speed"] = r.speed

                        to_fix = df[df.max_speed < df.speed].index.values
                        if to_fix.shape[0] > 0:
                            self.logger.debug(f"     Trip {trip.trip} had {to_fix.shape[0]} segments too fast")
                            total_fast += to_fix.shape[0]
                            df.loc[to_fix[0] :, "source_time"] = 2
                            for i in to_fix:
                                df.loc[i:, "add_time"] += (
                                    df.elapsed_time[i] * (df.speed[i] / df.max_speed[i] - 1)
                                ).astype(int)

                            source_time[1:] = df.source_time[:]
                            times[1:] += df.add_time[:].astype(int)

                    assert min(times[1:] - times[:-1]) > 0
                    stop_times.arrival_time.values[:] = times[:].astype(int)
                    stop_times.departure_time.values[:] = times[:].astype(int)
                    stop_times.source_time.values[:] = source_time[:].astype(int)
                    trip.arrivals = stop_times.arrival_time.values
                    trip.departures = stop_times.departure_time.values

        if total_fast:
            self.logger.warning(f"There were a total of {total_fast} segments that were too fast and were corrected")

    def __load_fare_data(self):
        self.logger.debug("Starting __load_fare_data")
        fareatttxt = "fare_attributes.txt"
        self.fare_attributes = {}
        self.signal.emit(["set_text", "Loading fare data (Step: 7/12)"])
        if fareatttxt in self.zip_archive.namelist():
            self.logger.debug('  Loading "fare_attributes" table')

            with self.zip_archive.open(fareatttxt, "r") as file:
                fareatt = parse_csv(file, column_order[fareatttxt])
            self.data_arrays[fareatttxt] = fareatt

            for line in range(fareatt.shape[0]):
                data = tuple(fareatt[line][list(column_order[fareatttxt].keys())])
                headers = ["fare_id", "price", "currency", "payment_method", "transfer", "transfer_duration"]
                f = Fare(self.agency.agency_id)
                f.populate(data, headers)
                if f.fare in self.fare_attributes:
                    self.__fail(f"Fare ID {f.fare} for {self.agency.agency} is duplicated")
                self.fare_attributes[f.fare] = f

        farerltxt = "fare_rules.txt"
        self.fare_rules = []
        if farerltxt not in self.zip_archive.namelist():
            return

        self.logger.debug('  Loading "fare_rules" table')

        with self.zip_archive.open(farerltxt, "r") as file:
            farerl = parse_csv(file, column_order[farerltxt])
        self.data_arrays[farerltxt] = farerl

        for line in range(farerl.shape[0]):
            data = tuple(farerl[line][list(column_order[farerltxt].keys())])
            fr = FareRule()
            fr.populate(data, ["fare", "route", "origin", "destination", "contains"])
            fr.fare_id = self.fare_attributes[fr.fare].fare_id
            if fr.route in self.routes:
                fr.route_id = self.routes[fr.route].route_id
            fr.agency_id = self.agency.agency_id
            self.fare_rules.append(fr)

    def __load_shapes_table(self):
        self.logger.debug("Starting __load_shapes_table")

        self.logger.debug("    Loading route shapes")
        self.shapes.clear()
        shapestxt = "shapes.txt"
        if shapestxt not in self.zip_archive.namelist():
            return

        with self.zip_archive.open(shapestxt, "r") as file:
            shapes = parse_csv(file, column_order[shapestxt])

        all_shape_ids = np.unique(shapes["shape_id"]).tolist()

        self.data_arrays[shapestxt] = shapes
        lats, lons = self.transformer.transform(shapes[:]["shape_pt_lat"], shapes[:]["shape_pt_lon"])
        shapes[:]["shape_pt_lat"][:] = lats[:]
        shapes[:]["shape_pt_lon"][:] = lons[:]

        for shape_id in simple_progress(all_shape_ids, self.signal, "Loading shapes (Step: 4/12)"):
            items = shapes[shapes["shape_id"] == shape_id]
            items = items[np.argsort(items["shape_pt_sequence"])]
            shape = LineString(list(zip(items["shape_pt_lon"], items["shape_pt_lat"])))
            self.shapes[shape_id] = shape

    def __load_trips_table(self):
        self.logger.debug("Starting __load_trips_table")

        trip_replacements = self.__load_frequencies()

        self.logger.debug('    Loading "trips" table')
        tripstxt = "trips.txt"
        with self.zip_archive.open(tripstxt, "r") as file:
            trips_array = parse_csv(file, column_order[tripstxt])
        self.data_arrays[tripstxt] = trips_array

        if np.unique(trips_array["trip_id"]).shape[0] < trips_array.shape[0]:
            self.__fail("There are repeated trip IDs in trips.txt")

        stp_tm = self.data_arrays["stop_times.txt"]
        diff = np.setdiff1d(trips_array["trip_id"], stp_tm["trip_id"], assume_unique=False)
        if diff.shape[0] > 0:
            diff = ",".join([str(x) for x in diff.tolist()])
            msg = f"There are IDs in trips.txt without any stop on stop_times.txt -> {diff}"
            self.logger.error(msg)
            raise Exception(msg)

        incal = np.unique(list(self.services.keys()))
        diff = np.setdiff1d(trips_array["service_id"], incal, assume_unique=False)
        if diff.shape[0] > 0:
            diff = ",".join([str(x) for x in diff.tolist()])
            self.__fail(f"There are service IDs in trips.txt that are absent in the calendar -> {diff}")

        self.trips = {str(x): {} for x in np.unique(trips_array["route_id"])}

        for line in simple_progress(trips_array, self.signal, "Loading trips (Step: 5/12)"):
            trip = Trip()
            trip._populate(line, trips_array.dtype.names)
            trip.route_id = self.routes[trip.route].route_id
            trip.shape = self.shapes.get(trip.shape_id, trip.shape)
            all_trips = [trip]
            if trip.trip_id in trip_replacements:
                all_trips = [deepcopy(trip) for _ in range(len(trip_replacements[trip.trip_id]))]

            for trip in all_trips:
                stop_times = self.stop_times.get(trip.trip, [])
                if len(stop_times) < 2:
                    self.logger.warning(f"Trip {trip.trip} had less than two stops, so we skipped it.")
                    continue

                cleaner = stop_times.assign(seqkey=stop_times.stop.shift(-1) + "#####" + stop_times.stop)
                cleaner.drop_duplicates(["seqkey"], inplace=True, keep="first")
                stop_times = cleaner.drop(columns=["seqkey"])
                stop_times.loc[:, "arrival_time"] = stop_times.arrival_time.astype(int)
                stop_times.loc[:, "departure_time"] = stop_times.departure_time.astype(int)
                self.stop_times[trip.trip] = stop_times
                trip.stops = list(stop_times.stop_id.values)
                m = hashlib.md5()
                m.update(trip.route.encode())
                m.update("".join(stop_times.stop.values).encode())

                trip.pattern_hash = m.hexdigest()
                trip.arrivals = list(stop_times.arrival_time.values)
                trip.departures = list(stop_times.departure_time.values)
                trip.source_time = list(stop_times.source_time.values)
                self.logger.debug(f"{trip.trip} has {len(trip.stops)} stops")
                trip._stop_based_shape = LineString([self.stops[x].geo for x in trip.stops])
                # trip.shape = self.shapes.get(trip.shape)
                trip.pce = self.routes[trip.route].pce
                trip.seated_capacity = self.routes[trip.route].seated_capacity
                trip.total_capacity = self.routes[trip.route].total_capacity
                self.trips[trip.route] = self.trips.get(trip.route, {})
                self.trips[trip.route][trip.pattern_hash] = self.trips[trip.route].get(trip.pattern_hash, [])
                self.trips[trip.route][trip.pattern_hash].append(trip)

    def __load_frequencies(self):
        self.logger.debug("Starting __load_frequencies")

        trip_replacements = {}
        freqtxt = "frequencies.txt"
        if freqtxt in self.zip_archive.namelist():
            self.logger.debug('    Loading "frequencies" table')

            with self.zip_archive.open(freqtxt, "r") as file:
                freqs = parse_csv(file, column_order[freqtxt])
            self.data_arrays[freqtxt] = freqs
            for line in range(freqs.shape[0]):
                trip, start_time, end_time, headway = list(freqs[line][list(column_order[freqtxt].keys())])
                if trip not in self.stop_times:
                    self.__fail(f"trip id {trip} in frequency table has no corresponding entry in trips")

                headway = int(headway)
                if headway <= 0:
                    self.__fail(f"Trip {trip} has non-positive headway on table frequencies.txt")

                start_seconds = to_seconds(start_time)
                end_seconds = to_seconds(end_time)

                template = self.stop_times.pop(trip)

                trip_replacements[trip] = []
                steps = int(((end_seconds - start_seconds) / headway) + 1)
                for step in range(steps):
                    shift = step * headway
                    new_trip = template.copy()
                    new_trip_str = f"{trip}-{new_trip.arrival_time.values[0]}".replace(":", "")
                    new_trip.loc[:, "arrival_time"] += shift
                    new_trip.loc[:, "departure_time"] += shift
                    self.stop_times[new_trip_str] = new_trip
                    trip_replacements[trip].append(new_trip_str)
        return trip_replacements

    def __load_stop_times(self):
        self.logger.debug("Starting __load_stop_times")

        self.stop_times.clear()
        self.logger.debug('    Loading "stop times" table')
        stoptimestxt = "stop_times.txt"
        with self.zip_archive.open(stoptimestxt, "r") as file:
            stoptimes = parse_csv(file, column_order[stoptimestxt])
        self.data_arrays[stoptimestxt] = stoptimes

        df = pd.DataFrame(stoptimes)
        for col in ["arrival_time", "departure_time"]:
            df2 = df[col].str.split(":", expand=True)
            df2.fillna(0, inplace=True)
            df2.columns = ["h", "m", "s"]
            df2.loc[df2.h.str.len() < 1, "h"] = 0
            df2.loc[df2.m.str.len() < 1, "m"] = 0
            df2.loc[df2.s.str.len() < 1, "s"] = 0
            df2 = df2.assign(sec=0)
            df2.loc[:, "sec"] = df2.h.astype(int) * 3600 + df2.m.astype(int) * 60 + df2.s.astype(int)
            stoptimes[col] = df2.sec.values

        key = np.core.defchararray.add(stoptimes["trip_id"].astype(str), np.array(["##"] * stoptimes.shape[0]))
        key = np.core.defchararray.add(key, stoptimes["stop_sequence"].astype(str))
        if np.unique(key).shape[0] < stoptimes.shape[0]:
            self.__fail("There are repeated stop_sequences for a single trip_id on stop_times.txt")

        df = pd.DataFrame(stoptimes)

        df.loc[:, "arrival_time"] = df.loc[:, ["arrival_time", "departure_time"]].max(axis=1)
        df.loc[:, "departure_time"] = df.loc[:, "arrival_time"]

        counter = df.shape[0]
        df = df.assign(other_stop=df.stop_id.shift(-1), other_trip=df.trip_id.shift(-1))
        df = df.loc[~((df.other_stop == df.stop_id) & (df.trip_id == df.other_trip)), :]
        counter -= df.shape[0]
        df.drop(columns=["other_stop", "other_trip"], inplace=True)
        df.columns = ["stop" if x == "stop_id" else x for x in df.columns]

        stops = [s.stop for s in self.stops.values()]
        stop_ids = [s.stop_id for s in self.stops.values()]
        stop_list = pd.DataFrame({"stop": stops, "stop_id": stop_ids})
        df = df.merge(stop_list, on="stop")
        df.sort_values(["trip_id", "stop_sequence"], inplace=True)
        df = df.assign(source_time=0)

        msg = "Loading stop times (Step: 3/12)"
        for trip_id, data in simple_progress(df.groupby(df["trip_id"]), self.signal, msg):
            data.loc[:, "stop_sequence"] = np.arange(data.shape[0])
            self.stop_times[trip_id] = data

    def __load_stops_table(self):
        self.logger.debug("Starting __load_stops_table")

        self.logger.debug('    Loading "stops" table')
        self.stops = {}
        stopstxt = "stops.txt"
        with self.zip_archive.open(stopstxt, "r") as file:
            stops = parse_csv(file, column_order[stopstxt])
        self.data_arrays[stopstxt] = stops

        if np.unique(stops["stop_id"]).shape[0] < stops.shape[0]:
            self.__fail("There are repeated stop IDs in stops.txt")

        lats, lons = self.transformer.transform(stops[:]["stop_lat"], stops[:]["stop_lon"])
        stops[:]["stop_lat"][:] = lats[:]
        stops[:]["stop_lon"][:] = lons[:]

        for line in simple_progress(stops, self.signal, "Loading stops (Step: 2/12)"):
            s = Stop(self.agency.agency_id, line, stops.dtype.names)
            s.agency = self.agency.agency
            s.srid = self.srid
            s.get_node_id()
            self.stops[s.stop_id] = s

    def __load_routes_table(self):
        self.logger.debug("Starting __load_routes_table")

        self.logger.debug('    Loading "routes" table')
        self.routes = {}
        routetxt = "routes.txt"
        with self.zip_archive.open(routetxt, "r") as file:
            routes = parse_csv(file, column_order[routetxt])
        self.data_arrays[routetxt] = routes

        if np.unique(routes["route_id"]).shape[0] < routes.shape[0]:
            self.__fail("There are repeated route IDs in routes.txt")

        seated_cap, total_cap = self.__capacities__.get("other", [None, None])
        routes = pd.DataFrame(routes)
        routes = routes.assign(seated_capacity=seated_cap, total_capacity=total_cap, srid=self.srid)
        for route_type, cap in self.__capacities__.items():
            routes.loc[routes.route_type == route_type, ["seated_capacity", "total_capacity"]] = cap

        default_pce = self.__pces__.get("other", 2.0)
        routes = routes.assign(pce=default_pce)
        for route_type, pce in self.__pces__.items():
            routes.loc[routes.route_type == route_type, ["pce"]] = pce

        for i, line in simple_progress(routes.iterrows(), self.signal, "Loading routes (Step: 1/12)"):
            r = Route(self.agency.agency_id)
            r.populate(line.values, routes.columns)
            self.routes[r.route] = r

    def __load_feed_calendar(self):
        self.logger.debug("Starting __load_feed_calendar")
        self.services.clear()

        has_cal, has_caldate = True, True

        self.signal.emit(["set_text", "Loading feed calendar"])
        caltxt = "calendar.txt"
        if caltxt in self.zip_archive.namelist():
            self.logger.debug('    Loading "calendar" table')
            with self.zip_archive.open(caltxt, "r") as file:
                calendar = parse_csv(file, column_order[caltxt])

            if calendar.shape[0] > 0:
                calendar["start_date"] = [datetime.fromisoformat(format_date(i)) for i in calendar["start_date"]]
                calendar["end_date"] = [datetime.fromisoformat(format_date(i)) for i in calendar["end_date"]]
                self.data_arrays[caltxt] = calendar
                if np.unique(calendar["service_id"]).shape[0] < calendar.shape[0]:
                    self.__fail("There are repeated service IDs in calendar.txt")

                min_date = min(calendar["start_date"].tolist())
                max_date = max(calendar["end_date"].tolist())
                self.feed_dates = create_days_between(min_date, max_date)

                for line in calendar:
                    service = Service()
                    service._populate(line, calendar.dtype.names, True)
                    self.services[service.service_id] = service
            else:
                self.logger.warning('"calendar.txt" file is empty')
                has_cal = False
        else:
            self.logger.warning(f"{caltxt} not available in this feed")
            has_cal = False

        caldatetxt = "calendar_dates.txt"
        if caldatetxt not in self.zip_archive.namelist():
            self.logger.warning(f"{caldatetxt} not available in this feed")
            has_caldate = False

        if not has_cal and not has_caldate:
            raise FileNotFoundError('Missing "calendar" and "calendar_dates" in this feed')

        if not has_caldate:
            return

        self.logger.debug('    Loading "calendar dates" table')
        with self.zip_archive.open(caldatetxt, "r") as file:
            caldates = parse_csv(file, column_order[caldatetxt])

        if caldates.shape[0] == 0:
            self.logger.warning('"calendar_dates.txt" file is empty')
            return

        if caldates.shape[0] > 0 and not has_cal:
            min_date = datetime.fromisoformat(format_date(min(caldates["date"].tolist())))
            max_date = datetime.fromisoformat(format_date(max(caldates["date"].tolist())))
            self.feed_dates = create_days_between(min_date, max_date)

        exception_inconsistencies = 0
        svc_id_col = caldates.dtype.names.index("service_id")
        xcpt_tp_col = caldates.dtype.names.index("exception_type")
        for line in caldates:
            sd = format_date(line["date"])
            service_id = line[svc_id_col]
            exception_type = line[xcpt_tp_col]

            if service_id not in self.services:
                s = Service()
                s.service_id = service_id
                self.services[service_id] = s

            service = self.services[service_id]

            if exception_type == 1:
                if sd not in service.dates:
                    service.dates.append(sd)
                else:
                    exception_inconsistencies += 1
                    msg = "ignoring service ({}) addition on a day when the service is already active"
                    self.logger.debug(msg.format(service.service_id))
            elif exception_type == 2:
                if sd in service.dates:
                    _ = service.dates.remove(sd)
                else:
                    exception_inconsistencies += 1
                    msg = "ignoring service ({}) removal on a day from which the service was absent"
                    self.logger.debug(msg.format(service.service_id))
            else:
                self.__fail(f"illegal service exception type. {service.service_id}")

        if exception_inconsistencies:
            self.logger.info("    Minor inconsistencies found between calendar.txt and calendar_dates.txt")

    def __fail(self, msg: str) -> None:
        self.logger.error(msg)
        raise Exception(msg)
