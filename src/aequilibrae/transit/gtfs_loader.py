import hashlib
import zipfile
import logging
from copy import deepcopy
from datetime import datetime
from os.path import basename, splitext
from typing import Dict

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString

from aequilibrae.transit.column_order import column_order
from aequilibrae.transit.date_tools import create_days_between, format_date, to_seconds
from aequilibrae.transit.functions.get_srid import get_srid
from aequilibrae.transit.parse_csv import parse_csv
from aequilibrae.transit.route_types import lookup_by_route_type
from aequilibrae.transit.transit_elements import Agency, Fare, FareRule, Route, Service, Stop, Trip
from aequilibrae.utils.aeq_signal import SIGNAL, simple_progress
from aequilibrae.utils.geo_utils import haversine
from aequilibrae.utils.interface.worker_thread import WorkerThread

logger = logging.getLogger(__name__)

SERVICE_DAY = 86400


class GTFSReader(WorkerThread):
    signal = SIGNAL(object)

    """Loader for GTFS data. Not meant to be used directly by the user"""

    def __init__(self, conn):
        super().__init__(None)

        self.__capacities__ = {}
        self.__pces__ = {}
        self.__max_speeds__ = {}
        self.feed_date = ""
        self.agency = Agency(conn)
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
        self.srid = get_srid()
        self.transformer = Transformer.from_crs("epsg:4326", f"epsg:{self.srid}", always_xy=False)

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
        logger.info(f"Loading data for {service_date} from the {ag_id} GTFS feed. This may take some time")

        self.__load_date()

    def __load_date(self):
        logger.debug("Starting __load_date")
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
        logger.info("Starting deconflict_stop_times")

        msg = "De-conflicting stop times (Step: 6/12)"
        repairs = np.zeros(3, dtype=np.int64)
        for route in simple_progress(self.trips, self.signal, msg):
            max_speeds = lookup_by_route_type(self.__max_speeds__, self.routes[route].route_type, pd.DataFrame([]))
            for pattern in self.trips[route]:  # type: Trip
                for trip in self.trips[route][pattern]:
                    logger.debug(f"De-conflicting stops for route/trip {route}/{trip.trip}")
                    stop_times = self.stop_times[trip.trip]
                    if stop_times.shape[0] != len(trip.stops):
                        logger.error(f"Trip {trip.trip_id} has a different number of stop_times than actual stops.")

                    repairs += self.__deconflict_trip(trip, stop_times, max_speeds)
        repaired = ["departures before their arrival", "run times of zero or less", "runs above the maximum speed"]
        for count, problem in zip(repairs, repaired, strict=True):
            if count:
                logger.warning(f"{count} {problem} were repaired")

    def __deconflict_trip(self, trip: Trip, stop_times: pd.DataFrame, max_speeds: pd.DataFrame) -> np.ndarray:
        """Repairs negative dwell, non-positive run times and runs above the maximum speeds, and returns their counts"""
        arrivals = stop_times.arrival_time.to_numpy(dtype=np.int64)
        departures = stop_times.departure_time.to_numpy(dtype=np.int64)
        source_time = np.zeros(arrivals.shape[0], dtype=np.int64)

        # Clock times that go back by most of a day cross midnight (23:58, then 00:02 for 24:02)
        events = np.column_stack([arrivals, departures]).ravel()
        events[1:] += SERVICE_DAY * np.cumsum(np.diff(events) < -SERVICE_DAY / 2)
        arrivals, departures = events[0::2], events[1::2]

        negative_dwell = departures < arrivals
        dwell = np.maximum(departures - arrivals, 0)
        run = arrivals[1:] - (arrivals[:-1] + dwell[:-1])
        tied = run <= 0
        source_time[1:][tied] = 1
        run = np.maximum(run, 1)

        fast = np.zeros_like(run, dtype=bool)
        if max_speeds.shape[0] > 0:
            lons, lats = np.array([[self.stops[x].geo.x, self.stops[x].geo.y] for x in trip.stops]).T
            distances = haversine(lons[:-1], lats[:-1], lons[1:], lats[1:])
            max_speed = np.full(distances.shape[0], max_speeds.speed.max(), dtype=float)
            for r in max_speeds.itertuples():
                max_speed[(distances >= r.min_distance) & (distances < r.max_distance)] = r.speed

            min_run = np.ceil(distances / max_speed).astype(np.int64)
            fast = run < min_run
            source_time[1:][fast] = 2
            run = np.maximum(run, min_run)

        arrivals = arrivals[0] + np.concatenate([[0], np.cumsum(dwell[:-1] + run)])
        stop_times["arrival_time"] = trip.arrivals = arrivals
        stop_times["departure_time"] = trip.departures = arrivals + dwell
        stop_times["source_time"] = trip.source_time = source_time
        return np.array([negative_dwell.sum(), tied.sum(), fast.sum()])

    def __load_fare_data(self):
        logger.debug("Starting __load_fare_data")
        fareatttxt = "fare_attributes.txt"
        self.fare_attributes = {}
        self.signal.emit(["set_text", "Loading fare data (Step: 7/12)"])
        if fareatttxt in self.zip_archive.namelist():
            logger.debug('  Loading "fare_attributes" table')

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

        logger.debug('  Loading "fare_rules" table')

        with self.zip_archive.open(farerltxt, "r") as file:
            farerl = parse_csv(file, column_order[farerltxt])
        self.data_arrays[farerltxt] = farerl

        for line in range(farerl.shape[0]):
            data = tuple(farerl[line][list(column_order[farerltxt].keys())])
            fr = FareRule()
            fr.populate(data, ["fare", "route", "origin", "destination", "contains"])
            # Without its route, the rule would apply to every route
            if fr.fare not in self.fare_attributes or (fr.route is not None and fr.route not in self.routes):
                logger.warning(f"Fare rule {data} refers to a fare or route absent from the feed and was skipped")
                continue
            fr.fare_id = self.fare_attributes[fr.fare].fare_id
            if fr.route is not None:
                fr.route_id = self.routes[fr.route].route_id
            fr.agency_id = self.agency.agency_id
            self.fare_rules.append(fr)

    def __load_shapes_table(self):
        logger.debug("Starting __load_shapes_table")

        logger.debug("    Loading route shapes")
        self.shapes.clear()
        shapestxt = "shapes.txt"
        if shapestxt not in self.zip_archive.namelist():
            return

        with self.zip_archive.open(shapestxt, "r") as file:
            shapes = parse_csv(file, column_order[shapestxt])

        all_shape_ids = np.unique(shapes["shape_id"]).tolist()

        self.data_arrays[shapestxt] = shapes
        lats, lons = self.transformer.transform(shapes[:]["shape_pt_lat"], shapes[:]["shape_pt_lon"])
        shapes["shape_pt_lat"], shapes["shape_pt_lon"] = lats, lons

        for shape_id in simple_progress(all_shape_ids, self.signal, "Loading shapes (Step: 4/12)"):
            items = shapes[shapes["shape_id"] == shape_id]
            items = items[np.argsort(items["shape_pt_sequence"])]
            shape = LineString(list(zip(items["shape_pt_lon"], items["shape_pt_lat"], strict=True)))
            self.shapes[shape_id] = shape

    def __load_trips_table(self):
        logger.debug("Starting __load_trips_table")

        trip_replacements = self.__load_frequencies()

        logger.debug('    Loading "trips" table')
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
            logger.error(msg)
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
            if trip.trip in trip_replacements:
                all_trips = [deepcopy(trip) for _ in trip_replacements[trip.trip]]
                for new_trip, new_trip_str in zip(all_trips, trip_replacements[trip.trip], strict=True):
                    new_trip.trip = new_trip_str

            for trip in all_trips:
                stop_times = self.stop_times.get(trip.trip, [])
                if len(stop_times) < 2:
                    logger.warning(f"Trip {trip.trip} had less than two stops, so we skipped it.")
                    continue

                trip.stops = list(stop_times.stop_id.values)
                pattern_key = "\x1f".join([trip.route, *stop_times.stop.values])
                trip.pattern_hash = hashlib.md5(pattern_key.encode()).hexdigest()
                trip.arrivals = list(stop_times.arrival_time.values)
                trip.departures = list(stop_times.departure_time.values)
                trip.source_time = list(stop_times.source_time.values)
                logger.debug(f"{trip.trip} has {len(trip.stops)} stops")
                trip._stop_based_shape = LineString([self.stops[x].geo for x in trip.stops])
                # trip.shape = self.shapes.get(trip.shape)
                trip.pce = self.routes[trip.route].pce
                trip.seated_capacity = self.routes[trip.route].seated_capacity
                trip.total_capacity = self.routes[trip.route].total_capacity
                self.trips[trip.route] = self.trips.get(trip.route, {})
                self.trips[trip.route][trip.pattern_hash] = self.trips[trip.route].get(trip.pattern_hash, [])
                self.trips[trip.route][trip.pattern_hash].append(trip)

    def __load_frequencies(self):
        logger.debug("Starting __load_frequencies")

        trip_replacements = {}
        freqtxt = "frequencies.txt"
        if freqtxt in self.zip_archive.namelist():
            logger.debug('    Loading "frequencies" table')

            with self.zip_archive.open(freqtxt, "r") as file:
                freqs = parse_csv(file, column_order[freqtxt])
            self.data_arrays[freqtxt] = freqs
            templates = {}
            for line in range(freqs.shape[0]):
                trip, start_time, end_time, headway = list(freqs[line][list(column_order[freqtxt].keys())])
                if trip not in self.stop_times and trip not in templates:
                    self.__fail(f"trip id {trip} in frequency table has no corresponding entry in trips")

                headway = int(headway)
                if headway <= 0:
                    self.__fail(f"Trip {trip} has non-positive headway on table frequencies.txt")

                if trip not in templates:
                    templates[trip] = self.stop_times.pop(trip)
                template = templates[trip]

                for departure in range(int(to_seconds(start_time)), int(to_seconds(end_time)), headway):
                    shift = departure - template.departure_time.values[0]
                    new_trip_str = f"{trip}-{departure}"
                    while new_trip_str in self.stop_times or new_trip_str in templates:
                        new_trip_str += "-copy"
                    self.stop_times[new_trip_str] = template.assign(
                        arrival_time=template.arrival_time + shift, departure_time=template.departure_time + shift
                    )
                    trip_replacements.setdefault(trip, []).append(new_trip_str)
        return trip_replacements

    def __load_stop_times(self):
        logger.debug("Starting __load_stop_times")

        self.stop_times.clear()
        logger.debug('    Loading "stop times" table')
        stoptimestxt = "stop_times.txt"
        with self.zip_archive.open(stoptimestxt, "r") as file:
            stoptimes = parse_csv(file, column_order[stoptimestxt])
        self.data_arrays[stoptimestxt] = stoptimes

        trips = stoptimes["trip_id"].astype(str)
        sequences = stoptimes["stop_sequence"].astype(str)
        key = np.char.add(np.char.add(trips, "##"), sequences)
        if np.unique(key).shape[0] < stoptimes.shape[0]:
            self.__fail("There are repeated stop_sequences for a single trip_id on stop_times.txt")

        df = pd.DataFrame(stoptimes)
        df["arrival_time"] = _gtfs_seconds(df.arrival_time)
        df["departure_time"] = _gtfs_seconds(df.departure_time)

        df["arrival_time"] = df.arrival_time.fillna(df.departure_time)
        df["departure_time"] = df.departure_time.fillna(df.arrival_time)
        df.columns = ["stop" if x == "stop_id" else x for x in df.columns]

        stops = [s.stop for s in self.stops.values()]
        stop_ids = [s.stop_id for s in self.stops.values()]
        stop_list = pd.DataFrame({"stop": stops, "stop_id": stop_ids})
        unknown = df.stop[~df.stop.isin(stop_list.stop)].unique()
        if unknown.shape[0]:
            logger.warning(f"Stop times for stops absent from stops.txt were dropped: {', '.join(map(str, unknown))}")
        df = df.merge(stop_list, on="stop")
        df = df.sort_values(["trip_id", "stop_sequence"]).reset_index(drop=True)

        # Consecutive events at the same stop are one stop, dwelling from the first arrival to the last departure
        new_stop = (df.stop != df.stop.shift()) | (df.trip_id != df.trip_id.shift())
        df = df.groupby(new_stop.cumsum()).agg(dict.fromkeys(df.columns, "first") | {"departure_time": "last"})

        # GTFS requires times at the first and last stops of a trip, and the stops in between take the previous time
        ends = (df.trip_id != df.trip_id.shift()) | (df.trip_id != df.trip_id.shift(-1))
        untimed = df.trip_id[ends & df.departure_time.isna()].unique()
        if untimed.shape[0]:
            logger.warning(f"Trips without a first or last stop time were skipped: {', '.join(map(str, untimed))}")
            df = df[~df.trip_id.isin(untimed)]
        times = df.departure_time.groupby(df.trip_id).ffill()
        df = df.fillna({"arrival_time": times, "departure_time": times})
        df = df.astype({"arrival_time": np.int64, "departure_time": np.int64}).assign(source_time=0)

        msg = "Loading stop times (Step: 3/12)"
        for trip_id, data in simple_progress(df.groupby(df["trip_id"]), self.signal, msg):
            self.stop_times[trip_id] = data.assign(stop_sequence=np.arange(data.shape[0]))

    def __load_stops_table(self):
        logger.debug("Starting __load_stops_table")

        logger.debug('    Loading "stops" table')
        self.stops = {}
        stopstxt = "stops.txt"
        with self.zip_archive.open(stopstxt, "r") as file:
            stops = parse_csv(file, column_order[stopstxt])
        self.data_arrays[stopstxt] = stops

        if np.unique(stops["stop_id"]).shape[0] < stops.shape[0]:
            self.__fail("There are repeated stop IDs in stops.txt")

        lats, lons = self.transformer.transform(stops[:]["stop_lat"], stops[:]["stop_lon"])
        stops["stop_lat"], stops["stop_lon"] = lats, lons

        for line in simple_progress(stops, self.signal, "Loading stops (Step: 2/12)"):
            s = Stop(self.agency.agency_id, line, stops.dtype.names)
            s.agency = self.agency.agency
            s.srid = self.srid
            s.get_node_id()
            self.stops[s.stop_id] = s

    def __load_routes_table(self):
        logger.debug("Starting __load_routes_table")

        logger.debug('    Loading "routes" table')
        self.routes = {}
        routetxt = "routes.txt"
        with self.zip_archive.open(routetxt, "r") as file:
            routes = parse_csv(file, column_order[routetxt])
        self.data_arrays[routetxt] = routes

        if np.unique(routes["route_id"]).shape[0] < routes.shape[0]:
            self.__fail("There are repeated route IDs in routes.txt")

        # Extended route types take the defaults of their basic type unless they have their own
        routes = pd.DataFrame(routes)
        default_cap = self.__capacities__.get("other", [None, None])
        caps = [lookup_by_route_type(self.__capacities__, rt, default_cap) for rt in routes.route_type]
        default_pce = self.__pces__.get("other", 2.0)
        routes = routes.assign(
            seated_capacity=[cap[0] for cap in caps],
            total_capacity=[cap[1] for cap in caps],
            srid=self.srid,
            pce=[lookup_by_route_type(self.__pces__, rt, default_pce) for rt in routes.route_type],
        )

        for _i, line in simple_progress(routes.iterrows(), self.signal, "Loading routes (Step: 1/12)"):
            r = Route(self.agency.agency_id)
            r.populate(line.values, routes.columns)
            self.routes[r.route] = r

    def __load_feed_calendar(self):
        logger.debug("Starting __load_feed_calendar")
        self.services.clear()

        has_cal, has_caldate = True, True

        self.signal.emit(["set_text", "Loading feed calendar"])
        caltxt = "calendar.txt"
        if caltxt in self.zip_archive.namelist():
            logger.debug('    Loading "calendar" table')
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
                logger.warning('"calendar.txt" file is empty')
                has_cal = False
        else:
            logger.warning(f"{caltxt} not available in this feed")
            has_cal = False

        caldatetxt = "calendar_dates.txt"
        if caldatetxt not in self.zip_archive.namelist():
            logger.warning(f"{caldatetxt} not available in this feed")
            has_caldate = False

        if not has_cal and not has_caldate:
            raise FileNotFoundError('Missing "calendar" and "calendar_dates" in this feed')

        if not has_caldate:
            return

        logger.debug('    Loading "calendar dates" table')
        with self.zip_archive.open(caldatetxt, "r") as file:
            caldates = parse_csv(file, column_order[caldatetxt])

        if caldates.shape[0] == 0:
            logger.warning('"calendar_dates.txt" file is empty')
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
                    logger.debug(msg.format(service.service_id))
            elif exception_type == 2:
                if sd in service.dates:
                    _ = service.dates.remove(sd)
                else:
                    exception_inconsistencies += 1
                    msg = "ignoring service ({}) removal on a day from which the service was absent"
                    logger.debug(msg.format(service.service_id))
            else:
                self.__fail(f"illegal service exception type. {service.service_id}")

        if exception_inconsistencies:
            logger.info("    Minor inconsistencies found between calendar.txt and calendar_dates.txt")

    def __fail(self, msg: str) -> None:
        logger.error(msg)
        raise Exception(msg)


def _gtfs_seconds(times: pd.Series) -> pd.Series:
    parts = times.astype(str).str.extract(r"^\s*(\d+):(\d+)(?::(\d+))?\s*$").astype(float)
    return parts[0] * 3600 + parts[1] * 60 + parts[2].fillna(0)
