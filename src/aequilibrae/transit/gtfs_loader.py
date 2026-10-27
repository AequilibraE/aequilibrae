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
        total_ties, total_fast = 0, 0
        for route in simple_progress(self.trips, self.signal, msg):
            max_speeds = lookup_by_route_type(self.__max_speeds__, self.routes[route].route_type, pd.DataFrame([]))
            for pattern in self.trips[route]:  # type: Trip
                for trip in self.trips[route][pattern]:
                    logger.debug(f"De-conflicting stops for route/trip {route}/{trip.trip}")
                    stop_times = self.stop_times[trip.trip]
                    if stop_times.shape[0] != len(trip.stops):
                        logger.error(f"Trip {trip.trip_id} has a different number of stop_times than actual stops.")

                    ties, fast = self.__deconflict_trip(trip, stop_times, max_speeds)
                    total_ties += ties
                    total_fast += fast
        if total_ties:
            logger.info(f"There were a total of {total_ties} segments with no (or negative) run time, now 1 s each")
        if total_fast:
            logger.warning(f"There were a total of {total_fast} segments that were too fast and were corrected")

    def __deconflict_trip(self, trip: Trip, stop_times: pd.DataFrame, max_speeds: pd.DataFrame) -> tuple:
        """Repairs the run times between stops of a trip, keeping the scheduled dwell at each stop.

        A run time is from the departure from a stop to the arrival at the next one. Run times of zero (common in
        timetables to the minute) become one second, negative ones are repaired the same way with a warning, and
        those faster than the maximum speed for their distance are slowed down to it. Every change shifts the rest
        of the trip. ``source_time`` is 0 for stops with times as in GTFS, 1 for stops whose times were changed by
        de-conflicting and 2 for those changed by the maximum speeds.

        Returns the number of run times de-conflicted and slowed down.
        """
        arrivals = stop_times.arrival_time.to_numpy(dtype=np.int64)
        departures = stop_times.departure_time.to_numpy(dtype=np.int64)
        source_time = np.zeros(arrivals.shape[0], dtype=np.int64)

        dwell = departures - arrivals
        if (dwell < 0).any():
            logger.warning(f"Trip {trip.trip} departs before arriving at {(dwell < 0).sum()} stops. Dwell set to zero")
            source_time[dwell < 0] = 1
            dwell = np.maximum(dwell, 0)

        run = arrivals[1:] - (arrivals[:-1] + dwell[:-1])
        if (run < 0).any():
            logger.warning(
                f"Trip {trip.trip} arrives at {(run < 0).sum()} stops before leaving the previous one. "
                "Their arrivals were shifted forward, with the rest of the trip"
            )
        ties = run <= 0
        source_time[1:][ties] = 1
        run = np.maximum(run, 1)

        fast = np.zeros_like(ties)
        if max_speeds.shape[0] > 0:
            lons, lats = np.array([[self.stops[x].geo.x, self.stops[x].geo.y] for x in trip.stops]).T
            distances = haversine(lons[:-1], lats[:-1], lons[1:], lats[1:])
            max_speed = np.full(distances.shape[0], max_speeds.speed.max(), dtype=float)
            for r in max_speeds.itertuples():
                max_speed[(distances >= r.min_distance) & (distances < r.max_distance)] = r.speed

            min_run = np.ceil(distances / max_speed).astype(np.int64)
            fast = run < min_run
            if fast.any():
                logger.debug(f"     Trip {trip.trip} had {fast.sum()} segments too fast")
                source_time[1:][fast] = 2
                run = np.maximum(run, min_run)

        new_arrivals = arrivals.copy()
        new_arrivals[1:] = arrivals[0] + np.cumsum(dwell[:-1] + run)
        new_departures = new_arrivals + dwell

        # Changes carry over to the rest of the trip, which keeps the most significant reason it was shifted for
        changed = (new_arrivals != arrivals) | (new_departures != departures)
        source_time = np.where(changed, np.maximum.accumulate(source_time), 0)

        stop_times["arrival_time"] = new_arrivals
        stop_times["departure_time"] = new_departures
        stop_times["source_time"] = source_time
        trip.arrivals = new_arrivals
        trip.departures = new_departures
        trip.source_time = source_time
        return int(ties.sum()), int(fast.sum())

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
            fr.fare_id = self.fare_attributes[fr.fare].fare_id
            if fr.route is not None:
                # Without its route, the rule would apply to every route
                if fr.route not in self.routes:
                    logger.warning(f"Fare rule for fare {fr.fare} refers to route {fr.route}, absent from routes.txt")
                    continue
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
                # A trip in frequencies.txt is a template, replaced by one trip per departure in its windows
                all_trips = []
                for new_trip_str in trip_replacements[trip.trip]:
                    all_trips.append(deepcopy(trip))
                    all_trips[-1].trip = new_trip_str

            for trip in all_trips:
                stop_times = self.stop_times.get(trip.trip, [])
                if len(stop_times) < 2:
                    logger.warning(f"Trip {trip.trip} had less than two stops, so we skipped it.")
                    continue

                trip.stops = list(stop_times.stop_id.values)
                # Delimited, so that stops such as "1", "23" and "12", "3" do not make up the same pattern
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
            windows = {}
            for line in range(freqs.shape[0]):
                trip, start_time, end_time, headway = list(freqs[line][list(column_order[freqtxt].keys())])
                if trip not in self.stop_times:
                    self.__fail(f"trip id {trip} in frequency table has no corresponding entry in trips")

                headway = int(float(headway))
                if headway <= 0:
                    self.__fail(f"Trip {trip} has non-positive headway on table frequencies.txt")

                windows.setdefault(trip, []).append((int(to_seconds(start_time)), int(to_seconds(end_time)), headway))

            # The template trip departs its first stop every headway from the start of each window, up to (but not
            # at) its end. Every window is taken as having exact times
            for trip, trip_windows in windows.items():
                template = self.stop_times.pop(trip)
                trip_replacements[trip] = []
                for start_seconds, end_seconds, headway in sorted(trip_windows):
                    for departure in range(start_seconds, end_seconds, headway):
                        hhmmss = f"{departure // 3600:02d}{departure % 3600 // 60:02d}{departure % 60:02d}"
                        new_trip_str = f"{trip}-{hhmmss}"
                        if new_trip_str in self.stop_times:
                            logger.warning(f"Trip {new_trip_str}, from the frequencies of trip {trip}, is repeated")
                            continue
                        shift = departure - template.departure_time.values[0]
                        new_trip = template.assign(
                            trip_id=new_trip_str,
                            arrival_time=template.arrival_time + shift,
                            departure_time=template.departure_time + shift,
                        )
                        self.stop_times[new_trip_str] = new_trip
                        trip_replacements[trip].append(new_trip_str)
                if not trip_replacements[trip]:
                    logger.warning(f"Trip {trip} has no departures within its windows on frequencies.txt")
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

        # GTFS allows either time to be empty, and both of them at stops that are not timepoints
        df["arrival_time"] = df.arrival_time.fillna(df.departure_time)
        df["departure_time"] = df.departure_time.fillna(df.arrival_time)
        df.columns = ["stop" if x == "stop_id" else x for x in df.columns]

        stops = [s.stop for s in self.stops.values()]
        stop_ids = [s.stop_id for s in self.stops.values()]
        stop_list = pd.DataFrame({"stop": stops, "stop_id": stop_ids})
        df = df.merge(stop_list, on="stop")
        df = df.sort_values(["trip_id", "stop_sequence"]).reset_index(drop=True)

        # Consecutive events at the same stop are a single stop, with a dwell from the first arrival to the last
        # departure. "first" and "last" skip missing times
        new_stop = (df.stop != df.stop.shift()) | (df.trip_id != df.trip_id.shift())
        if not new_stop.all():
            logger.info(f"    Merged {(~new_stop).sum()} stop_times records at the same stop as the previous one")
            aggregation = dict.fromkeys(df.columns, "first") | {"departure_time": "last"}
            df = df.groupby(new_stop.cumsum()).agg(aggregation).reset_index(drop=True)

        # Stops without any time take the departure from the previous stop, which de-conflicting spreads out later
        no_time = df.arrival_time.isna()
        if no_time.any():
            times = df.departure_time.groupby(df.trip_id).ffill()
            times = times.fillna(df.arrival_time.groupby(df.trip_id).bfill())
            df.loc[no_time, "arrival_time"] = times[no_time]
            df.loc[no_time, "departure_time"] = times[no_time]
            if df.arrival_time.isna().any():
                no_times = df.loc[df.arrival_time.isna(), "trip_id"].unique().tolist()
                logger.error(f"Trips without any arrival or departure time on stop_times.txt are skipped: {no_times}")
                df = df.loc[df.arrival_time.notna(), :]

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
    """Converts GTFS times (H:MM:SS, with hours past 24 after midnight) to seconds, with NaN for empty times"""
    parts = times.astype(str).str.extract(r"^\s*(\d+):(\d+)(?::(\d+))?\s*$").astype(float)
    return parts[0] * 3600 + parts[1] * 60 + parts[2].fillna(0)
