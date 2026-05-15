import hashlib
import zipfile
from copy import deepcopy
from datetime import datetime
from io import TextIOWrapper
from os.path import splitext, basename
from typing import Any, Dict, cast

import pandas as pd
from pandas.errors import EmptyDataError
from pyproj import Transformer
from shapely.geometry import LineString

from aequilibrae.context import get_logger
from aequilibrae.transit.column_order import column_order
from aequilibrae.transit.date_tools import to_seconds, create_days_between, format_date
from aequilibrae.transit.functions.get_srid import get_srid
from aequilibrae.transit.transit_elements import Fare, Agency, FareRule, Service, Trip, Stop, Route
from aequilibrae.utils.aeq_signal import SIGNAL, simple_progress
from aequilibrae.utils.interface.worker_thread import WorkerThread


def _clean_string_series(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
        .str.strip()
    )


def _coerce_gtfs_column(series: pd.Series, expected_type):
    cleaned = _clean_string_series(series)

    if expected_type is str:
        return cleaned.fillna("")

    numeric = cleaned.replace("", pd.NA)
    if expected_type is int:
        numeric = numeric.astype("string").str.split(".", n=1, regex=False).str[0]
        return pd.to_numeric(numeric, errors="raise").astype("Int64")

    if expected_type is float:
        return pd.to_numeric(numeric, errors="raise").astype("Float64")

    return cleaned


def _time_series_to_seconds(series: pd.Series, field_name: str) -> pd.Series:
    cleaned = _clean_string_series(series).replace("", pd.NA)
    try:
        return cleaned.map(to_seconds, na_action="ignore").astype("Int64")
    except ValueError as err:
        raise ValueError(f"Invalid GTFS time found in field {field_name}: {err}") from err


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
        self.routes: Dict[str, Route] = {}
        self.trips: Dict[str, Dict[str, list[Trip]]] = {}
        self.stops: Dict[int, Stop] = {}
        self.stop_times = {}
        self.shapes = {}
        self.trip_data = {}
        self.fare_rules = []
        self.fare_attributes = {}
        self.feed_dates = []
        self.gtfs_tables = {}
        self.srid = get_srid()
        self.transformer = Transformer.from_crs("epsg:4326", f"epsg:{self.srid}", always_xy=False)
        self.logger = get_logger()

    def __read_gtfs_table(self, file_name: str) -> pd.DataFrame:
        schema = column_order[file_name]

        try:
            with self.zip_archive.open(file_name, "r") as raw_file:
                csv_file = TextIOWrapper(raw_file, encoding="utf-8-sig", newline="")
                data = pd.read_csv(cast(Any, csv_file), sep=",", dtype="string", keep_default_na=False)
        except EmptyDataError:
            data = pd.DataFrame(columns=list(schema.keys()))

        data.columns = [col.lower().strip() for col in data.columns]
        for column_name in schema:
            if column_name not in data.columns:
                data[column_name] = "" if schema[column_name] is str else pd.NA

        data = data.loc[:, list(schema.keys())].copy()
        if not data.empty:
            cleaned = data.apply(_clean_string_series)
            data = cleaned.loc[~cleaned.eq("").all(axis=1)].copy()

        for column_name, expected_type in schema.items():
            data[column_name] = _coerce_gtfs_column(data[column_name], expected_type)

        return data.reset_index(drop=True)

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
                        stop_times.loc[stop_times.arrival_time == 0, "arrival_time"] = pd.NA
                        stop_times.loc[:, "arrival_time"] = stop_times.arrival_time.ffill()
                    diffs = stop_times.arrival_time.diff().iloc[1:].to_numpy(dtype=int)

                    stop_geos = [self.stops[x].geo for x in trip.stops]
                    distances = pd.Series([x.distance(y) for x, y in zip(stop_geos[:-1], stop_geos[1:], strict=True)])

                    times = stop_times.arrival_time.to_numpy(copy=True, dtype=int)
                    source_time = pd.Series(0, index=stop_times.index, dtype="int64")

                    if times[-1] == times[-2]:
                        self.logger.info(f"De-conflicting stops for route/trip {route}/{trip.trip}")
                        self.logger.info("    Had conflicting stop times in its end")
                        times[-1] += 1
                        source_time.iloc[-1] = 1
                        diffs = pd.Series(times).diff().iloc[1:].to_numpy(dtype=int)

                    to_override = stop_times.index[1:][pd.Series(diffs).eq(0)].tolist()
                    if to_override:
                        self.logger.info(f"De-conflicting stops for route/trip {route}/{trip.trip}")
                        self.logger.info("     Had consecutive stops with the same timestamp")
                        for i in to_override:
                            position = stop_times.index.get_loc(i)
                            source_time.iloc[position] = 1
                            times[position:] += 1
                        diffs = pd.Series(times).diff().iloc[1:].to_numpy(dtype=int)

                    if max_speeds.shape[0] > 0:
                        speeds = distances / pd.Series(diffs)
                        df = pd.DataFrame(
                            {
                                "speed": speeds,
                                "max_speed": max_speeds.speed.max(),
                                "dist": distances,
                                "elapsed_time": diffs,
                                "add_time": pd.Series(0, index=distances.index, dtype="int64"),
                                "source_time": source_time.iloc[1:].to_numpy(dtype=int),
                            }
                        )

                        for _, r in max_speeds.iterrows():
                            df.loc[(df.dist >= r.min_distance) & (df.dist < r.max_distance), "max_speed"] = r.speed

                        to_fix = df.index[df.max_speed < df.speed].tolist()
                        if to_fix:
                            self.logger.debug(f"     Trip {trip.trip} had {len(to_fix)} segments too fast")
                            total_fast += len(to_fix)
                            df.loc[to_fix[0] :, "source_time"] = 2
                            for i in to_fix:
                                df.loc[i:, "add_time"] += (
                                    df.elapsed_time[i] * (df.speed[i] / df.max_speed[i] - 1)
                                ).astype(int)

                            source_time.iloc[1:] = df.source_time.to_numpy(dtype=int)
                            times[1:] += df.add_time.to_numpy(dtype=int)

                    assert min(times[1:] - times[:-1]) > 0
                    stop_times.loc[:, "arrival_time"] = times[:].astype(int)
                    stop_times.loc[:, "departure_time"] = times[:].astype(int)
                    stop_times.loc[:, "source_time"] = source_time.to_numpy(dtype=int)
                    trip.arrivals = stop_times.arrival_time.to_numpy(copy=True)
                    trip.departures = stop_times.departure_time.to_numpy(copy=True)

        if total_fast:
            self.logger.warning(f"There were a total of {total_fast} segments that were too fast and were corrected")

    def __load_fare_data(self):
        self.logger.debug("Starting __load_fare_data")
        fareatttxt = "fare_attributes.txt"
        self.fare_attributes = {}
        self.signal.emit(["set_text", "Loading fare data (Step: 7/12)"])
        if fareatttxt in self.zip_archive.namelist():
            self.logger.debug('  Loading "fare_attributes" table')

            fareatt = self.__read_gtfs_table(fareatttxt)
            self.gtfs_tables[fareatttxt] = fareatt

            for row in fareatt.to_dict(orient="records"):
                data = (
                    row["fare_id"],
                    row["price"],
                    row["currency_type"],
                    row["payment_method"],
                    row["transfers"],
                    row["transfer_duration"],
                )
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

        farerl = self.__read_gtfs_table(farerltxt)
        self.gtfs_tables[farerltxt] = farerl

        for row in farerl.to_dict(orient="records"):
            data = (row["fare_id"], row["route_id"], row["origin_id"], row["destination_id"], row["contains_id"])
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

        shapes = self.__read_gtfs_table(shapestxt)
        all_shape_ids = shapes["shape_id"].drop_duplicates().tolist()

        self.gtfs_tables[shapestxt] = shapes
        lats, lons = self.transformer.transform(shapes["shape_pt_lat"].tolist(), shapes["shape_pt_lon"].tolist())
        shapes["shape_pt_lat"] = lats
        shapes["shape_pt_lon"] = lons

        for shape_id in simple_progress(all_shape_ids, self.signal, "Loading shapes (Step: 4/12)"):
            items = shapes.loc[shapes["shape_id"] == shape_id].sort_values("shape_pt_sequence")
            shape = LineString(list(zip(items["shape_pt_lon"], items["shape_pt_lat"], strict=True)))
            self.shapes[shape_id] = shape

    def __load_trips_table(self):
        self.logger.debug("Starting __load_trips_table")

        trip_replacements = self.__load_frequencies()

        self.logger.debug('    Loading "trips" table')
        tripstxt = "trips.txt"
        trips_array = self.__read_gtfs_table(tripstxt)
        self.gtfs_tables[tripstxt] = trips_array

        if trips_array["trip_id"].duplicated().any():
            self.__fail("There are repeated trip IDs in trips.txt")

        stp_tm = self.gtfs_tables["stop_times.txt"]
        diff = trips_array.loc[~trips_array["trip_id"].isin(stp_tm["trip_id"]), "trip_id"].drop_duplicates()
        if not diff.empty:
            diff = ",".join(diff.astype(str).tolist())
            msg = f"There are IDs in trips.txt without any stop on stop_times.txt -> {diff}"
            self.logger.error(msg)
            raise Exception(msg)

        incal = pd.Index(self.services.keys(), dtype="string")
        diff = trips_array.loc[~trips_array["service_id"].isin(incal), "service_id"].drop_duplicates()
        if not diff.empty:
            diff = ",".join(diff.astype(str).tolist())
            self.__fail(f"There are service IDs in trips.txt that are absent in the calendar -> {diff}")

        self.trips = {str(x): {} for x in trips_array["route_id"].drop_duplicates().tolist()}

        records = list(trips_array.itertuples(index=False, name=None))
        for line in simple_progress(records, self.signal, "Loading trips (Step: 5/12)"):
            trip = Trip()
            trip._populate(line, list(trips_array.columns))
            trip.route_id = self.routes[trip.route].route_id
            trip.shape = self.shapes.get(trip.shape_id, trip.shape)
            replacement_ids = trip_replacements.get(trip.trip)
            all_trips = [trip]
            if replacement_ids is not None:
                all_trips = []
                for replacement_id in replacement_ids:
                    replacement_trip = deepcopy(trip)
                    replacement_trip.trip = replacement_id
                    all_trips.append(replacement_trip)

            for trip in all_trips:
                stop_times = self.stop_times.get(trip.trip, pd.DataFrame())
                if stop_times.shape[0] < 2:
                    self.logger.warning(f"Trip {trip.trip} had less than two stops, so we skipped it.")
                    continue

                cleaner = stop_times.assign(seqkey=stop_times.stop.shift(-1).fillna("") + "#####" + stop_times.stop)
                cleaner.drop_duplicates(["seqkey"], inplace=True, keep="first")
                stop_times = cleaner.drop(columns=["seqkey"])
                stop_times.loc[:, "arrival_time"] = stop_times.arrival_time.astype(int)
                stop_times.loc[:, "departure_time"] = stop_times.departure_time.astype(int)
                self.stop_times[trip.trip] = stop_times
                trip.stops = stop_times.stop_id.tolist()
                m = hashlib.md5()
                m.update(trip.route.encode())
                m.update(stop_times.stop.astype(str).str.cat().encode())

                trip.pattern_hash = m.hexdigest()
                trip.arrivals = stop_times.arrival_time.tolist()
                trip.departures = stop_times.departure_time.tolist()
                trip.source_time = stop_times.source_time.tolist()
                self.logger.debug(f"{trip.trip} has {len(trip.stops)} stops")
                trip_points = [self.stops[x].geo for x in trip.stops if self.stops[x].geo is not None]
                trip._stop_based_shape = LineString(trip_points)
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

            freqs = self.__read_gtfs_table(freqtxt)
            self.gtfs_tables[freqtxt] = freqs
            for row in freqs.to_dict(orient="records"):
                trip = row["trip_id"]
                start_time = row["start_time"]
                end_time = row["end_time"]
                headway = row["headway_secs"]
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
                    new_trip_str = f"{trip}-{int(new_trip.arrival_time.iloc[0])}"
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
        stoptimes = self.__read_gtfs_table(stoptimestxt)
        self.gtfs_tables[stoptimestxt] = stoptimes

        for col in ["arrival_time", "departure_time"]:
            stoptimes[col] = _time_series_to_seconds(stoptimes[col], col)

        if stoptimes.duplicated(["trip_id", "stop_sequence"]).any():
            self.__fail("There are repeated stop_sequences for a single trip_id on stop_times.txt")

        df = stoptimes.copy()
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
        for trip_id, data in simple_progress(list(df.groupby("trip_id", sort=False)), self.signal, msg):
            data = data.copy()
            data.loc[:, "stop_sequence"] = range(data.shape[0])
            self.stop_times[trip_id] = data

    def __load_stops_table(self):
        self.logger.debug("Starting __load_stops_table")

        self.logger.debug('    Loading "stops" table')
        self.stops = {}
        stopstxt = "stops.txt"
        stops = self.__read_gtfs_table(stopstxt)
        self.gtfs_tables[stopstxt] = stops

        if stops["stop_id"].duplicated().any():
            self.__fail("There are repeated stop IDs in stops.txt")

        lats, lons = self.transformer.transform(stops["stop_lat"].tolist(), stops["stop_lon"].tolist())
        stops["stop_lat"] = lats
        stops["stop_lon"] = lons

        stop_records = list(stops.itertuples(index=False, name=None))
        for line in simple_progress(stop_records, self.signal, "Loading stops (Step: 2/12)"):
            s = Stop(self.agency.agency_id, line, list(stops.columns))
            s.agency = self.agency.agency
            s.srid = self.srid
            s.get_node_id()
            self.stops[s.stop_id] = s

    def __load_routes_table(self):
        self.logger.debug("Starting __load_routes_table")

        self.logger.debug('    Loading "routes" table')
        self.routes = {}
        routetxt = "routes.txt"
        routes = self.__read_gtfs_table(routetxt)
        self.gtfs_tables[routetxt] = routes

        if routes["route_id"].duplicated().any():
            self.__fail("There are repeated route IDs in routes.txt")

        seated_cap, total_cap = self.__capacities__.get("other", [None, None])
        routes = routes.assign(seated_capacity=seated_cap, total_capacity=total_cap, srid=self.srid)
        for route_type, cap in self.__capacities__.items():
            if route_type == "other":
                continue
            routes.loc[routes.route_type == route_type, ["seated_capacity", "total_capacity"]] = cap

        default_pce = self.__pces__.get("other", 2.0)
        routes = routes.assign(pce=default_pce)
        for route_type, pce in self.__pces__.items():
            if route_type == "other":
                continue
            routes.loc[routes.route_type == route_type, ["pce"]] = pce

        route_records = list(routes.itertuples(index=False, name=None))
        for line in simple_progress(route_records, self.signal, "Loading routes (Step: 1/12)"):
            r = Route(self.agency.agency_id)
            r.populate(line, list(routes.columns))
            self.routes[r.route] = r

    def __load_feed_calendar(self):
        self.logger.debug("Starting __load_feed_calendar")
        self.services.clear()

        has_cal, has_caldate = True, True

        self.signal.emit(["set_text", "Loading feed calendar"])
        caltxt = "calendar.txt"
        if caltxt in self.zip_archive.namelist():
            self.logger.debug('    Loading "calendar" table')
            calendar = self.__read_gtfs_table(caltxt)

            if calendar.shape[0] > 0:
                calendar["start_date"] = calendar["start_date"].map(format_date).map(datetime.fromisoformat)
                calendar["end_date"] = calendar["end_date"].map(format_date).map(datetime.fromisoformat)
                self.gtfs_tables[caltxt] = calendar
                if calendar["service_id"].duplicated().any():
                    self.__fail("There are repeated service IDs in calendar.txt")

                min_date = min(calendar["start_date"].tolist())
                max_date = max(calendar["end_date"].tolist())
                self.feed_dates = create_days_between(min_date, max_date)

                for line in calendar.itertuples(index=False, name=None):
                    service = Service()
                    service._populate(line, list(calendar.columns), True)
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
        caldates = self.__read_gtfs_table(caldatetxt)

        if caldates.shape[0] == 0:
            self.logger.warning('"calendar_dates.txt" file is empty')
            return

        if caldates.shape[0] > 0 and not has_cal:
            min_date = datetime.fromisoformat(format_date(min(caldates["date"].tolist())))
            max_date = datetime.fromisoformat(format_date(max(caldates["date"].tolist())))
            self.feed_dates = create_days_between(min_date, max_date)

        exception_inconsistencies = 0
        for row in caldates.to_dict(orient="records"):
            sd = format_date(row["date"])
            service_id = row["service_id"]
            exception_type = row["exception_type"]

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
