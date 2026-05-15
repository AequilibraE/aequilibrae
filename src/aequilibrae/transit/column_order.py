from collections import OrderedDict

column_order = {
    "agency.txt": OrderedDict(
        [
            ("agency_id", str),
            ("agency_name", str),
            ("agency_url", str),
            ("agency_timezone", str),
            ("agency_lang", str),
            ("agency_phone", str),
            ("agency_fare_url", str),
            ("agency_email", str),
        ]
    ),
    "routes.txt": OrderedDict(
        [
            ("route_id", str),
            ("route_short_name", str),
            ("route_long_name", str),
            ("route_desc", str),
            ("route_type", int),
            # ("route_url", str),
            # ("route_color", str),
            # ("route_text_color", str),
            # ("route_sort_order", int),
            # ("agency_id", str),
        ]
    ),
    "trips.txt": OrderedDict(
        [
            ("route_id", str),
            ("service_id", str),
            ("trip_id", str),
            # ("trip_headsign", str),
            # ("trip_short_name", str),
            # ("block_id", str),
            ("shape_id", str),
            ("direction_id", int),
            # ("wheelchair_accessible", int),
            # ("bikes_allowed", int),
        ]
    ),
    "stop_times.txt": OrderedDict(
        [
            ("trip_id", str),
            ("arrival_time", str),
            ("departure_time", str),
            ("stop_id", str),
            ("stop_sequence", int),
            # ("stop_headsign", str),
            # ("pickup_type", int),
            # ("shape_dist_traveled", float),
            # ("timepoint", int),
        ]
    ),
    "calendar.txt": OrderedDict(
        [
            ("service_id", str),
            ("monday", int),
            ("tuesday", int),
            ("wednesday", int),
            ("thursday", int),
            ("friday", int),
            ("saturday", int),
            ("sunday", int),
            ("start_date", str),
            ("end_date", str),
        ]
    ),
    "calendar_dates.txt": OrderedDict([("service_id", str), ("date", str), ("exception_type", int)]),
    "fare_attributes.txt": OrderedDict(
        [
            ("fare_id", str),
            ("price", float),
            ("currency_type", str),
            ("payment_method", int),
            ("transfers", int),
            # ("agency_id", str),
            ("transfer_duration", float),
        ]
    ),
    "fare_rules.txt": OrderedDict(
        [("fare_id", str), ("route_id", str), ("origin_id", str), ("destination_id", str), ("contains_id", str)]
    ),
    "frequencies.txt": OrderedDict(
        [
            ("trip_id", str),
            ("start_time", str),
            ("end_time", str),
            ("headway_secs", str),
            # ("exact_times", int)
        ]
    ),
    "transfers.txt": OrderedDict(
        [("from_stop_id", str), ("to_stop_id", str), ("transfer_type", int), ("min_transfer_time", int)]
    ),
    "feed_info.txt": OrderedDict(
        [
            ("feed_publisher_name", str),
            ("feed_publisher_url", str),
            ("feed_lang", str),
            ("feed_start_date", str),
            ("feed_end_date", str),
            ("feed_version", str),
        ]
    ),
    "stops.txt": OrderedDict(
        [
            ("stop_id", str),
            # ("stop_code", str),
            ("stop_name", str),
            ("stop_desc", str),
            ("stop_lat", float),
            ("stop_lon", float),
            ("stop_street", str),
            ("zone_id", str),
            # ("stop_url", str),
            # ("location_type", int),
            ("parent_station", str),
            # ("stop_timezone", str),
            # ("wheelchair_boarding", int),
        ]
    ),
    "shapes.txt": OrderedDict(
        [
            ("shape_id", str),
            ("shape_pt_lat", float),
            ("shape_pt_lon", float),
            ("shape_pt_sequence", int),
            # ("shape_dist_traveled", float),
        ]
    ),
}
