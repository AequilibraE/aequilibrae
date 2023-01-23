from typing import Dict, Any

AGENCY_MULTIPLIER = 10_000_000_000
ROUTE_ID_MULTIPLIER = 1000000
PATTERN_ID_MULTIPLIER = 1000
TRIP_ID_MULTIPLIER = 1

TRANSIT_STOP_RANGE = 1000000
WALK_LINK_RANGE = 30000000
TRANSIT_LINK_RANGE = 20000000
WALK_AGENCY_ID = 1

# 1 for right, -1 for wrong (left)
DRIVING_SIDE = 1


class Constants:
    agencies: Dict[str, Any] = dict()
    srid: Dict[int, int] = dict()
    routes: Dict[int, int] = dict()
    trips: Dict[int, int] = dict()
    patterns: Dict[int, int] = dict()
    pattern_lookup: Dict[int, int] = dict()
    stops: Dict[int, int] = dict()
    fares: Dict[int, int] = dict()
    links: Dict[int, int] = dict()
    transit_links: Dict[int, int] = dict()
